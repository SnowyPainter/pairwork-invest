"""
Event Detector using FT-Transformer
3-class 이벤트 분류 모델: 상승 이벤트(+1), 정상(0), 하락 이벤트(-1)

Based on analysis results:
- Top features for event detection: rel_range, obv, parkinson20, macd, macd_hist, rsi10, atr5, tr14, vol_roc5, vol_z20
- Focus on volatility and volume-based indicators for event detection  
- Multi-class classification: Up Event (+1) vs Normal (0) vs Down Event (-1)
- Temporal split: 2018-2019 train, 2020 validation/test
- Category embeddings for categorical/temporal features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import polars as pl
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import json
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
import warnings
warnings.filterwarnings('ignore')


class FocalLoss(nn.Module):
    """
    Focal Loss for addressing class imbalance
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.alpha, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class EventDataset(Dataset):
    """Event Detection Dataset with Category Embeddings"""
    
    def __init__(self, features: np.ndarray, labels: np.ndarray, 
                 categorical_features: Optional[np.ndarray] = None):
        self.features = torch.FloatTensor(features)
        self.labels = torch.LongTensor(labels)
        self.categorical_features = torch.LongTensor(categorical_features) if categorical_features is not None else None
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.categorical_features is not None:
            return self.features[idx], self.categorical_features[idx], self.labels[idx]
        else:
            return self.features[idx], self.labels[idx]


class FTTransformerBlock(nn.Module):
    """Feature Tokenizer + Transformer Block for Tabular Data with Category Embeddings"""
    
    def __init__(self, n_features: int, d_model: int = 192, n_heads: int = 8, 
                 d_ff: int = 512, dropout: float = 0.25,
                 categorical_dims: Optional[Dict[str, int]] = None):
        super().__init__()
        self.n_features = n_features
        self.d_model = d_model
        self.categorical_dims = categorical_dims or {}
        
        # Feature Tokenizer: Each feature gets its own embedding
        self.feature_tokenizer = nn.Linear(1, d_model)
        self.feature_bias = nn.Parameter(torch.randn(n_features, d_model))
        
        # Category embeddings
        self.category_embeddings = nn.ModuleDict()
        if self.categorical_dims:
            for cat_name, cat_size in self.categorical_dims.items():
                self.category_embeddings[cat_name] = nn.Embedding(cat_size, d_model)
        
        # Positional embeddings for features
        total_features = n_features + len(self.categorical_dims)
        self.feature_embeddings = nn.Parameter(torch.randn(total_features, d_model))
        
        # Transformer layers
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                activation='gelu',
                batch_first=True
            ),
            num_layers=3
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, categorical_x=None):
        # x shape: (batch_size, n_features)
        batch_size = x.shape[0]
        
        # Tokenize numerical features: (batch_size, n_features, 1) -> (batch_size, n_features, d_model)
        x_expanded = x.unsqueeze(-1)  # (batch_size, n_features, 1)
        tokens = self.feature_tokenizer(x_expanded)  # (batch_size, n_features, d_model)
        
        # Add feature-specific bias
        tokens = tokens + self.feature_bias.unsqueeze(0)
        
        # Handle categorical features
        all_tokens = [tokens]
        if categorical_x is not None and self.categorical_dims:
            for i, (cat_name, _) in enumerate(self.categorical_dims.items()):
                cat_tokens = self.category_embeddings[cat_name](categorical_x[:, i])
                cat_tokens = cat_tokens.unsqueeze(1)  # (batch_size, 1, d_model)
                all_tokens.append(cat_tokens)
        
        # Concatenate all tokens
        tokens = torch.cat(all_tokens, dim=1)  # (batch_size, total_features, d_model)
        
        # Add positional embeddings
        tokens = tokens + self.feature_embeddings.unsqueeze(0)
        
        # Apply layer norm and dropout
        tokens = self.layer_norm(tokens)
        tokens = self.dropout(tokens)
        
        # Apply transformer
        output = self.transformer(tokens)  # (batch_size, total_features, d_model)
        
        return output


class EventDetector(nn.Module):
    """
    FT-Transformer based Event Detector
    
    Multi-class classification: Up Event (+1) vs Normal (0) vs Down Event (-1)
    Optimized for detecting volatility events (±5% moves)
    """
    
    def __init__(self, n_features: int, d_model: int = 192, n_heads: int = 8,
                 d_ff: int = 512, dropout: float = 0.25, n_classes: int = 3,
                 categorical_dims: Optional[Dict[str, int]] = None):
        super().__init__()
        
        self.n_features = n_features
        self.d_model = d_model
        self.categorical_dims = categorical_dims or {}
        
        # FT-Transformer backbone
        self.ft_transformer = FTTransformerBlock(
            n_features=n_features,
            d_model=d_model,
            n_heads=n_heads,
            d_ff=d_ff,
            dropout=dropout,
            categorical_dims=categorical_dims
        )
        
        # Calculate total features (numerical + categorical)
        total_features = n_features + len(self.categorical_dims)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model * total_features, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_ff // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff // 2, n_classes)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, categorical_x=None):
        # x shape: (batch_size, n_features)
        
        # Get transformer output
        transformer_output = self.ft_transformer(x, categorical_x)  # (batch_size, total_features, d_model)
        
        # Flatten for classification
        flattened = transformer_output.flatten(1)  # (batch_size, total_features * d_model)
        
        # Classification
        logits = self.classifier(flattened)  # (batch_size, n_classes)
        
        return logits


class EventDetectorTrainer:
    """Training and evaluation utilities for Event Detector"""
    
    def __init__(self, model: EventDetector, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.scaler = StandardScaler()
        self.feature_names = None
        self.training_history = []
    
    def prepare_data(self, df: pl.DataFrame, target_col: str = "label_1d_cls", 
                    feature_cols: Optional[List[str]] = None) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], List[str]]:
        """
        Prepare data for training
        
        Args:
            df: DataFrame with features and labels
            target_col: Target column name
            feature_cols: List of feature columns to use
        
        Returns:
            features, labels, categorical_features, feature_names
        """
                
        # Add categorical features
        categorical_cols = []
        if 'year' in df.columns:
            categorical_cols.append('year')
        if 'month' in df.columns:
            categorical_cols.append('month')
        elif 'date' in df.columns:
            # Extract month from date
            df = df.with_columns(
                pl.col("date").dt.month().alias("month")
            )
            categorical_cols.append('month')
        
        # Add ticker embedding (강추!)
        if 'ticker' in df.columns:
            # Encode tickers as integers
            unique_tickers = df.select("ticker").unique().to_pandas()["ticker"].tolist()
            ticker_to_id = {ticker: i for i, ticker in enumerate(unique_tickers)}
            
            df = df.with_columns(
                pl.col("ticker").map_elements(lambda x: ticker_to_id.get(x, 0), return_dtype=pl.Int32).alias("ticker_id")
            )
            categorical_cols.append('ticker_id')
            
            print(f"[EventDetector] Added ticker embedding: {len(unique_tickers)} unique tickers")
        
        # Convert labels to 3-class: -1, 0, +1 -> 0, 1, 2 for CrossEntropyLoss
        df_processed = df.with_columns([
            (pl.col(target_col) + 1).cast(pl.Int32).alias("label_mapped")  # -1->0, 0->1, +1->2
        ])
        
        # Extract features and labels
        features = df_processed.select(feature_cols).to_pandas().values.astype(np.float32)
        labels = df_processed.select("label_mapped").to_pandas().values.ravel().astype(np.int64)
        
        # Extract categorical features
        categorical_features = None
        if categorical_cols:
            categorical_data = df_processed.select(categorical_cols).to_pandas()
            # Encode categorical features
            for col in categorical_cols:
                if col == 'year':
                    # Years: 2018->0, 2019->1, 2020->2
                    categorical_data[col] = categorical_data[col] - 2018
                elif col == 'month':
                    # Months: 1->0, 2->1, ..., 12->11
                    categorical_data[col] = categorical_data[col] - 1
                # ticker_id는 이미 0부터 시작하는 정수로 인코딩됨
            categorical_features = categorical_data.values.astype(np.int64)
        
        # Handle NaN values
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Store feature names and categorical info
        self.feature_names = feature_cols
        self.categorical_cols = categorical_cols
        
        # Calculate class distribution
        unique, counts = np.unique(labels, return_counts=True)
        class_dist = dict(zip(unique, counts))
        
        # Calculate actual event rate (non-normal events)
        normal_class = 1  # Normal is mapped to 1
        total_samples = len(labels)
        normal_count = class_dist.get(normal_class, 0)
        event_count = total_samples - normal_count
        event_rate = event_count / total_samples if total_samples > 0 else 0.0
        
        print(f"[EventDetector] Prepared data:")
        print(f"  Features: {features.shape}")
        print(f"  Labels: {labels.shape}")
        print(f"  Categorical features: {categorical_features.shape if categorical_features is not None else None}")
        print(f"  Class distribution (mapped): {class_dist}")
        print(f"    Down (-1→0): {class_dist.get(0, 0)} ({class_dist.get(0, 0)/total_samples:.1%})")
        print(f"    Normal (0→1): {class_dist.get(1, 0)} ({class_dist.get(1, 0)/total_samples:.1%})")
        print(f"    Up (+1→2): {class_dist.get(2, 0)} ({class_dist.get(2, 0)/total_samples:.1%})")
        print(f"  Event rate (±1 events): {event_rate:.1%}")
        print(f"  Feature columns: {len(feature_cols)}")
        print(f"  Categorical columns: {categorical_cols}")
        
        return features, labels, categorical_features, feature_cols
    
    def train(self, df: pl.DataFrame, epochs: int = 100, batch_size: int = 512, 
              lr: float = 1e-3, weight_decay: float = 1e-4, 
              early_stopping_patience: int = 15,
              target_col: str = "label_1d_cls", feature_cols: Optional[List[str]] = None,
              train_years: List[int] = [2018, 2019], val_years: List[int] = [2020]):
        """
        Train the event detector with temporal split
        """
        
        # Temporal split to prevent data leakage
        train_df = df.filter(pl.col("year").is_in(train_years))
        val_df = df.filter(pl.col("year").is_in(val_years))
        
        print(f"[EventDetector] Temporal split:")
        print(f"  Train years: {train_years}, samples: {len(train_df)}")
        print(f"  Val years: {val_years}, samples: {len(val_df)}")
        
        # Check if we have data
        if len(train_df) == 0:
            raise ValueError(f"No training data found for years {train_years}")
        if len(val_df) == 0:
            raise ValueError(f"No validation data found for years {val_years}")
        
        # Prepare data
        X_train, y_train, cat_train, feature_names = self.prepare_data(train_df, target_col, feature_cols)
        X_val, y_val, cat_val, _ = self.prepare_data(val_df, target_col, feature_cols)
        
        # Additional check after data preparation
        if X_train.shape[0] == 0:
            raise ValueError(f"No training samples after data preparation. Check feature columns: {feature_cols}")
        if X_val.shape[0] == 0:
            raise ValueError(f"No validation samples after data preparation. Check feature columns: {feature_cols}")
        
        # Recreate model with correct dimensions
        actual_n_features = len(feature_names)
        n_tickers = len(set(train_df.select("ticker").to_pandas()["ticker"])) if "ticker" in train_df.columns else 3000
        
        categorical_dims = {
            'year': 3,
            'month': 12,
            'ticker_id': n_tickers
        }
        
        print(f"[EventDetector] Creating model with {actual_n_features} features and {n_tickers} tickers")
        
        # Create new model with correct dimensions
        self.model = EventDetector(
            n_features=actual_n_features,
            d_model=self.model.d_model,
            n_heads=8,  # From original config
            d_ff=512,   # From original config  
            dropout=0.25,
            n_classes=3,
            categorical_dims=categorical_dims
        ).to(self.device)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Create datasets
        train_dataset = EventDataset(X_train_scaled, y_train, cat_train)
        val_dataset = EventDataset(X_val_scaled, y_val, cat_val)
        
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # Setup training
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=0.5, patience=7
        )
        
        # Class weights for imbalanced data (3-class)
        class_counts = np.bincount(y_train, minlength=3)  # Ensure all 3 classes
        class_weights = len(y_train) / (len(class_counts) * class_counts)
        class_weights = torch.FloatTensor(class_weights).to(self.device)
        
        # Use Focal Loss for better handling of class imbalance
        criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        
        print(f"[EventDetector] Class weights: {class_weights.cpu().numpy()}")
        print(f"[EventDetector] Using Focal Loss (gamma=2.0) to address class imbalance")
        
        # Training loop
        best_val_auc = 0
        patience_counter = 0
        
        print(f"\n[EventDetector] Starting training...")
        print(f"  Train samples: {len(X_train)}")
        print(f"  Val samples: {len(X_val)}")
        print(f"  Event rate - Train: {y_train.mean():.3f}, Val: {y_val.mean():.3f}")
        from tqdm import tqdm
        
        for epoch in range(epochs):
            # Training
            self.model.train()
            train_loss = 0
            train_correct = 0
            train_total = 0
            
            # Training progress bar
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Train]', 
                            leave=False, ncols=100)
            
            for batch_data in train_pbar:
                if len(batch_data) == 3:  # With categorical features
                    batch_features, batch_categorical, batch_labels = batch_data
                    batch_categorical = batch_categorical.to(self.device)
                else:  # Without categorical features
                    batch_features, batch_labels = batch_data
                    batch_categorical = None
                
                batch_features = batch_features.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_features, batch_categorical)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                train_total += batch_labels.size(0)
                train_correct += (predicted == batch_labels).sum().item()
                
                # Update progress bar
                current_acc = train_correct / train_total if train_total > 0 else 0
                train_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{current_acc:.3f}'
                })
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_correct = 0
            val_total = 0
            val_probs = []
            val_true = []
            
            # Validation progress bar
            val_pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{epochs} [Val]', 
                          leave=False, ncols=100)
            
            with torch.no_grad():
                for batch_data in val_pbar:
                    if len(batch_data) == 3:  # With categorical features
                        batch_features, batch_categorical, batch_labels = batch_data
                        batch_categorical = batch_categorical.to(self.device)
                    else:  # Without categorical features
                        batch_features, batch_labels = batch_data
                        batch_categorical = None
                    
                    batch_features = batch_features.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                    
                    outputs = self.model(batch_features, batch_categorical)
                    loss = criterion(outputs, batch_labels)
                    
                    val_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    val_total += batch_labels.size(0)
                    val_correct += (predicted == batch_labels).sum().item()
                    
                    # For multiclass AUC calculation - collect all class probabilities
                    probs = F.softmax(outputs, dim=1)
                    val_probs.extend(probs.cpu().numpy())
                    val_true.extend(batch_labels.cpu().numpy())
                    
                    # Update progress bar
                    current_acc = val_correct / val_total if val_total > 0 else 0
                    val_pbar.set_postfix({
                        'loss': f'{loss.item():.4f}',
                        'acc': f'{current_acc:.3f}'
                    })
            
            # Calculate metrics
            train_acc = train_correct / train_total
            val_acc = val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            
            # Better metrics for trading decisions
            val_probs_array = np.array(val_probs)
            val_true_array = np.array(val_true)
            
            # PR-AUC for Up events (class 2)
            up_binary = (val_true_array == 2).astype(int)
            up_probs = val_probs_array[:, 2] if val_probs_array.ndim > 1 else val_probs_array
            try:
                up_pr_auc = average_precision_score(up_binary, up_probs)
            except:
                up_pr_auc = 0.0
            
            # PR-AUC for Down events (class 0)
            down_binary = (val_true_array == 0).astype(int)
            down_probs = val_probs_array[:, 0] if val_probs_array.ndim > 1 else val_probs_array
            try:
                down_pr_auc = average_precision_score(down_binary, down_probs)
            except:
                down_pr_auc = 0.0
            
            # Combined event PR-AUC
            event_binary = ((val_true_array == 0) | (val_true_array == 2)).astype(int)
            event_probs = up_probs + down_probs if val_probs_array.ndim > 1 else val_probs_array
            try:
                event_pr_auc = average_precision_score(event_binary, event_probs)
            except:
                event_pr_auc = 0.0
            
            # Use event PR-AUC as primary metric for early stopping
            val_auc = event_pr_auc
            
            # Learning rate scheduling
            scheduler.step(val_auc)
            
            # Save training history
            self.training_history.append({
                'epoch': epoch,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'train_acc': train_acc,
                'val_acc': val_acc,
                'val_auc': val_auc,
                'up_pr_auc': up_pr_auc,
                'down_pr_auc': down_pr_auc,
                'event_pr_auc': event_pr_auc,
                'lr': optimizer.param_groups[0]['lr']
            })
            
            # TensorFlow-like epoch summary
            val_preds = np.argmax(val_probs_array, axis=1)
            pred_dist = np.bincount(val_preds, minlength=3)
            total_val = len(val_preds)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f" - train_loss: {avg_train_loss:.4f} - train_acc: {train_acc:.4f}")
            print(f" - val_loss: {avg_val_loss:.4f} - val_acc: {val_acc:.4f}")
            print(f" - up_pr_auc: {up_pr_auc:.4f} - down_pr_auc: {down_pr_auc:.4f} - event_pr_auc: {event_pr_auc:.4f}")
            print(f" - lr: {optimizer.param_groups[0]['lr']:.6f}")
            print(f" - pred_dist: down={pred_dist[0]/total_val:.1%} normal={pred_dist[1]/total_val:.1%} up={pred_dist[2]/total_val:.1%}")
            
            # Early stopping
            if val_auc > best_val_auc:
                best_val_auc = val_auc
                patience_counter = 0
                # Save best model
                self.save_model("models/event_detector_best.pth")
                print(f" - val_auc improved to {best_val_auc:.4f}, saving model")
            else:
                patience_counter += 1
                print(f" - val_auc did not improve from {best_val_auc:.4f}")
                
            if patience_counter >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {patience_counter} epochs without improvement")
                break
            
            print()  # Empty line for readability
        
        print(f"Training completed! Best validation AUC: {best_val_auc:.4f}")
        
        # Load best model
        self.load_model("models/event_detector_best.pth")
        
        return self.training_history
    
    def predict(self, features: np.ndarray, categorical_features: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict event classes
        
        Args:
            features: Feature matrix
            categorical_features: Categorical feature matrix
            
        Returns:
            predictions (0/1/2), probabilities (3-class)
        """
        self.model.eval()
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        features_tensor = torch.FloatTensor(features_scaled).to(self.device)
        
        categorical_tensor = None
        if categorical_features is not None:
            categorical_tensor = torch.LongTensor(categorical_features).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(features_tensor, categorical_tensor)
            probabilities = F.softmax(outputs, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
        
        return predictions.cpu().numpy(), probabilities.cpu().numpy()
    
    def evaluate(self, df: pl.DataFrame, target_col: str = "label_1d_cls") -> Dict:
        """Evaluate model performance"""
        
        features, labels, categorical_features, _ = self.prepare_data(df, target_col, self.feature_names)
        predictions, probabilities = self.predict(features, categorical_features)
        
        # Calculate metrics
        try:
            auc_score = roc_auc_score(labels, probabilities, multi_class='ovr', average='macro')
        except:
            auc_score = 0.0
        
        # Classification report
        report = classification_report(labels, predictions, output_dict=True, zero_division=0)
        
        # Confusion matrix
        cm = confusion_matrix(labels, predictions)
        
        # Map back to original labels: 0->-1, 1->0, 2->+1
        class_names = ['Down (-1)', 'Normal (0)', 'Up (+1)']
        
        results = {
            'auc': auc_score,
            'accuracy': report['accuracy'],
            'macro_precision': report['macro avg']['precision'],
            'macro_recall': report['macro avg']['recall'],
            'macro_f1': report['macro avg']['f1-score'],
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'class_names': class_names
        }
        
        print(f"\n[EventDetector] Evaluation Results:")
        print(f"  AUC (macro): {auc_score:.4f}")
        print(f"  Accuracy: {report['accuracy']:.4f}")
        print(f"  Macro Precision: {report['macro avg']['precision']:.4f}")
        print(f"  Macro Recall: {report['macro avg']['recall']:.4f}")
        print(f"  Macro F1: {report['macro avg']['f1-score']:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"    Classes: {class_names}")
        for i, row in enumerate(cm):
            print(f"    {class_names[i]}: {row}")
        
        return results
    
    def save_model(self, path: str):
        """Save model and preprocessing components"""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'training_history': self.training_history,
            'model_config': {
                'n_features': self.model.n_features,
                'd_model': self.model.d_model
            }
        }
        torch.save(checkpoint, path)
        print(f"[EventDetector] Model saved to {path}")
    
    def load_model(self, path: str):
        """Load model and preprocessing components"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.scaler = checkpoint['scaler']
        self.feature_names = checkpoint['feature_names']
        self.training_history = checkpoint.get('training_history', [])
        print(f"[EventDetector] Model loaded from {path}")


def create_event_detector(n_features: int, categorical_dims: Optional[Dict[str, int]] = None, n_tickers: int = 3000, **kwargs) -> EventDetector:
    """Factory function to create EventDetector model"""
    # Default categorical dimensions for temporal + ticker features
    if categorical_dims is None:
        categorical_dims = {
            'year': 3,        # 2018, 2019, 2020 -> 0, 1, 2
            'month': 12,      # 1-12 -> 0-11
            'ticker_id': n_tickers  # Dynamic based on dataset
        }
    
    return EventDetector(n_features=n_features, categorical_dims=categorical_dims, **kwargs)
