#!/bin/bash

# Improved Event Detector Training Script
echo "🚀 Starting Improved Event Detector Training..."

export PYTHONPATH="${PWD}:${PYTHONPATH}"

# Parameters
MARKET="KR"
TRAIN_YEARS="2018,2019,2020,2021"
MAX_TICKERS=3000
EPOCHS=30
BATCH_SIZE=256
LR=0.001

MODEL_NAME="event_detector_improved_$(date +%Y%m%d_%H%M)"

echo "🔧 Configuration:"
echo "   Market: $MARKET"
echo "   Train Years: $TRAIN_YEARS"  
echo "   Max Tickers: $MAX_TICKERS"
echo "   Epochs: $EPOCHS"
echo "   Batch Size: $BATCH_SIZE"
echo "   Model: $MODEL_NAME"
echo ""

# Run training
python models/train_event_detector.py \
    --market $MARKET \
    --years $TRAIN_YEARS \
    --max_tickers $MAX_TICKERS \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --d_model 192 \
    --n_heads 8 \
    --dropout 0.25 \
    --early_stopping 10 \
    --model_name $MODEL_NAME

echo ""
echo "✅ Training completed!"
echo "📁 Check models/checkpoints/ for results"
