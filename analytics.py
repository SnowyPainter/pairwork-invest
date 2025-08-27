# analytics.py
from __future__ import annotations
import os, math, warnings
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict
from itertools import product

from data.dataset_builder import build_dataset  # 유저가 이미 갖고 있는 함수
from data.clusters import SECTORS  # 섹터별 종목 코드 import

# -----------------------------
# 설정
# -----------------------------
BASE_OUTDIR = "reports/analytics"
os.makedirs(BASE_OUTDIR, exist_ok=True)

# 실험 설정: 다양한 horizon과 label_thresh 조합
HORIZONS = [1, 2, 5]
LABEL_THRESHOLDS = [0.1, 0.3]

BASE_CFG = dict(
    years=[2018, 2019, 2020, 2021],
    market="KR",
    feature_set="v3",               # v1 / v2 / v3 중 선택
    label_task="regression",        # regression or classification
    group_cols=["ticker"],
    date_col="date",                # 날짜 컬럼명에 맞춰 수정
    max_tickers=None,  # 섹터별로 제한하므로 전체 제한 해제
)

# -----------------------------
# 유틸
# -----------------------------
def to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    return df.to_pandas(use_pyarrow_extension_array=True)

def _is_num_dtype(dt: pl.DataType) -> bool:
    import polars.selectors as cs
    return dt.is_numeric()

def save_df(df: pd.DataFrame, name: str, sector_dir: str = None):
    if sector_dir:
        outdir = os.path.join(BASE_OUTDIR, sector_dir)
        os.makedirs(outdir, exist_ok=True)
    else:
        outdir = BASE_OUTDIR
    path = os.path.join(outdir, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"saved: {path}")

def safe_corr(df: pd.DataFrame, method: str="pearson") -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return df.corr(method=method, numeric_only=True)

def plot_heatmap(corr: pd.DataFrame, title: str, fname: str, vmax: float=1.0, sector_dir: str = None):
    fig = plt.figure(figsize=(10, 8))
    ax = plt.gca()
    im = ax.imshow(corr.values, vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=90, fontsize=7)
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index, fontsize=7)
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, shrink=0.8)
    plt.tight_layout()
    
    if sector_dir:
        outdir = os.path.join(BASE_OUTDIR, sector_dir)
        os.makedirs(outdir, exist_ok=True)
    else:
        outdir = BASE_OUTDIR
    path = os.path.join(outdir, f"{fname}.png")
    plt.savefig(path, dpi=200)
    plt.close(fig)
    print(f"saved: {path}")

def compute_vif(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    """
    간단 VIF 근사: 각 x ~ 나머지 X 회귀의 R^2 로부터 1/(1-R^2) 계산.
    statsmodels 없이, 선형회귀의 닫힌형식 대신 QR 기반의 np.linalg.lstsq 사용.
    """
    X = df[features].astype("float64").replace([np.inf, -np.inf], np.nan).dropna()
    out = []
    for i, col in enumerate(features):
        y = X[col].values
        X_ = X.drop(columns=[col]).values
        if X_.shape[1] == 0:
            vif = np.nan
        else:
            # add bias
            Xb = np.c_[np.ones(len(X_)), X_]
            beta, *_ = np.linalg.lstsq(Xb, y, rcond=None)
            yhat = Xb @ beta
            ssr = np.sum((yhat - y.mean())**2)
            sst = np.sum((y - y.mean())**2) + 1e-12
            r2 = ssr / sst
            vif = 1.0 / max(1e-9, (1.0 - r2))
        out.append((col, vif))
    return pd.DataFrame(out, columns=["feature", "vif"]).sort_values("vif", ascending=False)

def spearman_ic(df: pd.DataFrame, features: List[str], target: str) -> pd.DataFrame:
    out = []
    for f in features:
        s = df[[f, target]].dropna()
        if len(s) < 10:
            ic = np.nan
        else:
            ic = s[f].rank().corr(s[target].rank(), method="pearson")
        out.append((f, ic))
    return pd.DataFrame(out, columns=["feature", "IC"]).sort_values("IC", ascending=False)

def rolling_ic(df: pd.DataFrame, date_col: str, features: List[str], target: str, window: int=60) -> pd.DataFrame:
    """
    날짜 기준 롤링 스피어만 IC. 날짜는 일 단위라고 가정.
    """
    d = df[[date_col, target] + features].dropna().copy()
    d = d.sort_values(date_col)
    res = []
    for f in features:
        # 롤링 구현: 윈도우마다 스피어만
        vals = []
        dates = []
        s = d[[date_col, f, target]].dropna()
        for i in range(window, len(s)+1):
            w = s.iloc[i-window:i]
            ic = w[f].rank().corr(w[target].rank(), method="pearson")
            vals.append(ic)
            dates.append(w[date_col].iloc[-1])
        res.append(pd.DataFrame({date_col: dates, "feature": f, "IC": vals}))
    return pd.concat(res, ignore_index=True)

def time_series_cv_splits(dates: pd.Series, n_splits: int=5) -> List[Tuple[np.ndarray, np.ndarray]]:
    uniq = np.sort(dates.unique())
    folds = []
    for i in range(n_splits):
        cut = int(len(uniq)*(i+1)/(n_splits+1))
        train_end = uniq[cut]
        val_end_cut = int(len(uniq)*(i+2)/(n_splits+1))
        val_end = uniq[val_end_cut] if val_end_cut < len(uniq) else uniq[-1]
        tr_idx = dates <= train_end
        va_idx = (dates > train_end) & (dates <= val_end)
        if va_idx.sum() == 0: 
            continue
        folds.append((np.where(tr_idx)[0], np.where(va_idx)[0]))
    return folds

def lightgbm_importance(df: pd.DataFrame, features: List[str], target: str, date_col: str, n_splits: int=5) -> pd.DataFrame:
    try:
        import lightgbm as lgb
    except ImportError:
        print("LightGBM 미설치: pip install lightgbm")
        return pd.DataFrame(columns=["feature", "gain", "split"])
    X = df[features].astype("float32").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y = df[target].astype("float32")
    dates = df[date_col]
    folds = time_series_cv_splits(dates, n_splits=n_splits)

    feat_gain = pd.Series(0.0, index=features)
    feat_split = pd.Series(0.0, index=features)

    params = dict(
        objective="regression",
        metric="rmse",
        boosting_type="gbdt",
        learning_rate=0.05,
        num_leaves=31,
        max_depth=-1,
        subsample=0.8,
        colsample_bytree=0.8,
        min_data_in_leaf=20,
        verbose=-1,
        n_estimators=500,
    )

    for (tr, va) in folds:
        dtr = lgb.Dataset(X.iloc[tr], label=y.iloc[tr])
        dva = lgb.Dataset(X.iloc[va], label=y.iloc[va])
        model = lgb.train(params, dtr, valid_sets=[dtr, dva], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)])
        imp = pd.Series(model.feature_importance(importance_type="gain"), index=features)
        imp_split = pd.Series(model.feature_importance(importance_type="split"), index=features)
        feat_gain += imp
        feat_split += imp_split

    out = pd.DataFrame({"feature": features, "gain": feat_gain.values, "split": feat_split.values})
    out = out.sort_values("gain", ascending=False)
    return out

# -----------------------------
# 섹터별 분석 함수들
# -----------------------------
def analyze_single_sector(sector_name: str, tickers: List[str], horizon: int, label_thresh: float) -> Dict:
    """단일 섹터에 대한 분석을 수행하고 결과를 반환"""
    target_col = f"futret_{horizon}"
    
    print(f"\n=== 분석 중: {sector_name} (horizon={horizon}, thresh={label_thresh}) ===")
    
    # 데이터셋 빌드
    try:
        df_pl = build_dataset(
            years=BASE_CFG["years"],
            market=BASE_CFG["market"],
            feature_set=BASE_CFG["feature_set"],
            label_task=BASE_CFG["label_task"],
            label_thresh=label_thresh,
            label_horizon=horizon,
            tickers=tickers,  # 섹터별 종목만 필터링
            streaming_collect=False,
            use_cache=False
        )
    except Exception as e:
        print(f"데이터 빌드 실패: {sector_name} - {e}")
        return {}
    
    if df_pl.height == 0:
        print(f"데이터 없음: {sector_name}")
        return {}
    
    # 컬럼 분리
    cols = df_pl.columns
    non_feat = set(BASE_CFG["group_cols"] + [BASE_CFG["date_col"], target_col])
    feature_cols = [c for c in cols if c not in non_feat and _is_num_dtype(df_pl[c].dtype)]
    
    if len(feature_cols) == 0:
        print(f"피처 없음: {sector_name}")
        return {}
    
    print(f"  데이터: rows={df_pl.height:,}, tickers={df_pl['ticker'].n_unique()}, feats={len(feature_cols)}")
    
    # 분석 결과 저장용
    results = {
        'sector': sector_name,
        'horizon': horizon,
        'label_thresh': label_thresh,
        'n_rows': df_pl.height,
        'n_tickers': df_pl['ticker'].n_unique(),
        'n_features': len(feature_cols)
    }
    
    # pandas 변환
    pdf = to_pandas(df_pl.select([target_col] + feature_cols))
    
    # 1) 타깃과의 스피어만 IC
    ic = spearman_ic(pdf, feature_cols, target_col)
    if not ic.empty:
        results['mean_ic'] = ic['IC'].mean()
        results['median_ic'] = ic['IC'].median()
        results['top5_ic_mean'] = ic.head(5)['IC'].mean()
        results['best_feature'] = ic.iloc[0]['feature']
        results['best_ic'] = ic.iloc[0]['IC']
    
    # 2) LightGBM 피처 중요도
    try:
        imp = lightgbm_importance(
            to_pandas(df_pl.select([BASE_CFG["date_col"], target_col] + feature_cols)),
            feature_cols, target_col, BASE_CFG["date_col"], n_splits=3  # 빠른 분석을 위해 3-fold
        )
        if not imp.empty:
            results['lgbm_top_feature'] = imp.iloc[0]['feature']
            results['lgbm_top_gain'] = imp.iloc[0]['gain']
            results['lgbm_mean_gain'] = imp['gain'].mean()
    except Exception as e:
        print(f"  LightGBM 분석 실패: {e}")
    
    # 3) 파일 저장 - 섹터별 디렉토리 사용
    sector_dirname = sector_name.replace('/', '_').replace(' ', '_')
    filename_prefix = f"h{horizon}__t{label_thresh}"
    
    # IC 저장
    if not ic.empty:
        save_df(ic, f"ic__{filename_prefix}", sector_dir=sector_dirname)
    
    # 상위 피처들의 상관관계 히트맵
    if len(ic) >= 5:
        top_features = ic.head(10)['feature'].tolist()
        if len(top_features) >= 2:
            corr_s = safe_corr(pdf[top_features], method="spearman")
            plot_heatmap(corr_s, f"{sector_name} Top Features Correlation", f"heatmap__{filename_prefix}", sector_dir=sector_dirname)
    
    return results

def generate_sector_comparison_report(all_results: List[Dict]):
    """섹터 간 비교 리포트 생성"""
    if not all_results:
        return
    
    df_results = pd.DataFrame(all_results)
    
    # 1) 전체 결과 요약
    save_df(df_results, "sector_comparison_summary")
    
    # 2) horizon별 최고 성과 섹터
    for horizon in HORIZONS:
        h_data = df_results[df_results['horizon'] == horizon]
        if not h_data.empty:
            # IC 기준 최고 섹터
            best_ic = h_data.loc[h_data['mean_ic'].idxmax()] if 'mean_ic' in h_data.columns else None
            if best_ic is not None:
                print(f"\n[Horizon {horizon}] 최고 IC 섹터: {best_ic['sector']} (IC={best_ic['mean_ic']:.4f})")
    
    # 3) label_thresh별 최고 성과 섹터
    for thresh in LABEL_THRESHOLDS:
        t_data = df_results[df_results['label_thresh'] == thresh]
        if not t_data.empty:
            best_ic = t_data.loc[t_data['mean_ic'].idxmax()] if 'mean_ic' in t_data.columns else None
            if best_ic is not None:
                print(f"\n[Threshold {thresh}] 최고 IC 섹터: {best_ic['sector']} (IC={best_ic['mean_ic']:.4f})")
    
    # 4) 섹터별 horizon 안정성 분석
    sector_stability = []
    for sector in df_results['sector'].unique():
        sector_data = df_results[df_results['sector'] == sector]
        if len(sector_data) >= 3:  # 여러 horizon 결과가 있는 경우
            ic_std = sector_data['mean_ic'].std() if 'mean_ic' in sector_data.columns else np.nan
            ic_mean = sector_data['mean_ic'].mean() if 'mean_ic' in sector_data.columns else np.nan
            sector_stability.append({
                'sector': sector,
                'ic_mean': ic_mean,
                'ic_std': ic_std,
                'stability_score': ic_mean / (ic_std + 1e-6) if not np.isnan(ic_std) else np.nan
            })
    
    if sector_stability:
        stability_df = pd.DataFrame(sector_stability).sort_values('stability_score', ascending=False)
        save_df(stability_df, "sector_stability_analysis")
        print(f"\n가장 안정적인 섹터: {stability_df.iloc[0]['sector']}")

def create_sector_performance_visualization(all_results: List[Dict]):
    """섹터 성과 시각화"""
    if not all_results:
        return
    
    df_results = pd.DataFrame(all_results)
    
    # IC 성과 히트맵 (섹터 x horizon)
    if 'mean_ic' in df_results.columns:
        pivot_ic = df_results.pivot_table(
            values='mean_ic', 
            index='sector', 
            columns='horizon', 
            aggfunc='mean'
        )
        
        if not pivot_ic.empty:
            fig, ax = plt.subplots(figsize=(10, 8))
            im = ax.imshow(pivot_ic.values, aspect='auto', cmap='RdYlBu_r')
            ax.set_xticks(range(len(pivot_ic.columns)))
            ax.set_xticklabels([f"H{h}" for h in pivot_ic.columns])
            ax.set_yticks(range(len(pivot_ic.index)))
            ax.set_yticklabels(pivot_ic.index, fontsize=8)
            ax.set_title("Sector Performance by Horizon (Mean IC)", fontsize=12)
            
            # 값 표시
            for i in range(len(pivot_ic.index)):
                for j in range(len(pivot_ic.columns)):
                    val = pivot_ic.iloc[i, j]
                    if not np.isnan(val):
                        ax.text(j, i, f'{val:.3f}', ha='center', va='center', fontsize=8)
            
            plt.colorbar(im, ax=ax)
            plt.tight_layout()
            
            path = os.path.join(BASE_OUTDIR, "sector_performance_heatmap.png")
            plt.savefig(path, dpi=200)
            plt.close()
            print(f"saved: {path}")

# -----------------------------
# 메인 분석 루틴
# -----------------------------
def main():
    """섹터별 다중 조건 분석 실행"""
    print("=== 섹터별 분석 시작 ===")
    print(f"분석 대상: {len(SECTORS)}개 섹터")
    print(f"Horizons: {HORIZONS}")
    print(f"Label thresholds: {LABEL_THRESHOLDS}")
    
    all_results = []
    
    # 모든 섹터 x horizon x threshold 조합 분석
    total_combinations = len(SECTORS) * len(HORIZONS) * len(LABEL_THRESHOLDS)
    current = 0
    
    for sector_name, tickers in SECTORS.items():
        for horizon, label_thresh in product(HORIZONS, LABEL_THRESHOLDS):
            current += 1
            print(f"\n진행률: {current}/{total_combinations}")
            
            result = analyze_single_sector(sector_name, tickers, horizon, label_thresh)
            if result:  # 결과가 있는 경우만 추가
                all_results.append(result)
    
    # 결과 분석 및 리포트 생성
    print(f"\n=== 분석 완료: {len(all_results)}개 결과 ===")
    
    if all_results:
        generate_sector_comparison_report(all_results)
        create_sector_performance_visualization(all_results)
        
        # 주요 발견사항 출력
        df_results = pd.DataFrame(all_results)
        
        print("\n=== 주요 발견사항 ===")
        
        # 1) 전체 최고 성과 섹터
        if 'mean_ic' in df_results.columns:
            best_overall = df_results.loc[df_results['mean_ic'].idxmax()]
            print(f"전체 최고 IC: {best_overall['sector']} (H{best_overall['horizon']}, T{best_overall['label_thresh']}) - IC={best_overall['mean_ic']:.4f}")
        
        # 2) 고수익 임계값(0.3)에서도 잘 되는 섹터
        high_thresh_data = df_results[df_results['label_thresh'] == 0.3]
        if not high_thresh_data.empty and 'mean_ic' in high_thresh_data.columns:
            best_high_thresh = high_thresh_data.loc[high_thresh_data['mean_ic'].idxmax()]
            print(f"고수익 임계값 최적 섹터: {best_high_thresh['sector']} (H{best_high_thresh['horizon']}) - IC={best_high_thresh['mean_ic']:.4f}")
        
        # 3) horizon 안정성이 좋은 섹터 (여러 horizon에서 일관된 성과)
        stability_analysis = []
        for sector in df_results['sector'].unique():
            sector_data = df_results[df_results['sector'] == sector]
            if len(sector_data) >= 3 and 'mean_ic' in sector_data.columns:
                ic_mean = sector_data['mean_ic'].mean()
                ic_std = sector_data['mean_ic'].std()
                stability_analysis.append((sector, ic_mean, ic_std, ic_mean/(ic_std + 1e-6)))
        
        if stability_analysis:
            stability_analysis.sort(key=lambda x: x[3], reverse=True)
            best_stable = stability_analysis[0]
            print(f"가장 안정적 섹터: {best_stable[0]} - 평균IC={best_stable[1]:.4f}, 안정성점수={best_stable[3]:.2f}")
    
    print("\n분석 완료!")

if __name__ == "__main__":
    main()
