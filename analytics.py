# analytics.py
from __future__ import annotations
import os, math, warnings, json
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
HORIZONS = [1]
LABEL_THRESHOLDS = [0.05]

BASE_CFG = dict(
    years=[2018, 2019, 2020, 2021],
    market="KR",
    feature_set="v3",               # v1 / v2 / v3 중 선택
    label_task="classification",        # regression or classification
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
            try:
                # 스피어만 상관계수 계산: rank().corr()의 결과가 scalar인지 확인
                ic_series = s[f].rank().corr(s[target].rank())
                # Series.corr()는 scalar를 반환하므로 float 변환
                ic = float(ic_series)
            except (ValueError, TypeError, AttributeError):
                # 상관계수 계산 실패시 NaN 반환
                ic = np.nan
        out.append((f, ic))
    return pd.DataFrame(out, columns=["feature", "IC"]).sort_values("IC", ascending=False)

def remove_highly_correlated_features(df: pd.DataFrame, features: List[str], corr_threshold: float = 0.8) -> List[str]:
    """
    상관계수가 높은 피처들을 제거하여 다중공선성 문제를 완화
    """
    if len(features) <= 1:
        return features

    # 피처간 상관계수 계산
    corr_matrix = safe_corr(df[features], method="spearman")

    # 상관계수가 높은 피처 쌍 찾기
    high_corr_pairs = []
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            if abs(corr_matrix.iloc[i, j]) > corr_threshold:
                high_corr_pairs.append((features[i], features[j], abs(corr_matrix.iloc[i, j])))

    # IC가 높은 피처를 우선적으로 유지
    ic_scores = spearman_ic(df, features, df.columns[-1])  # target은 마지막 컬럼으로 가정
    ic_dict = dict(zip(ic_scores['feature'], ic_scores['IC']))

    # 제거할 피처 결정
    to_remove = set()
    for feat1, feat2, corr_val in high_corr_pairs:
        # IC가 낮은 피처를 제거
        if ic_dict.get(feat1, 0) >= ic_dict.get(feat2, 0):
            to_remove.add(feat2)
        else:
            to_remove.add(feat1)

    filtered_features = [f for f in features if f not in to_remove]

    print(f"  상관계수 제거: {len(features)} -> {len(filtered_features)} 피처 (제거됨: {list(to_remove)})")

    return filtered_features

def analyze_rolling_ic_volatility(rolling_ic_df: pd.DataFrame) -> Dict:
    """
    롤링 IC 데이터프레임에서 변동성 분석
    """
    if rolling_ic_df.empty:
        return {}

    # 피처별로 그룹화하여 변동성 계산
    volatility_stats = []
    for feature in rolling_ic_df['feature'].unique():
        feature_data = rolling_ic_df[rolling_ic_df['feature'] == feature]['IC'].dropna()

        if len(feature_data) < 5:  # 최소 데이터 요구
            continue

        stats = {
            'feature': feature,
            'ic_mean': feature_data.mean(),
            'ic_std': feature_data.std(),
            'ic_min': feature_data.min(),
            'ic_max': feature_data.max(),
            'ic_volatility': feature_data.std() / (abs(feature_data.mean()) + 1e-9),  # 변동계수
            'n_periods': len(feature_data),
            'ic_stability': abs(feature_data.mean()) / (feature_data.std() + 1e-9)
        }
        volatility_stats.append(stats)

    return pd.DataFrame(volatility_stats).sort_values('ic_stability', ascending=False)

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
            try:
                ic_series = w[f].rank().corr(w[target].rank())
                ic = float(ic_series)
            except (ValueError, TypeError, AttributeError):
                ic = np.nan
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
    """단일 섹터에 대한 포괄적 분석을 수행하고 결과를 반환"""
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
            use_cache=True
        )
    except Exception as e:
        print(f"데이터 빌드 실패: {sector_name} - {e}")
        return {}

    if df_pl.height == 0:
        print(f"데이터 없음: {sector_name}")
        return {}

    # 컬럼 분리
    cols = df_pl.columns
    target_col = cols[-1]
    non_feat = set(BASE_CFG["group_cols"] + [BASE_CFG["date_col"], target_col])
    feature_cols = [c for c in cols if c not in non_feat and _is_num_dtype(df_pl[c].dtype)]

    print(f"\n=== 분석 중: {sector_name} (horizon={horizon}, thresh={label_thresh}) target={target_col} ===")


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

    # pandas 변환 (전체 데이터)
    full_pdf = to_pandas(df_pl.select([BASE_CFG["date_col"], target_col] + feature_cols))

    # 1) 타깃과의 스피어만 IC
    ic = spearman_ic(full_pdf, feature_cols, target_col)
    if not ic.empty:
        results['mean_ic'] = ic['IC'].mean()
        results['median_ic'] = ic['IC'].median()
        results['top5_ic_mean'] = ic.head(5)['IC'].mean()
        results['best_feature'] = ic.iloc[0]['feature']
        results['best_ic'] = ic.iloc[0]['IC']

    # 2) 상관계수 높은 피처 제거
    filtered_features = remove_highly_correlated_features(full_pdf, feature_cols, corr_threshold=0.8)
    results['n_features_after_corr_filter'] = len(filtered_features)

    # 3) 롤링 IC 분석 (필터링된 피처 사용)
    rolling_ic_df = None
    rolling_volatility = None
    try:
        rolling_ic_df = rolling_ic(full_pdf, BASE_CFG["date_col"], filtered_features, target_col, window=60)
        if not rolling_ic_df.empty:
            rolling_volatility = analyze_rolling_ic_volatility(rolling_ic_df)
            if not rolling_volatility.empty:
                results['rolling_ic_mean'] = rolling_volatility['ic_mean'].mean()
                results['rolling_ic_volatility'] = rolling_volatility['ic_volatility'].mean()
                results['most_stable_feature'] = rolling_volatility.iloc[0]['feature']
                results['most_stable_ic'] = rolling_volatility.iloc[0]['ic_stability']
    except Exception as e:
        print(f"  롤링 IC 분석 실패: {e}")

    # 4) LightGBM 피처 중요도 (필터링된 피처 사용)
    imp = None
    try:
        imp = lightgbm_importance(full_pdf, filtered_features, target_col, BASE_CFG["date_col"], n_splits=3)
        if not imp.empty:
            results['lgbm_top_feature'] = imp.iloc[0]['feature']
            results['lgbm_top_gain'] = imp.iloc[0]['gain']
            results['lgbm_mean_gain'] = imp['gain'].mean()
    except Exception as e:
        print(f"  LightGBM 분석 실패: {e}")

    # 5) VIF 분석 (다중공선성)
    vif_df = None
    try:
        vif_df = compute_vif(full_pdf, filtered_features[:20])  # 상위 20개 피처만
        if not vif_df.empty:
            results['mean_vif'] = vif_df['vif'].mean()
            results['max_vif'] = vif_df['vif'].max()
            results['high_vif_features'] = (vif_df['vif'] > 5).sum()  # VIF > 5인 피처 수
    except Exception as e:
        print(f"  VIF 분석 실패: {e}")

    # 6) 파일 저장 - 섹터별 디렉토리 사용
    sector_dirname = sector_name.replace('/', '_').replace(' ', '_')
    filename_prefix = f"h{horizon}__t{label_thresh}"

    # IC 저장
    if not ic.empty:
        save_df(ic, f"ic__{filename_prefix}", sector_dir=sector_dirname)

    # 필터링된 IC 저장
    if len(filtered_features) < len(feature_cols):
        filtered_ic = spearman_ic(full_pdf, filtered_features, target_col)
        if not filtered_ic.empty:
            save_df(filtered_ic, f"ic_filtered__{filename_prefix}", sector_dir=sector_dirname)

    # 롤링 IC 저장
    if rolling_ic_df is not None and not rolling_ic_df.empty:
        save_df(rolling_ic_df, f"rolling_ic__{filename_prefix}", sector_dir=sector_dirname)

    # 롤링 IC 변동성 저장
    if rolling_volatility is not None and not rolling_volatility.empty:
        save_df(rolling_volatility, f"rolling_ic_volatility__{filename_prefix}", sector_dir=sector_dirname)

    # LightGBM 중요도 저장
    if imp is not None and not imp.empty:
        save_df(imp, f"lgbm_importance__{filename_prefix}", sector_dir=sector_dirname)

    # VIF 저장
    if vif_df is not None and not vif_df.empty:
        save_df(vif_df, f"vif__{filename_prefix}", sector_dir=sector_dirname)

    # 상위 피처들의 상관관계 히트맵
    if len(ic) >= 5:
        top_features = ic.head(10)['feature'].tolist()
        if len(top_features) >= 2:
            corr_s = safe_corr(full_pdf[top_features], method="spearman")
            plot_heatmap(corr_s, f"{sector_name} Top Features Correlation", f"heatmap__{filename_prefix}", sector_dir=sector_dirname)

    return results

def save_all_results_summary(all_results: List[Dict]):
    """모든 분석 결과를 요약하여 저장"""
    if not all_results:
        return

    df_results = pd.DataFrame(all_results)
    save_df(df_results, "all_results_summary")
    print(f"\n=== 분석 완료 요약 ===")
    print(f"총 분석 수: {len(all_results)}")
    print(f"섹터 수: {df_results['sector'].n_unique()}")
    print(f"평균 IC: {df_results.get('mean_ic', pd.Series()).mean():.4f}")
    print(f"평균 롤링 IC: {df_results.get('rolling_ic_mean', pd.Series()).mean():.4f}")
    print(f"평균 변동성: {df_results.get('rolling_ic_volatility', pd.Series()).mean():.4f}")

# -----------------------------
# 메인 분석 루틴
# -----------------------------
def main():
    """섹터별 포괄적 분석 실행 - 모든 결과를 파일로 저장"""
    print("=== 섹터별 포괄적 분석 시작 ===")
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

    # 전체 결과 요약 저장
    print(f"\n=== 분석 완료: {len(all_results)}개 결과 ===")

    if all_results:
        save_all_results_summary(all_results)

    print("\n분석 완료! 각 섹터별 폴더에 모든 결과가 저장되었습니다.")

if __name__ == "__main__":
    main()

