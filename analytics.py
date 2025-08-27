# analytics.py
from __future__ import annotations
import os, math, warnings
import numpy as np
import pandas as pd
import polars as pl
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional

from data.dataset_builder import build_dataset  # 유저가 이미 갖고 있는 함수

# -----------------------------
# 설정
# -----------------------------
OUTDIR = "reports/analytics"
os.makedirs(OUTDIR, exist_ok=True)

CFG = dict(
    years=[2020],
    market="KR",
    feature_set="v3",               # v1 / v2 / v3 중 선택
    label_task="regression",        # regression or classification
    target_col="target",            # dataset_builder가 반환하는 타깃 이름에 맞춰 수정
    group_cols=["ticker"],
    date_col="date",                # 날짜 컬럼명에 맞춰 수정
)

# -----------------------------
# 유틸
# -----------------------------
def to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    return df.to_pandas(use_pyarrow_extension_array=True)

def _is_num_dtype(dt: pl.DataType) -> bool:
    return dt in pl.NUMERIC_DTYPES

def save_df(df: pd.DataFrame, name: str):
    path = os.path.join(OUTDIR, f"{name}.csv")
    df.to_csv(path, index=False)
    print(f"saved: {path}")

def safe_corr(df: pd.DataFrame, method: str="pearson") -> pd.DataFrame:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return df.corr(method=method, numeric_only=True)

def plot_heatmap(corr: pd.DataFrame, title: str, fname: str, vmax: float=1.0):
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
    path = os.path.join(OUTDIR, f"{fname}.png")
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
        model = lgb.train(params, dtr, valid_sets=[dtr, dva], verbose_eval=False, early_stopping_rounds=50)
        imp = pd.Series(model.feature_importance(importance_type="gain"), index=features)
        imp_split = pd.Series(model.feature_importance(importance_type="split"), index=features)
        feat_gain += imp
        feat_split += imp_split

    out = pd.DataFrame({"feature": features, "gain": feat_gain.values, "split": feat_split.values})
    out = out.sort_values("gain", ascending=False)
    return out

# -----------------------------
# 메인 분석 루틴
# -----------------------------
def main():

    # analytics.py main() 초반 디버그
    df_pl = build_dataset(
        years=CFG["years"],
        market=CFG["market"],
        feature_set=CFG["feature_set"],
        label_task=CFG["label_task"],
        streaming_collect=False
    )

    # 2) 기본 컬럼 분리
    cols = df_pl.columns
    non_feat = set(CFG["group_cols"] + [CFG["date_col"], CFG["target_col"]])
    feature_cols = [c for c in cols if c not in non_feat and _is_num_dtype(df_pl[c].dtype)]
    print(f"rows={df_pl.height:,}, feats={len(feature_cols)}")

    exit()

    # 3) 기초 통계 & 누락 요약
    basic = df_pl.select(feature_cols).describe()
    save_df(to_pandas(basic), "basic_describe")

    miss = to_pandas(
        pl.DataFrame({
            "feature": feature_cols,
            "na_rate": [df_pl[c].null_count()/len(df_pl) for c in feature_cols]
        })
    ).sort_values("na_rate", ascending=False)
    save_df(miss, "missing_rate")

    # 4) 상관(HM): Pearson / Spearman
    pdf = to_pandas(df_pl.select([CFG["target_col"]] + feature_cols))
    corr_p = safe_corr(pdf[feature_cols], method="pearson")
    corr_s = safe_corr(pdf[feature_cols], method="spearman")
    save_df(corr_p.reset_index(names="feature").rename(columns={"index":"feature"}), "corr_pearson_matrix_flat")
    save_df(corr_s.reset_index(names="feature").rename(columns={"index":"feature"}), "corr_spearman_matrix_flat")
    # 타깃과의 상관 Top
    tgt_corr_p = pdf[feature_cols + [CFG["target_col"]]].corr(method="pearson")[CFG["target_col"]].drop(CFG["target_col"]).sort_values(ascending=False)
    tgt_corr_s = pdf[feature_cols + [CFG["target_col"]]].corr(method="spearman")[CFG["target_col"]].drop(CFG["target_col"]).sort_values(ascending=False)
    save_df(tgt_corr_p.reset_index().rename(columns={"index":"feature", CFG["target_col"]:"pearson_with_target"}), "target_corr_pearson")
    save_df(tgt_corr_s.reset_index().rename(columns={"index":"feature", CFG["target_col"]:"spearman_with_target"}), "target_corr_spearman")
    # 히트맵(대형이면 상위 30개만)
    top30 = tgt_corr_s.abs().sort_values(ascending=False).head(30).index.tolist()
    if len(top30) >= 2:
        plot_heatmap(corr_s.loc[top30, top30], "Spearman Corr (Top30 by |IC|)", "heatmap_top30_spearman")

    # 5) VIF(다중공선성)
    vif = compute_vif(pdf, features=top30 if len(top30)>2 else feature_cols[:30])
    save_df(vif, "vif_top")

    # 6) 정보계수(IC) + 롤링 IC(시간 안정성)
    ic = spearman_ic(pdf, feature_cols, CFG["target_col"])
    save_df(ic, "ic_overall")
    ric = rolling_ic(
        to_pandas(df_pl.select([CFG["date_col"], CFG["target_col"]] + feature_cols)),
        CFG["date_col"], top30 if len(top30)>0 else feature_cols[:20], CFG["target_col"], window=60
    )
    save_df(ric, "rolling_ic_60d")

    # 7) LightGBM 피처중요도 (시계열 CV)
    imp = lightgbm_importance(
        to_pandas(df_pl.select([CFG["date_col"], CFG["target_col"]] + feature_cols)),
        feature_cols, CFG["target_col"], CFG["date_col"], n_splits=5
    )
    if not imp.empty:
        save_df(imp, "lgbm_importance")

    print("Done.")

if __name__ == "__main__":
    main()
