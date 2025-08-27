# data/dataset_builder.py
import os
import polars as pl
from typing import Iterable, Optional, Literal
from datetime import date
from .load_silver import scan_ohlcv, filter_ohlcv, sample_tickers
from features.feature_sets import add_feature_set
from labelers.basic import future_return_labels

Market = Literal["KR","US"]

def _quick_counts(lf: pl.LazyFrame, tag: str):
    out = (
        lf.select([
            pl.len().alias("rows"),
            pl.n_unique("ticker").alias("n_tickers"),
            pl.col("date").min().alias("min_date"),
            pl.col("date").max().alias("max_date"),
        ])
        .collect(streaming=False)
    )
    print(f"[{tag}] rows={int(out['rows'][0])}, tickers={int(out['n_tickers'][0])}, "
          f"{out['min_date'][0]}..{out['max_date'][0]}")

def build_dataset(
    years: Optional[Iterable[int]] = None,
    market: Optional[Market] = None,
    exchanges: Optional[Iterable[str]] = None,
    tickers: Optional[Iterable[str]] = None,
    max_tickers: Optional[int] = None,
    start: Optional[date] = None,
    end: Optional[date] = None,
    feature_set: str = "v1",
    label_horizon: int = 5,
    label_task: str = "regression",   # or "classification"
    label_thresh: float = 0.02,
    select_cols: Optional[Iterable[str]] = None,  # 최종 반환 컬럼 제한(선택)
    drop_na_rows: bool = True,
    streaming_collect: bool = True,
    use_cache: bool = True,
    cache_dir: Optional[str] = None,
) -> pl.DataFrame:
    """빌드된 데이터셋을 반환. use_cache=True 이면 디스크 캐시 사용.

    캐시 키: (years, market, feature_set, label_task)
    - analytics.py 에서 원하는 네 가지만으로 캐싱합니다.
    - 다른 인자(exchanges, tickers, select_cols 등)는 키에 포함하지 않습니다.
    """

    # -----------------
    # 캐시 경로 계산
    # -----------------
    def _years_key(yrs: Optional[Iterable[int]]) -> str:
        if yrs is None:
            return "all"
        try:
            ys = sorted({int(y) for y in yrs})
        except Exception:
            ys = list(yrs)
        return "-".join(str(y) for y in ys) if ys else "all"

    cache_root = cache_dir or os.path.join("data", "cache", "datasets")
    os.makedirs(cache_root, exist_ok=True)
    cache_name = f"ds_y-{_years_key(years)}__m-{market or 'all'}__fs-{feature_set}__lt-{label_task}.parquet"
    cache_path = os.path.join(cache_root, cache_name)

    if use_cache and os.path.exists(cache_path):
        try:
            df_cached = pl.read_parquet(cache_path)
            return df_cached
        except Exception:
            pass  # 캐시 읽기 실패 시 아래에서 재빌드

    # -----------------
    # 캐시 미존재: 실제 빌드
    # -----------------
    lf = scan_ohlcv()
    _quick_counts(lf, "scan")
    lf = filter_ohlcv(lf, market=market, exchanges=exchanges, tickers=tickers, years=years, start=start, end=end)
    _quick_counts(lf, "filter")
    lf = add_feature_set(lf, feature_set=feature_set)
    _quick_counts(lf, "features")
    lf = future_return_labels(lf, horizon=label_horizon, task=label_task, thresh=label_thresh)
    _quick_counts(lf, "labels")
    lf = sample_tickers(lf, max_tickers=max_tickers)
    _quick_counts(lf, "sample")

    if drop_na_rows:
        # 피처 계산 초기 구간/미래 라벨로 생기는 NaN 제거
        schema_names = lf.collect_schema().names()
        feat_cols = [c for c in schema_names if c not in ["date","ticker","market","exchange","open","high","low","close","adj_close","volume","turnover","year"]]

        lf = lf.drop_nulls(feat_cols)

    if select_cols:
        lf = lf.select(list(select_cols))

    BASE = {"date","ticker","market","exchange","open","high","low","close","adj_close","volume","turnover","year"}
    schema_names = lf.collect_schema().names()
    feat_cols = [c for c in schema_names if c not in BASE]

    dbg = lf.select([pl.len().alias("_n"), *[pl.col(c).is_null().sum().alias(c) for c in feat_cols]]).collect()
    n = int(dbg["_n"][0]) if dbg.height else 0
    bad = [c for c in feat_cols if (n == 0) or (int(dbg[c][0]) >= 0.99*n)]
    print("ALL-NULL(or ~ALL) features:", bad, "n=", n)


    df = lf.collect(streaming=streaming_collect)

    if use_cache:
        try:
            df.write_parquet(cache_path)
        except Exception:
            pass  # 쓰기 실패는 무시하고 그냥 반환

    return df
