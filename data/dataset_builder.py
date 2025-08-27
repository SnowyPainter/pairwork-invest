# data/dataset_builder.py
import os
import polars as pl
from typing import Iterable, Optional, Literal
from datetime import date
from .load_silver import scan_ohlcv, filter_ohlcv, sample_tickers, load_silver
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

    캐시 키: 필터 조건들을 포함하여 더 세부적인 캐싱
    - tickers, exchanges 등의 필터가 있을 때는 별도 캐시
    """

    # -----------------
    # 캐시 경로 계산 (최적화)
    # -----------------
    def _years_key(yrs: Optional[Iterable[int]]) -> str:
        if yrs is None:
            return "all"
        try:
            ys = sorted({int(y) for y in yrs})
        except Exception:
            ys = list(yrs)
        return "-".join(str(y) for y in ys) if ys else "all"

    def _tickers_key(tickers: Optional[Iterable[str]]) -> str:
        if tickers is None:
            return "all"
        if len(tickers) <= 5:
            return "-".join(sorted([t.upper() for t in tickers]))
        else:
            return f"{len(tickers)}tickers"

    def _exchanges_key(exchanges: Optional[Iterable[str]]) -> str:
        if exchanges is None:
            return "all"
        return "-".join(sorted([e.upper() for e in exchanges]))

    # 캐시 키에 주요 필터들을 포함
    cache_root = cache_dir or os.path.join("data", "cache", "datasets")
    os.makedirs(cache_root, exist_ok=True)

    cache_parts = [
        f"y-{_years_key(years)}",
        f"m-{market or 'all'}",
        f"fs-{feature_set}",
        f"lt-{label_task}",
        f"th-{label_thresh}",
        f"h-{label_horizon}"
    ]

    # tickers나 exchanges가 있으면 캐시 키에 포함
    if tickers:
        cache_parts.append(f"t-{_tickers_key(tickers)}")
    if exchanges:
        cache_parts.append(f"e-{_exchanges_key(exchanges)}")
    if start or end:
        cache_parts.append(f"d-{start or 'none'}_{end or 'none'}")

    cache_name = f"ds__{'__'.join(cache_parts)}.parquet"
    cache_path = os.path.join(cache_root, cache_name)

    if use_cache and os.path.exists(cache_path):
        try:
            df_cached = pl.read_parquet(cache_path)
            print(f"[cache] 캐시에서 로드: {cache_path}")
            return df_cached
        except Exception as e:
            print(f"[cache] 캐시 로드 실패: {e}")

    # -----------------
    # 캐시 미존재: 최적화된 빌드
    # -----------------
    # 1. 최적화된 스캔 (파티션 필터 적용)
    lf = scan_ohlcv(market=market, tickers=tickers, years=years, exchanges=exchanges)
    _quick_counts(lf, "scan")

    # 2. 최적화된 필터링 (scan에서 적용된 필터는 건너뜀)
    lf = filter_ohlcv(
        lf,
        market=market,
        exchanges=exchanges,
        tickers=tickers,
        years=years,
        start=start,
        end=end,
        scan_market=market,
        scan_years=years
    )
    _quick_counts(lf, "filter")

    # 3. 피처 추가
    lf = add_feature_set(lf, feature_set=feature_set)
    _quick_counts(lf, "features")

    # 4. 라벨 추가
    lf = future_return_labels(lf, horizon=label_horizon, task=label_task, thresh=label_thresh)
    _quick_counts(lf, "labels")

    # 5. 티커 샘플링 (필요한 경우)
    lf = sample_tickers(lf, max_tickers=max_tickers)
    _quick_counts(lf, "sample")

    # NaN 처리
    if drop_na_rows:
        # 피처 계산 초기 구간/미래 라벨로 생기는 NaN 제거
        schema_names = lf.collect_schema().names()
        feat_cols = [c for c in schema_names if c not in ["date","ticker","market","exchange","open","high","low","close","adj_close","volume","turnover","year"]]

        lf = lf.drop_nulls(feat_cols)
        _quick_counts(lf, "drop_na")

    # 컬럼 선택
    if select_cols:
        lf = lf.select(list(select_cols))

    # 최종 수집
    print(f"[build] 캐시 저장: {cache_path}")
    df = lf.collect(streaming=streaming_collect)

    # 캐시 저장
    if use_cache:
        try:
            df.write_parquet(cache_path)
            print(f"[cache] 캐시 저장 완료")
        except Exception as e:
            print(f"[cache] 캐시 저장 실패: {e}")

    return df
