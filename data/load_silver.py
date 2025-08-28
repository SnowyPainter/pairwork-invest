import polars as pl
import pyarrow as pa
import pyarrow.dataset as ds
from pathlib import Path
from glob import glob
from typing import Iterable, Optional, Literal
from datetime import date

from data.load_etl import PARTITION_SCHEMA

SILVER_ROOT = Path("data/silver/ohlcv")

Market = Literal["KR","US"]

def scan_ohlcv(
    market: Optional[Market] = None,
    tickers: Optional[Iterable[str]] = None,
    years: Optional[Iterable[int]] = None,
    exchanges: Optional[Iterable[str]] = None,
    **kwargs  # 추가 파라미터 호환성을 위해
) -> pl.LazyFrame:
    """
    최적화된 스캔: 가능한 경우 파티션 필터를 scan 단계에서 적용
    """
    # 기본 데이터셋
    dset = ds.dataset(
        str(SILVER_ROOT),
        format="parquet",
        partitioning=ds.partitioning(PARTITION_SCHEMA, flavor="hive"),
    )

    # 기본 스캔부터 시작 (파티션 필터 대신 Polars 필터 사용)
    lf = pl.scan_pyarrow_dataset(dset)

    # 최소한의 필터만 scan 단계에서 적용 (market만)
    applied_scan_filters = []
    if market:
        lf = lf.filter(pl.col("market") == market)
        applied_scan_filters.append(f"market={market}")
    if tickers:
        lf = lf.filter(pl.col("ticker").is_in(tickers))
        applied_scan_filters.append(f"tickers={len(tickers)}개")
    if years:
        lf = lf.filter(pl.col("year").is_in(years))
        applied_scan_filters.append(f"years={years}")
    if exchanges:
        lf = lf.filter(pl.col("exchange").is_in(exchanges))
        applied_scan_filters.append(f"exchanges={exchanges}")
    if applied_scan_filters:
        print(f"[scan] 기본 필터 적용: {applied_scan_filters}")

    # 나머지 필터는 filter_ohlcv에서 적용

    return lf

def filter_ohlcv(
    lf: pl.LazyFrame,
    market: Optional[Market] = None,
    exchanges: Optional[Iterable[str]] = None,   # US: ["NYSE","NASDAQ","SP500","FORBES2000"]
    tickers: Optional[Iterable[str]] = None,
    years: Optional[Iterable[int]] = None,
    start: Optional[date] = None,
    end: Optional[date] = None,
    scan_market: Optional[Market] = None,
    scan_years: Optional[Iterable[int]] = None,
    *,
    sort_result: bool = True,
) -> pl.LazyFrame:
    """
    최적화된 필터링: scan 단계에서 적용된 필터는 건너뜀
    필터링 순서: 데이터 양을 가장 많이 줄이는 순서로 적용
    """
    applied_filters = []

    # 1. 이미 scan 단계에서 적용된 필터는 건너뜀
    if market and market != scan_market:
        lf = lf.filter(pl.col("market")==market)
        applied_filters.append(f"market={market}")

    # 2. 티커 필터 (데이터 양을 크게 줄이는 필터)
    if tickers:
        tickers_upper = [t.upper() for t in tickers]
        lf = lf.filter(pl.col("ticker").is_in(tickers_upper))
        applied_filters.append(f"tickers={len(tickers)}개")

    # 3. 연도 필터
    if years:
        years_list = list(years)
        if len(years_list) == 1:
            lf = lf.filter(pl.col("year") == years_list[0])
        else:
            lf = lf.filter(pl.col("year").is_in(years_list))
        applied_filters.append(f"years={years_list}")

    # 4. 날짜 필터 (시간 범위 제한)
    if start:
        lf = lf.filter(pl.col("date")>=pl.lit(start))
        applied_filters.append(f"start={start}")

    if end:
        lf = lf.filter(pl.col("date")<=pl.lit(end))
        applied_filters.append(f"end={end}")

    # 5. 거래소 필터
    if exchanges:
        lf = lf.filter(pl.col("exchange").is_in([e.upper() for e in exchanges]))
        applied_filters.append(f"exchanges={list(exchanges)}")

    if applied_filters:
        print(f"[filter] 적용된 필터: {', '.join(applied_filters)}")

    return lf.sort(["ticker","date"]) if sort_result else lf

def sample_tickers(
    lf: pl.LazyFrame, max_tickers: Optional[int] = None, seed: int = 42
) -> pl.LazyFrame:
    """
    최적화된 티커 샘플링
    """
    if max_tickers is None:
        return lf

    # 단일 패스로 고유 티커 추출 후 샘플링 결정
    import random
    random.seed(seed)

    all_tickers = lf.select("ticker").unique().collect(streaming=False)["ticker"].to_list()
    current_count = len(all_tickers)
    if current_count <= max_tickers:
        print(f"[sample] 티커 수 {current_count}개 <= {max_tickers}개, 샘플링 생략")
        return lf

    sampled_tickers = random.sample(all_tickers, max_tickers)

    print(f"[sample] 티커 샘플링: {current_count}개 -> {max_tickers}개")
    return lf.filter(pl.col("ticker").is_in(sampled_tickers))

def load_silver(
    market: Optional[Market]=None,
    exchanges: Optional[Iterable[str]] = None,
    tickers: Optional[Iterable[str]] = None,
    years: Optional[Iterable[int]] = None,
    start: Optional[date] = None,
    end: Optional[date] = None,
    max_tickers: Optional[int] = None,
    columns: Optional[Iterable[str]] = None,
) -> pl.DataFrame:
    """최적화된 데이터 로드"""
    # 최적화된 스캔 사용
    lf = scan_ohlcv(market=market, tickers=tickers, years=years, exchanges=exchanges)
    lf = filter_ohlcv(
        lf, market, exchanges, tickers, years, start, end,
        scan_market=market, scan_years=years
    )
    lf = sample_tickers(lf, max_tickers)
    if columns:
        lf = lf.select(list(columns))
    return lf.collect(streaming=True)