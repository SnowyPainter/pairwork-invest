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

def scan_ohlcv() -> pl.LazyFrame:
    dset = ds.dataset(
        str(SILVER_ROOT),
        format="parquet",
        partitioning=ds.partitioning(PARTITION_SCHEMA, flavor="hive"),
    )
    lf = pl.scan_pyarrow_dataset(dset).with_columns(
        pl.col("symbol").alias("ticker")   # ticker 항상 보장
    )
    return lf

def filter_ohlcv(
    lf: pl.LazyFrame,
    market: Optional[Market] = None,
    exchanges: Optional[Iterable[str]] = None,   # US: ["NYSE","NASDAQ","SP500","FORBES2000"]
    tickers: Optional[Iterable[str]] = None,
    years: Optional[Iterable[int]] = None,
    start: Optional[date] = None,
    end: Optional[date] = None,
) -> pl.LazyFrame:
    if market:
        lf = lf.filter(pl.col("market")==market)
    if exchanges:
        lf = lf.filter(pl.col("exchange").is_in([e.upper() for e in exchanges]))
    if tickers:
        lf = lf.filter(pl.col("ticker").is_in([t.upper() for t in tickers]))
    if years:
        lf = lf.filter(pl.col("year").is_in(list(years)))
    if start:
        lf = lf.filter(pl.col("date")>=pl.lit(start))
    if end:
        lf = lf.filter(pl.col("date")<=pl.lit(end))
    return lf.sort(["ticker","date"])

def sample_tickers(
    lf: pl.LazyFrame, max_tickers: Optional[int] = None, seed: int = 42
) -> pl.LazyFrame:
    if max_tickers is None:
        return lf
    # 티커 목록 뽑고 샘플링
    tics = (lf.select("ticker").unique().collect(streaming=True)["ticker"].to_list())
    if len(tics) > max_tickers:
        import random
        random.seed(seed)
        tics = random.sample(tics, max_tickers)
    return lf.filter(pl.col("ticker").is_in(tics))

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
    lf = scan_ohlcv()
    lf = filter_ohlcv(lf, market, exchanges, tickers, years, start, end)
    lf = sample_tickers(lf, max_tickers)
    if columns:
        lf = lf.select(list(columns))
    return lf.collect(streaming=True)