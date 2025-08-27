# pip install polars pyarrow tqdm
import os
from pathlib import Path
from datetime import datetime
import polars as pl
import pyarrow as pa
import pyarrow.dataset as pads
from tqdm import tqdm

RAW = Path("data/raw")
SILVER = Path("data/silver/ohlcv")  # 최종 파티션 루트
SILVER.mkdir(parents=True, exist_ok=True)
PARTITION_SCHEMA = pa.schema([
            pa.field("market", pa.string()),
            pa.field("ticker", pa.string()),
            pa.field("year", pa.int32()),
    ])

def _numf(col: str) -> pl.Expr:
    # 문자열 → 숫자(float). 쉼표/공백/기타 문자 제거, 빈문자/NaN/null → null
    return (pl.col(col)
              .cast(pl.Utf8)
              .str.strip_chars()
              .str.replace_all(",", "")
              .str.replace_all(r"[^0-9eE\.\+\-]", "")   # 숫자/부호/소수점/지수만 남김
              .replace("", None)
              .cast(pl.Float64, strict=False))


def _numi(col: str) -> pl.Expr:
    # 위 float화를 거쳐 정수로; 필요시 round→Int64
    return _numf(col).round(0).cast(pl.Int64, strict=False)

def _to_arrow(df: pl.DataFrame) -> pa.Table:
    return df.to_arrow()

def _write_partitioned(table: pa.Table):
    part = pads.partitioning(schema=PARTITION_SCHEMA, flavor="hive")
    pads.write_dataset(
        data=table,
        base_dir=str(SILVER),
        format="parquet",
        partitioning=part,
        existing_data_behavior="overwrite_or_ignore",
    )

def _norm_batch_kr(df: pl.DataFrame, ticker: str) -> pl.DataFrame:
    # KR: date,open,high,low,close,volume,value (yyyymmdd)
    return (
        df.rename({c:c.lower() for c in df.columns})
          .with_columns([
            pl.col("date").cast(pl.Utf8).str.strptime(pl.Date, "%Y%m%d", strict=False),
            _numf("open").alias("open"),
            _numf("high").alias("high"),
            _numf("low").alias("low"),
            _numf("close").alias("close"),
            _numi("volume").alias("volume"),
            _numf("value").alias("turnover"),
          ])
          .select([
              "date","open","high","low","close",
              pl.col("close").alias("adj_close"),
              "volume","turnover",
          ])
          .with_columns([
              pl.lit(ticker.upper()).alias("ticker"),
              pl.lit("KR").alias("market"),
              pl.lit(None, dtype=pl.Utf8).alias("exchange"),
              pl.lit("KRW").alias("currency"),
          ])
          .drop_nulls(["date"])
          .unique(subset=["date"])
          .with_columns(pl.col("date").dt.year().alias("year"))
    )

def _norm_batch_us(df: pl.DataFrame, ticker: str, exchange_hint: str|None=None) -> pl.DataFrame:
    # US: Date,Low,Open,Volume,High,Close,Adjusted Close (dd-mm-YYYY)
    # 컬럼 대소문자 섞임 대비
    cols = {c.lower(): c for c in df.columns}
    def pick(name):  # 이름 안전하게 고르기
        for k in cols:
            if k.replace(" ", "") == name.replace(" ", "").lower():
                return cols[k]
        return name  # 없으면 그대로
    return (
        df.with_columns([
            pl.col(pick("Date")).cast(pl.Utf8).str.strptime(pl.Date, "%d-%m-%Y", strict=False).alias("date"),
            _numf(pick("Open")).alias("open"),
            _numf(pick("High")).alias("high"),
            _numf(pick("Low")).alias("low"),
            _numf(pick("Close")).alias("close"),
            _numf(pick("Adjusted Close")).alias("adj_close"),
            _numi(pick("Volume")).alias("volume"),
        ])
        .with_columns([
            (pl.col("close") * pl.col("volume")).alias("turnover")
        ])
        .select([
            "date","open","high","low","close","adj_close","volume","turnover",
        ])
        .with_columns([
            pl.lit(ticker.upper()).alias("ticker"),
            pl.lit("US").alias("market"),
            pl.lit(exchange_hint.upper() if exchange_hint else None).alias("exchange"),
            pl.lit("USD").alias("currency"),
        ])
        .drop_nulls(["date"])
        .unique(subset=["date"])
        .with_columns(pl.col("date").dt.year().alias("year"))
    )

def _process_csv_stream(csv_path: Path, kind: str, ticker: str, exchange_hint: str|None=None, batch_rows: int = 200_000):
    """
    kind: "KR" | "US"
    배치 단위로 읽어서 즉시 파티션 저장. 메모리 사용량이 배치 크기에 비례.
    """
    # batched reader (스트리밍)
    reader = pl.read_csv_batched(
        str(csv_path),
        batch_size=batch_rows,
        ignore_errors=True,
        infer_schema_length=0
    )

    # 배치 수 추정이 어려우니 파일 크기 기준 세부 바를 표시
    total_bytes = os.path.getsize(csv_path)
    processed_bytes = 0
    # 내부 배치 진행바
    with tqdm(total=total_bytes, unit="B", unit_scale=True, desc=f"[{kind}] {ticker} ({csv_path.name})") as pbar:
        while True:
            batch = reader.next_batches(1)  # 한 번에 1배치
            if not batch:
                break
            df = batch[0]
            if df.height == 0:
                continue

            if kind == "KR":
                ndf = _norm_batch_kr(df, ticker)
            else:
                ndf = _norm_batch_us(df, ticker, exchange_hint)

            _write_partitioned(_to_arrow(ndf))

            # 진행률 갱신: 대략적으로 현재까지 읽은 바이트 수 추정
            processed_bytes += df.estimated_size()
            pbar.update(df.estimated_size())

def ingest_all(batch_rows: int = 200_000):
    # 상위 진행바(파일 단위)
    kr_files = list((RAW/"korean-stock-data").rglob("*.csv"))
    us_groups = []
    root = RAW / "us-stock-data" / "stock_market_data"
    for exch in ["nyse","nasdaq","sp500","forbes2000"]:
        csv_root = root / exch / "csv"
        if csv_root.exists():
            for f in csv_root.rglob("*.csv"):
                us_groups.append((f, exch))

    # 한국
    for f in tqdm(kr_files, desc="KR files", unit="file"):
        stem = f.stem.strip()
        if stem.lower() in {"code", "readme"}:  # 혹시 있을 메타 파일 스킵
            continue
        _process_csv_stream(f, kind="KR", ticker=stem, batch_rows=batch_rows)

    # 미국
    for f, exch in tqdm(us_groups, desc="US files", unit="file"):
        stem = f.stem.strip()
        if stem.lower() in {"code", "readme"}:
            continue
        _process_csv_stream(f, kind="US", ticker=stem, exchange_hint=exch, batch_rows=batch_rows)

# 실행 예:
ingest_all(batch_rows=100_000)
