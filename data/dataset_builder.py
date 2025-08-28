# data/dataset_builder.py
import os
import polars as pl
from typing import Iterable, Optional, Literal
from datetime import date
from .load_silver import scan_ohlcv, filter_ohlcv, sample_tickers, load_silver, SILVER_ROOT
from features.feature_sets import add_feature_set
from labelers.basic import future_return_labels
from pathlib import Path
import pyarrow as pa
import pyarrow.dataset as pads
import pyarrow.parquet as pq
import hashlib
import json

Market = Literal["KR","US"]

def _quick_counts(lf: pl.LazyFrame, tag: str, verbose: bool=False):
    if not verbose:
        return
    out = (
        lf.select([
            pl.len().alias("rows"),
            pl.n_unique("ticker").alias("n_tickers"),
            pl.col("date").min().alias("min_date"),
            pl.col("date").max().alias("max_date"),
        ])
        .collect(streaming=True)
    )
    print(f"[{tag}] rows={int(out['rows'][0])}, tickers={int(out['n_tickers'][0])}, "
          f"{out['min_date'][0]}..{out['max_date'][0]}")

def _dir_latest_mtime(path: Path) -> int:
    """Return latest modified time (ns) under the given directory. 0 if not exists."""
    try:
        if not Path(path).exists():
            return 0
        latest = 0
        for root, dirs, files in os.walk(path):
            for name in files:
                try:
                    fp = os.path.join(root, name)
                    mtime_ns = os.stat(fp).st_mtime_ns
                    if mtime_ns > latest:
                        latest = mtime_ns
                except OSError:
                    continue
        return latest
    except Exception:
        return 0

def _stable_list(value: Optional[Iterable]) -> Optional[list]:
    if value is None:
        return None
    try:
        return sorted(list(value))
    except Exception:
        return list(value)

def _make_build_cache_key(
    *,
    years: Optional[Iterable[int]],
    market: Optional[str],
    exchanges: Optional[Iterable[str]],
    tickers: Optional[Iterable[str]],
    max_tickers: Optional[int],
    start: Optional[date],
    end: Optional[date],
    feature_set: str,
    label_horizon: int,
    label_task: str,
    label_thresh: float,
    select_cols: Optional[Iterable[str]],
    drop_na_rows: bool,
    cache_invalidate_on: str,
) -> str:
    params = {
        "v": "build_dataset_v2_normalized",  # 정규화 버전으로 업데이트
        "years": _stable_list(years),
        "market": market,
        "exchanges": _stable_list(exchanges),
        "tickers": _stable_list([t.upper() for t in tickers] if tickers else None),
        "max_tickers": max_tickers,
        "start": start.isoformat() if start else None,
        "end": end.isoformat() if end else None,
        "feature_set": feature_set,
        "label_horizon": label_horizon,
        "label_task": label_task,
        "label_thresh": label_thresh,
        "select_cols": _stable_list(select_cols),
        "drop_na_rows": drop_na_rows,
        "invalidate": cache_invalidate_on,
    }
    if cache_invalidate_on == "silver_mtime":
        params["silver_latest_mtime_ns"] = _dir_latest_mtime(Path(SILVER_ROOT))

    blob = json.dumps(params, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()

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
    verbose: bool = False,
    *,
    use_cache: bool = True,
    cache_dir: str | Path = "data/cache/datasets",
    force_recompute: bool = False,
    cache_invalidate_on: Literal["never", "silver_mtime"] = "silver_mtime",
) -> pl.DataFrame:
    """빌드된 데이터셋을 반환."""
    # 캐시 확인
    cache_path: Path | None = None
    if use_cache and not force_recompute:
        key = _make_build_cache_key(
            years=years,
            market=market,
            exchanges=exchanges,
            tickers=tickers,
            max_tickers=max_tickers,
            start=start,
            end=end,
            feature_set=feature_set,
            label_horizon=label_horizon,
            label_task=label_task,
            label_thresh=label_thresh,
            select_cols=select_cols,
            drop_na_rows=drop_na_rows,
            cache_invalidate_on=cache_invalidate_on,
        )
        cache_dir_path = Path(cache_dir)
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir_path / f"{key[:16]}.parquet"
        if cache_path.exists():
            if verbose:
                print(f"[cache] hit: {cache_path}")
            return pl.read_parquet(str(cache_path))

    # 1. 최적화된 스캔 (파티션 필터 적용)
    lf = scan_ohlcv(market=market, tickers=tickers, years=years, exchanges=exchanges)
    _quick_counts(lf, "scan", verbose)

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
        scan_years=years,
        sort_result=False,
    )
    _quick_counts(lf, "filter", verbose)

    # 3. 티커 샘플링을 먼저 수행하여 이후 연산량을 줄임
    lf = sample_tickers(lf, max_tickers=max_tickers)
    _quick_counts(lf, "sample_pre", verbose)

    # 샘플링 이후에 정렬하여 윈도우/시프트 계산의 순서를 보장
    lf = lf.sort(["ticker","date"])

    # 4. 피처 추가
    lf = add_feature_set(lf, feature_set=feature_set)
    _quick_counts(lf, "features", verbose)

    # 5. 라벨 추가
    lf = future_return_labels(lf, horizon=label_horizon, task=label_task, thresh=label_thresh)
    _quick_counts(lf, "labels", verbose)

    # 6. 피처 정규화 (Z-Score) - 기본 컬럼 제외하고 모든 피처에 적용
    if verbose:
        print("[normalize] Applying Z-score normalization to features...")
    
    # 정규화할 컬럼 식별 (기본 컬럼과 라벨 컬럼 제외)
    schema_names = lf.collect_schema().names()
    base_cols = {"date","ticker","market","exchange","currency","year","open","high","low","close","adj_close","volume","turnover"}
    
    feature_cols = [c for c in schema_names 
                   if c not in base_cols 
                   and not c.startswith("label_") 
                   and not c.startswith("futret_")
                   and lf.collect_schema()[c] in (pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64)]
    
    if feature_cols:
        # 각 피처별로 Z-score 정규화 (전체 데이터셋 기준)
        normalize_exprs = []
        for col in feature_cols:
            x = pl.col(col)
            z_score = (x - x.mean()) / (x.std() + 1e-9)  # eps 추가로 0으로 나누기 방지
            normalize_exprs.append(z_score.alias(col))
        
        # 기존 컬럼들 + 정규화된 피처들
        keep_cols = [c for c in schema_names if c not in feature_cols]
        lf = lf.with_columns(normalize_exprs).select(keep_cols + feature_cols)
        
        if verbose:
            print(f"[normalize] Applied Z-score to {len(feature_cols)} features")
    
    _quick_counts(lf, "normalized", verbose)

    # NaN 처리
    if drop_na_rows:
        # 피처 계산 초기 구간/미래 라벨로 생기는 NaN 제거
        schema_names = lf.collect_schema().names()
        feat_cols = [c for c in schema_names if c not in ["date","ticker","market","exchange","open","high","low","close","adj_close","volume","turnover","year"]]

        lf = lf.drop_nulls(feat_cols)
        _quick_counts(lf, "drop_na", verbose)

    # 컬럼 선택
    if select_cols:
        lf = lf.select(list(select_cols))

    df = lf.collect(streaming=False)

    # 캐시에 저장
    if use_cache:
        try:
            if cache_path is None:
                key = _make_build_cache_key(
                    years=years,
                    market=market,
                    exchanges=exchanges,
                    tickers=tickers,
                    max_tickers=max_tickers,
                    start=start,
                    end=end,
                    feature_set=feature_set,
                    label_horizon=label_horizon,
                    label_task=label_task,
                    label_thresh=label_thresh,
                    select_cols=select_cols,
                    drop_na_rows=drop_na_rows,
                    cache_invalidate_on=cache_invalidate_on,
                )
                cache_dir_path = Path(cache_dir)
                cache_dir_path.mkdir(parents=True, exist_ok=True)
                cache_path = cache_dir_path / f"{key[:16]}.parquet"
            # 단일 파일로 저장
            save_dataset(df, cache_path, format="parquet")
            if verbose:
                print(f"[cache] write: {cache_path}")
        except Exception as e:
            if verbose:
                print(f"[cache] write failed: {e}")

    return df
    
def _to_arrow(df_or_lf: pl.DataFrame | pl.LazyFrame) -> pa.Table:
    if isinstance(df_or_lf, pl.LazyFrame):
        df = df_or_lf.collect(streaming=True)
    else:
        df = df_or_lf
    return df.to_arrow()

def _infer_pa_type(dtype: pl.DataType) -> pa.DataType:
    # 최소 매핑(필요 시 확장)
    if dtype == pl.Utf8:
        return pa.string()
    if dtype in (pl.Int8, pl.Int16, pl.Int32, pl.Int64):
        return pa.int64()
    if dtype in (pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64):
        return pa.uint64()
    if dtype in (pl.Float32, pl.Float64):
        return pa.float64()
    if dtype == pl.Boolean:
        return pa.bool_()
    if dtype == pl.Date:
        return pa.date32()
    if dtype == pl.Datetime:
        return pa.timestamp("us")
    # fallback
    return pa.string()

def _build_partition_schema(df: pl.DataFrame, partition_cols: list[str]) -> pa.Schema:
    fields = []
    for c in partition_cols:
        if c not in df.columns:
            raise ValueError(f"Partition column '{c}' not found in dataframe.")
        fields.append(pa.field(c, _infer_pa_type(df.schema[c])))
    return pa.schema(fields)

def save_dataset(
    df_or_lf: pl.DataFrame | pl.LazyFrame,
    out_path: str | Path,
    *,
    format: Literal["parquet", "ipc", "feather"] = "parquet",
    partition_cols: list[str] | None = None,
    hive_style: bool = True,            # market=KR/ 형태로 저장
    existing_data_behavior: Literal["overwrite_or_ignore", "error", "delete_matching"] = "overwrite_or_ignore",
    max_rows_per_file: int | None = None,
    compression: str = "zstd",
) -> Path:
    """
    데이터셋을 디스크에 저장.
    - partition_cols 지정 시 파티션 디렉토리로 저장 (pyarrow.dataset.write_dataset)
    - 미지정 시 단일 파일로 저장

    existing_data_behavior:
      - "overwrite_or_ignore": 같은 파티션이 있으면 덮어쓰거나(파일 추가) 무시
      - "delete_matching": 같은 파티션 파일을 지우고 재작성
      - "error": 이미 있으면 에러
    """
    out_path = Path(out_path)
    table = _to_arrow(df_or_lf)

    if partition_cols:
        out_path.mkdir(parents=True, exist_ok=True)

        # 파티션 스키마 구성
        # LazyFrame일 수 있어, 스키마 추출 위해 한 번 collect된 DF 필요 → _to_arrow에서 처리됨
        if isinstance(df_or_lf, pl.LazyFrame):
            df_pl = df_or_lf.collect_schema()
            # collect_schema는 dtype만, 컬럼 존재 보장은 별도 체크 필요 → table로부터 확인
            df_cols = set(table.schema.names)
            for c in partition_cols:
                if c not in df_cols:
                    raise ValueError(f"Partition column '{c}' not found in table: {c}")
            # 간단화: Arrow 테이블에서 스키마 추론
            part_schema = pa.schema([table.schema.field(c) for c in partition_cols])
        else:
            part_schema = _build_partition_schema(df_or_lf, partition_cols)

        partitioning = pads.partitioning(part_schema, flavor="hive") if hive_style else pads.DirectoryPartitioning(part_schema)

        # 쓰기 옵션
        writer = dict(compression=compression)
        pads.write_dataset(
            data=table,
            base_dir=str(out_path),
            format="parquet" if format == "parquet" else format,
            partitioning=partitioning,
            existing_data_behavior=existing_data_behavior,
            max_rows_per_file=max_rows_per_file,
            file_options=writer,
        )
        return out_path

    # 파티션 없이 단일 파일 저장
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if format == "parquet":
        pq.write_table(table, str(out_path), compression=compression)
    elif format in ("ipc", "feather"):
        # ipc = Arrow IPC 파일, feather는 ipc의 일종(확장자만)
        with pa.ipc.new_file(str(out_path), table.schema) as sink:
            sink.write(table)
    else:
        raise ValueError(f"Unsupported format: {format}")
    return out_path