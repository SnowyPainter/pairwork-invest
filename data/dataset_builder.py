#!/usr/bin/env python3
"""
Dataset Builder - 깔끔하게 재구현된 버전

주요 기능:
- 원본 데이터 로드 및 필터링
- 피처 생성
- 라벨 생성
- 선택적 Z-score 정규화
- 캐싱 지원
"""

import os
import hashlib
import json
from pathlib import Path
from typing import Iterable, Optional, Literal
from datetime import date

import polars as pl

# 프로젝트 모듈 임포트
from .load_silver import scan_ohlcv, filter_ohlcv, sample_tickers, SILVER_ROOT
from features.feature_sets import add_feature_set
from labelers.basic import future_return_labels


Market = Literal["KR", "US"]


def _quick_counts(lf: pl.LazyFrame, tag: str, verbose: bool = False):
    """데이터 통계 출력"""
    if not verbose:
        return
    
    try:
        stats = (
            lf.select([
                pl.len().alias("rows"),
                pl.n_unique("ticker").alias("n_tickers"),
                pl.col("date").min().alias("min_date"),
                pl.col("date").max().alias("max_date"),
            ])
            .collect()
        )
        
        row = stats.row(0)
        print(f"[{tag}] rows={row[0]:,}, tickers={row[1]:,}, {row[2]}~{row[3]}")
    except Exception as e:
        print(f"[{tag}] stats error: {e}")


def _dir_latest_mtime(path: Path) -> int:
    """디렉토리 내 최신 수정 시간 반환"""
    try:
        if not path.exists():
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
    """안정적인 리스트 변환"""
    if value is None:
        return None
    try:
        return sorted(list(value))
    except Exception:
        return list(value)


def _make_build_cache_key(
    years: Optional[Iterable[int]] = None,
    market: Optional[str] = None,
    exchanges: Optional[Iterable[str]] = None,
    tickers: Optional[Iterable[str]] = None,
    max_tickers: Optional[int] = None,
    start: Optional[date] = None,
    end: Optional[date] = None,
    feature_set: str = "v1",
    label_horizon: int = 5,
    label_task: str = "regression",
    label_thresh: float = 0.05,
    select_cols: Optional[Iterable[str]] = None,
    drop_na_rows: bool = True,
    normalize_features: bool = True,
    cache_invalidate_on: str = "silver_mtime",
) -> str:
    """캐시 키 생성"""
    params = {
        "v": "build_dataset_v3_clean",
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
        "normalize_features": normalize_features,
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
    label_thresh: float = 0.05,
    select_cols: Optional[Iterable[str]] = None,
    drop_na_rows: bool = True,
    verbose: bool = False,
    *,
    use_cache: bool = True,
    cache_dir: str | Path = "data/cache/datasets",
    force_recompute: bool = False,
    cache_invalidate_on: Literal["never", "silver_mtime"] = "silver_mtime",
    normalize_features: bool = True,  # Z-score 정규화 적용 여부
) -> pl.DataFrame:
    """
    데이터셋 빌드 함수
    
    Args:
        years: 연도 리스트
        market: 시장 코드 (KR, US)
        exchanges: 거래소 리스트
        tickers: 티커 리스트
        max_tickers: 최대 티커 수
        start: 시작 날짜
        end: 종료 날짜
        feature_set: 피처 세트 (v1, v2, v3)
        label_horizon: 라벨 예측 기간
        label_task: 라벨 태스크 (regression, classification)
        label_thresh: 분류 임계값
        select_cols: 선택할 컬럼들
        drop_na_rows: NaN 행 제거 여부
        verbose: 상세 출력 여부
        use_cache: 캐시 사용 여부
        cache_dir: 캐시 디렉토리
        force_recompute: 강제 재계산 여부
        cache_invalidate_on: 캐시 무효화 조건
        normalize_features: Z-score 정규화 적용 여부
    
    Returns:
        빌드된 데이터프레임
    """
    if verbose:
        print(f"🚀 Building dataset: {market}, years={years}, normalize={normalize_features}")
    
    # 1. 캐시 확인
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
            normalize_features=normalize_features,
            cache_invalidate_on=cache_invalidate_on,
        )
        cache_dir_path = Path(cache_dir)
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir_path / f"{key[:16]}.parquet"
        
        if cache_path.exists():
            if verbose:
                print(f"[cache] hit: {cache_path}")
            return pl.read_parquet(str(cache_path))
    
    # 2. 데이터 스캔
    if verbose:
        print("📊 Scanning OHLCV data...")
    lf = scan_ohlcv(market=market, tickers=tickers, years=years, exchanges=exchanges)
    _quick_counts(lf, "scan", verbose)
    
    # 3. 데이터 필터링
    if verbose:
        print("🔍 Filtering data...")
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
    
    # 4. 티커 샘플링
    if max_tickers:
        if verbose:
            print(f"🎲 Sampling max {max_tickers} tickers...")
        lf = sample_tickers(lf, max_tickers=max_tickers)
        _quick_counts(lf, "sample", verbose)
    
    # 5. 정렬 (윈도우 함수용)
    if verbose:
        print("📈 Sorting by ticker and date...")
    lf = lf.sort(["ticker", "date"])
    
    # 6. 피처 추가
    if verbose:
        print(f"⚙️ Adding {feature_set} features...")
    lf = add_feature_set(lf, feature_set=feature_set)
    _quick_counts(lf, "features", verbose)
    
    # 7. 라벨 추가
    if verbose:
        print(f"🏷️ Adding labels (horizon={label_horizon}, task={label_task})...")
    lf = future_return_labels(lf, horizon=label_horizon, task=label_task, thresh=label_thresh)
    _quick_counts(lf, "labels", verbose)
    
    # 8. 피처 정규화 (선택적)
    if normalize_features:
        if verbose:
            print("🔄 Applying Z-score normalization...")
        
        # 정규화할 컬럼 식별
        schema_names = lf.collect_schema().names()
        base_cols = {
            "date", "ticker", "market", "exchange", "currency", "year",
            "open", "high", "low", "close", "adj_close", "volume", "turnover"
        }
        
        feature_cols = [
            c for c in schema_names 
            if c not in base_cols 
            and not c.startswith("label_") 
            and not c.startswith("futret_")
            and lf.collect_schema()[c] in (pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64)
        ]
        
        if feature_cols:
            # Z-score 정규화 적용
            normalize_exprs = []
            for col in feature_cols:
                x = pl.col(col)
                z_score = (x - x.mean()) / (x.std() + 1e-9)  # 0으로 나누기 방지
                normalize_exprs.append(z_score.alias(col))
            
            # 정규화된 피처로 교체
            keep_cols = [c for c in schema_names if c not in feature_cols]
            lf = lf.with_columns(normalize_exprs).select(keep_cols + feature_cols)
            
            if verbose:
                print(f"[normalize] Applied Z-score to {len(feature_cols)} features")
        else:
            if verbose:
                print("[normalize] No features to normalize")
    else:
        if verbose:
            print("[normalize] Skipping Z-score normalization")
    
    _quick_counts(lf, "normalized", verbose)
    
    # 9. NaN 처리
    if drop_na_rows:
        if verbose:
            print("🧹 Dropping NaN rows...")
        
        # 피처와 라벨 컬럼에서 NaN 제거
        schema_names = lf.collect_schema().names()
        feat_cols = [
            c for c in schema_names 
            if c not in ["date", "ticker", "market", "exchange", "open", "high", "low", "close", "adj_close", "volume", "turnover", "year"]
        ]
        
        if feat_cols:
            lf = lf.drop_nulls(feat_cols)
            _quick_counts(lf, "drop_na", verbose)
    
    # 10. 컬럼 선택
    if select_cols:
        if verbose:
            print(f"📋 Selecting {len(select_cols)} columns...")
        lf = lf.select(list(select_cols))
    
    # 11. 데이터프레임 수집
    if verbose:
        print("💾 Collecting final dataframe...")
    df = lf.collect(streaming=False)
    
    # 12. 캐시 저장
    if use_cache and cache_path:
        try:
            df.write_parquet(str(cache_path))
            if verbose:
                print(f"[cache] saved: {cache_path}")
        except Exception as e:
            if verbose:
                print(f"[cache] save failed: {e}")
    
    if verbose:
        print(f"✅ Dataset built: {len(df):,} rows × {len(df.columns)} columns")
    
    return df