#!/usr/bin/env python3
"""
M002 Full Architecture 백테스트 (backtesting.py 기반, 병렬 처리 포함)

구성:
- 모델: M002FullArchitecture (predict(df: pl.DataFrame) -> dict)
- 데이터: build_dataset(...) 로 로드 (polars DataFrame)
- 시그널: 모델 예측으로부터 Action(LONG/SHORT/FLAT), Policy Score 등 생성
- per-ticker 백테스트: backtesting.py로 개별 실행, 종목별 HTML 차트 저장
- 결과: 종목별 통계 stats 집계 -> CSV 저장

주의:
- backtesting.py는 종목 단위로 동작하므로 멀티티커는 병렬 루프 처리
- numpy 2.x / polars 최신 호환
"""

import warnings
warnings.filterwarnings("ignore")

import os
import math
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import polars as pl
import pandas as pd
import numpy as np

from joblib import Parallel, delayed
from tqdm import tqdm

from backtesting import Backtest, Strategy

# 프로젝트 내 모듈 경로 추가
import sys
sys.path.append(str(Path(__file__).parent.parent))

from data.dataset_builder import build_dataset
from models.M002_FullArchitecture import (
    M002FullArchitecture,
    FullArchitectureConfig,
    PolicyConfig,
    years_to_slug,
)
from models.M002_MultiTask import M002TrainingConfig
from models.M002_RegimeClassifier import M002RegimeClassifier, RegimeConfig


# =========================
# Strategy (단일 종목용)
# =========================
class M002FullStrategy(Strategy):
    """
    단일 종목용 전략.
    DataFrame에 다음 컬럼이 포함되어야 함:
    - Close (필수), Open/High/Low(없으면 Close로 대체)
    - PolicyScore (float)
    - Action (str: "LONG"|"SHORT"|"FLAT")
    """
    def init(self):
        # backtesting.py 규약: self.data.<Column> 접근 가능
        # 인디케이터 등록은 필요 없고 next에서 직접 참조
        self.min_hold_bars = max(int(getattr(self, "min_hold_bars", 0)), 0)
        self._bars_since_rebalance = self.min_hold_bars  # allow immediate first trade

    def next(self):
        i = len(self.data.Close) - 1
        action = self.data.Action[i]
        target_frac = float(self.data.PositionSize[i]) if hasattr(self.data, "PositionSize") else 0.0

        if not math.isfinite(target_frac):
            target_frac = 0.0

        if action == "LONG":
            target_frac = max(0.0, target_frac)
        else:
            target_frac = 0.0

        target_frac = min(target_frac, 1.0)

        price = float(self.data.Close[i])
        if price <= 0 or not math.isfinite(price):
            return

        equity = float(self.equity)
        if not math.isfinite(equity) or equity <= 0:
            return

        if target_frac > 0:
            desired_units = max(1.0, target_frac * equity / price)
        else:
            desired_units = 0.0
        target_units = int(round(desired_units)) if desired_units > 0 else 0
        if target_units == 0 and target_frac > 0:
            target_units = 1

        current_units = int(round(self.position.size))
        delta_units = target_units - current_units
        delta_abs = abs(delta_units)
        if delta_abs < 1e-6:
            self._bars_since_rebalance += 1
            return

        if self.min_hold_bars > 0 and self._bars_since_rebalance < self.min_hold_bars:
            self._bars_since_rebalance += 1
            return

        def _submit(order_func, units: int):
            units = max(1, int(units))
            order_func(size=units)

        traded = False
        if delta_units > 0:
            _submit(self.buy, delta_units)
            traded = True
        elif delta_units < 0:
            _submit(self.sell, -delta_units)
            traded = True

        if traded:
            self._bars_since_rebalance = 0
        else:
            self._bars_since_rebalance += 1

# =========================
# Backtester
# =========================
class M002FullBacktester:
    def __init__(
        self,
        model: M002FullArchitecture,
        commission: float = 0.001,
        initial_cash: float = 10_000,
        n_jobs: int = 8,                # 병렬 작업 수
        chart_tickers: Optional[List[str]] = None,  # HTML 저장 대상 (None이면 생성 안 함)
        min_hold_bars: int = 3,
        robust_min_trades: int = 30,
        robust_min_dd_pct: float = 1.0,
    ):
        self.model = model
        self.commission = commission
        self.initial_cash = initial_cash
        self.n_jobs = n_jobs
        self.chart_tickers = chart_tickers or []
        self.min_hold_bars = max(int(min_hold_bars), 0)
        self.robust_min_trades = max(int(robust_min_trades), 0)
        self.robust_min_dd_pct = max(float(robust_min_dd_pct), 0.0)

    # ---------- 데이터 준비 ----------
    def prepare_data(
        self,
        market: str = "US",
        years: List[int] = list(range(2000, 2019)),
        max_tickers: int = 100
    ) -> pl.DataFrame:
        print(f"[데이터 준비] market={market}, years={years}, max_tickers={max_tickers}")
        cfg = self.model.config
        df = build_dataset(
            years=list(years),
            market=market,
            max_tickers=max_tickers,
            feature_set=cfg.feature_set,
            label_horizon=cfg.horizon,
            label_task="regression",
            verbose=False,
            normalize_features=cfg.normalize_features
        )
        print(f"  로드: {len(df):,} rows × {len(df.columns)} cols")
        print(f"  기간: {df['date'].min()} ~ {df['date'].max()}")
        print(f"  종목: {df['ticker'].n_unique()}개")

        # 데이터 품질 검증 및 필터링
        print(f"\n[데이터 품질 검증]")

        # 거래량이 0인 데이터 비율 확인
        bad_tickers: List[str] = []
        zero_volume = df.filter(pl.col("volume") == 0)
        if len(zero_volume) > 0:
            zero_volume_by_ticker = zero_volume.group_by("ticker").agg(pl.len().alias("zero_count"))
            total_by_ticker = df.group_by("ticker").agg(pl.len().alias("total_count"))
            quality_check = zero_volume_by_ticker.join(total_by_ticker, on="ticker").with_columns(
                (pl.col("zero_count") / pl.col("total_count") * 100).alias("zero_pct")
            )
            print("  거래량 0 비율 (티커별):")
            for row in quality_check.sort("zero_pct", descending=True).head(10).iter_rows():
                ticker, zero_count, total_count, zero_pct = row
                print(f"    {ticker}: {zero_count}/{total_count} ({zero_pct:.1f}%)")

            # 거래량이 0인 티커는 제외 (품질이 너무 낮음) - 2019년은 덜 엄격하게
            bad_tickers = quality_check.filter(pl.col("zero_pct") > 80).select("ticker").to_series().to_list()

        if bad_tickers:
            print(f"  품질 낮은 티커 제외: {bad_tickers}")
            df = df.filter(~pl.col("ticker").is_in(bad_tickers))
            print(f"  필터링 후: {len(df):,} rows, {df['ticker'].n_unique()}개 티커")

        return df

    # ---------- 예측/시그널 생성 ----------
    def generate_signals(self, df: pl.DataFrame) -> pl.DataFrame:
        """
        모델 예측 → policy_score / position_size / action / expected_return 열 추가
        그리고 날짜별 abs(policy_score) 기준 top-N만 거래, 나머지는 FLAT으로 강제
        """
        print("\n[M002 Full 시그널 생성]")
        print(f"  입력 데이터: {len(df)} rows")
        print(f"  컬럼 수: {len(df.columns)}")
        print(f"  첫 3개 컬럼: {df.columns[:3]}")
        print(f"  마지막 3개 컬럼: {df.columns[-3:]}")

        # 입력 데이터 샘플 확인
        print(f"\n[입력 데이터 샘플 - ACG 티커만]")
        acg_data = df.filter(pl.col("ticker") == "ACG").head(3)
        if len(acg_data) > 0:
            print(acg_data.select(["ticker", "date", "close", "volume"]).to_pandas())

        # I_bd_early 등의 컬럼 존재 여부 확인
        indicator_cols = ["I_bd_early", "I_bd_late", "I_vr_and_vs", "atr_smooth"]
        print(f"\n[인디케이터 컬럼 확인]:")
        for col in indicator_cols:
            exists = col in df.columns
            print(f"  {col}: {'있음' if exists else '없음'}")
            if exists:
                # 값 분포 확인
                unique_vals = df[col].unique().to_list()
                print(f"    고유값: {unique_vals[:10]}..." if len(unique_vals) > 10 else f"    고유값: {unique_vals}")

        # AAWW 티커의 인디케이터 값 확인
        if "AAWW" in df["ticker"].unique():
            aaww_data = df.filter(pl.col("ticker") == "AAWW").head(5)
            print(f"\n[AAWW 인디케이터 값]:")
            cols_to_show = ["ticker", "date", "close"] + [c for c in indicator_cols if c in df.columns]
            print(aaww_data.select(cols_to_show).to_pandas())

        df_sorted = df.sort(["ticker", "date"])
        pred = self.model.predict(df_sorted)
        
        print(f"\n[예측 결과 디버그]")
        print(f"  pred 타입: {type(pred)}")
        print(f"  pred 컬럼들: {list(pred.columns) if hasattr(pred, 'columns') else 'N/A'}")
        print(f"  pred shape: {pred.shape if hasattr(pred, 'shape') else 'N/A'}")
        print(f"\n  예측 결과 샘플 (처음 5개):")
        print(pred.head())
        print(f"\n  Action 분포:")
        print(pred['action'].value_counts() if 'action' in pred else 'action 컬럼 없음')

        if "position_size" in pred:
            pos_abs = np.abs(pred["position_size"].to_numpy(dtype=float))
            mean_abs = float(np.nanmean(pos_abs))
            max_abs = float(np.nanmax(pos_abs))
            print(f"\n[포지션 사이즈 점검] |size| mean={mean_abs:.4f}, max={max_abs:.4f}")
            if max_abs > 10 or mean_abs > 1:
                print("  ⚠️ 포지션 사이즈 스케일이 비정상적으로 큽니다. 피처 정규화/스케일링 점검 필요.")

        # 데이터 품질 확인
        print(f"\n[데이터 품질 확인]")
        good_tickers = pred['ticker'].unique()[:3]  # 필터링 후 남은 티커들
        for ticker in good_tickers:
            ticker_pred = pred[pred['ticker'] == ticker]
            policy_unique = ticker_pred['policy_score'].nunique()
            action_dist = ticker_pred['action'].value_counts()
            print(f"  {ticker}: policy_score {policy_unique} 고유값, actions: {dict(action_dist)}")

        pred_pl = pl.from_pandas(pred)[
            ["ticker", "date", "policy_score", "position_size", "action", "pred_expected_ret_pct"]
        ].rename({"action": "action_raw", "pred_expected_ret_pct": "expected_return"})
        if "date" in pred_pl.columns and "date" in df.columns:
            pred_pl = pred_pl.with_columns(pl.col("date").cast(df.schema["date"]))

        policy_cfg = self.model.config.policy

        df_ranked = (
            df.join(pred_pl, on=["ticker", "date"], how="left")
              .with_columns([
                  pl.col("action_raw").fill_null("FLAT"),
                  pl.col("position_size").fill_null(0.0),
                  pl.col("policy_score").fill_null(0.0),
                  pl.col("expected_return").fill_null(0.0),
              ])
              .with_columns([
                  pl.col("action_raw").alias("action"),
                  pl.when(pl.col("action_raw") == pl.lit("LONG"))
                    .then(pl.col("position_size").clip(0.0, policy_cfg.size_max))
                    .otherwise(0.0)
                    .alias("position_size"),
              ])
        )

        # 액션 분포 출력
        stats = df_ranked.group_by("action").agg(pl.len().alias("count")).sort("count", descending=True)
        total = len(df_ranked)
        print("  Action 분포:")
        for a, c in stats.iter_rows():
            print(f"    {a:>5}: {c:,} ({c/total:.1%})")

        size_summary = df_ranked.select([
            pl.col("position_size").abs().mean().alias("mean_abs"),
            pl.col("position_size").abs().max().alias("max_abs")
        ]).to_dict(as_series=False)
        size_long_summary = df_ranked.filter(pl.col("action") == "LONG").select([
            pl.col("position_size").abs().mean().alias("mean_abs"),
            pl.col("position_size").abs().max().alias("max_abs")
        ]).to_dict(as_series=False)
        if size_summary:
            mean_abs = size_summary.get("mean_abs", [0.0])[0]
            max_abs = size_summary.get("max_abs", [0.0])[0]
            print(f"  최종 포지션 사이즈 |mean|={mean_abs:.4f}, |max|={max_abs:.4f}")
        if size_long_summary and size_long_summary.get("mean_abs"):
            mean_abs_long = size_long_summary["mean_abs"][0]
            max_abs_long = size_long_summary["max_abs"][0]
            print(f"  LONG 행 한정 |mean|={mean_abs_long:.4f}, |max|={max_abs_long:.4f}")
            if mean_abs_long < 0.02:
                print("  ⚠️ LONG 구간 평균 사이즈가 매우 낮습니다. 시그널 밀도/클리핑 로직을 확인하세요.")

        # 거래(롱/숏) 비율
        traded = df_ranked.filter(pl.col("action").is_in(["LONG", "SHORT"]))
        print(f"  Long/Short 레코드: {len(traded):,} / {total:,} ({len(traded)/total:.1%})")

        return df_ranked

    # ---------- 단일 종목 실행 ----------
    def _run_single_ticker(self, tdf: pl.DataFrame, save_dir: str, save_chart: bool) -> Dict[str, Any]:
        """
        단일 종목 backtesting.py 실행. 결과 dict 반환.
        """
        tdf = tdf.sort("date")
        # backtesting.py용 pandas DF 구성
        pdf = tdf.select(["date", "close", "policy_score", "position_size", "action"]).to_pandas()
        pdf.rename(
            columns={
                "date": "Date",
                "close": "Close",
                "policy_score": "PolicyScore",
                "position_size": "PositionSize",
                "action": "Action",
            },
            inplace=True,
        )

        # OHLC 컬럼 보정 (없으면 Close로 채움)
        for col in ("Open", "High", "Low"):
            pdf[col] = pdf["Close"]
        if "Volume" not in pdf:
            pdf["Volume"] = 0.0

        pdf.set_index("Date", inplace=True)

        strategy_cls = type(
            "ConfiguredM002FullStrategy",
            (M002FullStrategy,),
            {"min_hold_bars": self.min_hold_bars},
        )

        bt = Backtest(
            pdf,
            strategy_cls,
            cash=self.initial_cash,
            commission=self.commission,
            trade_on_close=False,
            exclusive_orders=True,
            finalize_trades=True
        )
        stats = bt.run()

        # 차트 저장 (과도한 파일 생성을 피하기 위해 일부만)
        if save_chart:
            chart_path = os.path.join(save_dir, f"chart_{tdf['ticker'][0]}.html")
            bt.plot(open_browser=False, filename=chart_path)

        # backtesting.Stats는 dict 변환 가능
        sdict = dict(stats)
        sdict["Ticker"] = tdf["ticker"][0]
        sdict["Start"] = pdf.index.min()
        sdict["End"] = pdf.index.max()
        return sdict

    # ---------- 전체 실행 ----------
    def run(
        self,
        market: str = "US",
        years: List[int] = list(range(2000, 2019)),
        max_tickers: int = 1000,
        save_dir: str = "reports/m002_full_backtest_btpy"
    ) -> pd.DataFrame:
        print("\n" + "=" * 60)
        print("[M002 Full Architecture 백테스트 시작 - backtesting.py]")
        print("=" * 60)

        Path(save_dir).mkdir(parents=True, exist_ok=True)

        # 1) 데이터
        df = self.prepare_data(market=market, years=list(years), max_tickers=max_tickers)

        # 2) 시그널
        df_sig = self.generate_signals(df)

        # 3) 종목 리스트
        tickers = df_sig.get_column("ticker").unique().to_list()
        print(f"\n[실행 대상 종목] {len(tickers)}개")

        # 4) 병렬 실행
        if self.chart_tickers:
            requested = set(self.chart_tickers)
            available = set(tickers)
            missing = requested - available
            if missing:
                print(f"  ⚠️ 차트 요청 티커 미존재: {sorted(missing)}")
            chart_set = requested & available
            if chart_set:
                print(f"  💾 HTML 저장 대상: {sorted(chart_set)}")
        else:
            chart_set = set()

        def run_ticker(t: str):
            tdf = df_sig.filter(pl.col("ticker") == t)
            return self._run_single_ticker(
                tdf=tdf,
                save_dir=save_dir,
                save_chart=(t in chart_set)
            )

        results: List[Dict[str, Any]] = Parallel(n_jobs=self.n_jobs, prefer="processes")(
            delayed(run_ticker)(t) for t in tqdm(tickers, desc="Tickers")
        )

        # 5) 결과 집계 및 저장
        stats_df = pd.DataFrame(results)
        stats_path = os.path.join(save_dir, "summary_stats.csv")
        stats_df.to_csv(stats_path, index=False)
        print(f"\n✅ 요약 통계 저장: {stats_path}")

        # 6) 간단한 집계 프린트
        if not stats_df.empty:
            cols_pref = [c for c in [
                "Ticker", "Start", "End",
                "Equity Final [$]", "Equity Peak [$]",
                "Return [%]", "Buy & Hold Return [%]",
                "Max. Drawdown [%]", "Win Rate [%]", "Sharpe Ratio"
            ] if c in stats_df.columns]
            print("\n[샘플 결과 5개]")
            print(stats_df[cols_pref].head(5).to_string(index=False))

            # 전체 평균/중앙값도 저장
            agg = stats_df.select_dtypes(include=[np.number]).agg(["mean", "median"]).T
            agg_path = os.path.join(save_dir, "summary_agg.csv")
            agg.to_csv(agg_path)
            print(f"✅ 통계 요약(평균/중앙값) 저장: {agg_path}")
            self._save_robust_summary(stats_df, save_dir)

        return stats_df

    def _save_robust_summary(self, stats_df: pd.DataFrame, save_dir: str) -> None:
        """
        trades/Drawdown 기준으로 필터링한 뒤 안정적인 집계를 별도 저장.
        """
        df = stats_df.copy()
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if "# Trades" not in df.columns or "Max. Drawdown [%]" not in df.columns:
            print("⚠️  로버스트 요약을 위한 필수 컬럼이 없어 건너뜀.")
            return

        df["# Trades"] = pd.to_numeric(df["# Trades"], errors="coerce")
        df["Max. Drawdown [%]"] = pd.to_numeric(df["Max. Drawdown [%]"], errors="coerce")
        mask = (
            (df["# Trades"] >= self.robust_min_trades)
            & (df["Max. Drawdown [%]"].abs() >= self.robust_min_dd_pct)
        )
        filtered = df[mask]
        if filtered.empty:
            print("⚠️  로버스트 요약: 조건을 만족하는 티커가 없어 생성하지 않음.")
            return

        robust_numeric = filtered.select_dtypes(include=[np.number]).agg(["mean", "median"]).T
        robust_numeric.insert(0, "count", filtered.shape[0])
        robust_path = os.path.join(save_dir, "summary_agg_robust.csv")
        robust_numeric.to_csv(robust_path)
        print(
            f"✅ 로버스트 통계 저장: {robust_path} "
            f"(조건: #Trades≥{self.robust_min_trades}, |MaxDD|≥{self.robust_min_dd_pct}%)"
        )


# =========================
# main
# =========================
def main():
    print("[M002 Full Architecture 모델 로드/준비]")
    import joblib


    year_slug = years_to_slug(M002TrainingConfig().years)
    model_path = f"models/saved/m002_full_architecture_US_{year_slug}.pkl"

    try:
        model = joblib.load(model_path)
        print(f"  ✅ 모델 로드: {model_path}")
        print(f"  📊 시장: {model.config.multitask.market}")
        print(f"  📊 학습 연도: {model.config.multitask.years}")
        print(f"  📊 예측 기간: {model.config.horizon}일")
        print(f"  📊 위험회피 λ: {model.config.policy.risk_aversion}")

    except FileNotFoundError:
        print(f"  ❌ 모델 파일 없음: {model_path}")
        print("  🔄 새 모델 학습 진행...")

        config = FullArchitectureConfig(
            horizon=5,
            feature_set="m002",
            normalize_features=True
        )
        model = M002FullArchitecture(config=config)

        train_df = build_dataset(
            years=list(config.multitask.years),
            market="US",
            max_tickers=100,
            feature_set=config.feature_set,
            label_horizon=config.horizon,
            label_task="regression",
            verbose=False,
            normalize_features=config.normalize_features
        )

        print("\n[모델 학습]")
        model.train(train_df)
        Path(Path(model_path).parent).mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        print(f"  💾 모델 저장: {model_path}")

    # 백테스터 실행
    chart_env = os.environ.get("M002_CHART_TICKERS", "").strip()
    chart_tickers = [token.strip().upper() for token in chart_env.split(",") if token.strip()] if chart_env else []
    if chart_tickers:
        print(f"  💾 HTML 차트 저장 대상: {chart_tickers}")

    backtester = M002FullBacktester(
        model=model,
        commission=0.0002,
        initial_cash=10_000,
        n_jobs=64,
        chart_tickers=chart_tickers,
        min_hold_bars=0,
    )

    stats_df = backtester.run(
        market="US",
        years=list(model.config.multitask.years),
        max_tickers=100,
        save_dir="reports/m002_full_backtest_btpy"
    )
    return stats_df


if __name__ == "__main__":
    try:
        _ = main()
        print("\n✅ M002 Full Architecture 백테스트 완료 (backtesting.py)!")
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
