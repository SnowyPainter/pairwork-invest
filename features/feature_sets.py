# features/feature_sets.py
import numpy as np
import polars as pl

def add_core_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    g = "ticker"
    close, high, low, vol = pl.col("close"), pl.col("high"), pl.col("low"), pl.col("volume").fill_null(0.0)

    # True Range 계산을 위한 중간 단계들
    prev_close = close.shift(1).over(g)
    hl_diff = high - low
    hc_diff = (high - prev_close).abs()
    lc_diff = (low - prev_close).abs()
    
    tr_raw = pl.max_horizontal(hl_diff, hc_diff, lc_diff)
    typical = (high + low + close) / 3

    return (
        lf.with_columns([
            # logret_1: close의 log 차분
            (close.log() - close.log().shift(1).over(g)).alias("logret_1"),

            # ret_5: 5일 수익률
            (close / close.shift(5).over(g) - 1).alias("ret_5"),

            # True Range 중간 계산
            tr_raw.alias("tr_raw"),
            typical.alias("typical"),
        ])
        .with_columns([
            # ATR: 다양한 윈도우 적용 (5, 10, 14일)
            pl.col("tr_raw").rolling_mean(5).over(g).fill_null(pl.col("tr_raw")).alias("atr5"),
            pl.col("tr_raw").rolling_mean(10).over(g).fill_null(pl.col("tr_raw").rolling_mean(5).over(g)).alias("atr10"),
            pl.col("tr_raw").rolling_mean(14).over(g).fill_null(pl.col("tr_raw").rolling_mean(10).over(g)).alias("atr14"),

            # VWAP: 다양한 윈도우 적용 (10, 20일)
            (pl.col("typical") * vol).rolling_sum(10).over(g).alias("vwap_num10"),
            vol.rolling_sum(10).over(g).alias("vwap_den10"),
            (pl.col("typical") * vol).rolling_sum(20).over(g).alias("vwap_num20"),
            vol.rolling_sum(20).over(g).alias("vwap_den20"),
        ])
        .with_columns([
            # VWAP 최종 계산 (null 대체 체인)
            (pl.col("vwap_num10") / (pl.col("vwap_den10") + 1e-9)).alias("vwap10"),
            (pl.col("vwap_num20") / (pl.col("vwap_den20") + 1e-9)).alias("vwap20_raw"),
        ])
        .with_columns([
            # VWAP20에 null 대체 적용
            pl.col("vwap20_raw").fill_null(pl.col("vwap10")).alias("vwap20"),
        ])
        .with_columns([
            # RSI 계산을 위한 gain/loss 계산
            (close - close.shift(1).over(g)).alias("price_change"),
        ])
        .with_columns([
            # gain과 loss를 별도로 계산
            pl.when(pl.col("price_change") > 0).then(pl.col("price_change")).otherwise(0.0).alias("gain_raw"),
            pl.when(pl.col("price_change") < 0).then(-pl.col("price_change")).otherwise(0.0).alias("loss_raw"),
        ])
        .with_columns([
            # RSI: 다양한 윈도우 적용 (5, 10, 14일)
            pl.col("gain_raw").rolling_mean(5).over(g).fill_null(pl.col("gain_raw")).alias("gain5"),
            pl.col("loss_raw").rolling_mean(5).over(g).fill_null(pl.col("loss_raw")).alias("loss5"),
            pl.col("gain_raw").rolling_mean(10).over(g).alias("gain10"),
            pl.col("loss_raw").rolling_mean(10).over(g).alias("loss10"),
            pl.col("gain_raw").rolling_mean(14).over(g).alias("gain14"),
            pl.col("loss_raw").rolling_mean(14).over(g).alias("loss14"),
        ])
        .with_columns([
            # RSI 계산 (null 대체 체인 적용)
            (100 - 100/(1 + (pl.col("gain5")/(pl.col("loss5")+1e-9)))).alias("rsi5"),
            (100 - 100/(1 + (pl.col("gain10").fill_null(pl.col("gain5"))/(pl.col("loss10").fill_null(pl.col("loss5"))+1e-9)))).alias("rsi10"),
            (100 - 100/(1 + (pl.col("gain14").fill_null(pl.col("gain10"))/(pl.col("loss14").fill_null(pl.col("loss10"))+1e-9)))).alias("rsi14")
        ])
        .drop(["tr_raw", "typical", "vwap_num10", "vwap_den10", "vwap_num20", "vwap_den20", "vwap20_raw", 
               "price_change", "gain_raw", "loss_raw", "gain5", "loss5", "gain10", "loss10", "gain14", "loss14"])
    )


# ==== 유틸 ====
def _tp() -> pl.Expr:
    return (pl.col("high") + pl.col("low") + pl.col("close")) / 3

def _true_range() -> pl.Expr:
    prev_close = pl.col("close").shift(1).over("ticker")
    return pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low")  - prev_close).abs(),
    )

def _safe_div(a: pl.Expr, b: pl.Expr, eps: float=1e-9) -> pl.Expr:
    return a / (b + eps)

def _poly_second_derivative(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.all(~np.isfinite(arr)):
        return np.nan

    mask = np.isfinite(arr)
    if mask.sum() < 5:
        return np.nan

    x = np.arange(mask.sum(), dtype=float)
    y = arr[mask]

    deg = min(3, int(mask.sum()) - 1)
    if deg < 2:
        return np.nan

    try:
        coeffs = np.polyfit(x, y, deg=deg)
        second = np.polyder(coeffs, 2)
        mid = x[len(x) // 2]
        return float(np.polyval(second, mid))
    except (np.linalg.LinAlgError, ValueError):
        return np.nan

# ==== v2 확장 ====
def add_v2_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    g = "ticker"

    W_S = [5, 10, 20, 50]
    W_VWAP = [10, 20]

    return (
        lf
        # 1) 기본 이동평균들 추가
        .with_columns([
            pl.col("close").rolling_mean(w).over(g).alias(f"sma{w}") for w in W_S
        ])
        .with_columns([
            pl.col("close").ewm_mean(alpha=2/(w+1), adjust=False).over(g).alias(f"ema{w}") for w in W_S
        ])
        # 2) ROC 지표들
        .with_columns([
            (pl.col("close")/pl.col("close").shift(w).over(g)-1).alias(f"roc{w}") for w in [5,10,20]
        ])
        # 3) RSI6 계산을 위한 중간 단계 (null 처리 강화)
        .with_columns([
            (pl.col("close") - pl.col("close").shift(1).over(g)).alias("price_change_6")
        ])
        .with_columns([
            pl.when(pl.col("price_change_6") > 0).then(pl.col("price_change_6")).otherwise(0.0).alias("gain6_raw"),
            pl.when(pl.col("price_change_6") < 0).then(-pl.col("price_change_6")).otherwise(0.0).alias("loss6_raw")
        ])
        .with_columns([
            pl.col("gain6_raw").rolling_mean(6).over(g).fill_null(pl.col("gain6_raw")).alias("gain6"),
            pl.col("loss6_raw").rolling_mean(6).over(g).fill_null(pl.col("loss6_raw")).alias("loss6")
        ])
        .with_columns([
            (100 - 100/(1 + _safe_div(pl.col("gain6"), pl.col("loss6")))).alias("rsi6")
        ])
        # 4) Stochastic 지표들 (다양한 윈도우 + null 처리)
        .with_columns([
            pl.col("low").rolling_min(5).over(g).alias("lowest5"),
            pl.col("high").rolling_max(5).over(g).alias("highest5"),
            pl.col("low").rolling_min(14).over(g).alias("lowest14"),
            pl.col("high").rolling_max(14).over(g).alias("highest14")
        ])
        .with_columns([
            (_safe_div(pl.col("close") - pl.col("lowest5"), pl.col("highest5") - pl.col("lowest5")) * 100).alias("stochk5"),
            (_safe_div(pl.col("close") - pl.col("lowest14"), pl.col("highest14") - pl.col("lowest14")) * 100).alias("stochk14_raw")
        ])
        .with_columns([
            pl.col("stochk14_raw").fill_null(pl.col("stochk5")).alias("stochk14")
        ])
        .with_columns([
            pl.col("stochk14").rolling_mean(3).over(g).fill_null(pl.col("stochk14")).alias("stochd14")
        ])
        .with_columns([
            (pl.col("stochk14") - pl.col("stochd14")).alias("stoch_spread")
        ])
        # 5) Williams %R (null 대체 적용)
        .with_columns([
            (-_safe_div(pl.col("highest14") - pl.col("close"), pl.col("highest14") - pl.col("lowest14")) * 100).alias("willr14_raw"),
            (-_safe_div(pl.col("highest5") - pl.col("close"), pl.col("highest5") - pl.col("lowest5")) * 100).alias("willr5")
        ])
        .with_columns([
            pl.col("willr14_raw").fill_null(pl.col("willr5")).alias("willr14")
        ])
        # 6) CCI 계산 (다양한 윈도우 + null 처리)
        .with_columns([
            _tp().alias("tp")
        ])
        .with_columns([
            pl.col("tp").rolling_mean(10).over(g).alias("sma_tp10"),
            pl.col("tp").rolling_mean(20).over(g).alias("sma_tp20")
        ])
        .with_columns([
            (pl.col("tp") - pl.col("sma_tp10")).abs().alias("dev_tp10"),
            (pl.col("tp") - pl.col("sma_tp20")).abs().alias("dev_tp20")
        ])
        .with_columns([
            pl.col("dev_tp10").rolling_mean(10).over(g).alias("md10"),
            pl.col("dev_tp20").rolling_mean(20).over(g).alias("md20")
        ])
        .with_columns([
            _safe_div(pl.col("tp") - pl.col("sma_tp10"), 0.015 * pl.col("md10")).alias("cci10"),
            _safe_div(pl.col("tp") - pl.col("sma_tp20"), 0.015 * pl.col("md20")).alias("cci20_raw")
        ])
        .with_columns([
            pl.col("cci20_raw").fill_null(pl.col("cci10")).alias("cci20")
        ])
        # 7) MACD 지표
        .with_columns([
            pl.col("close").ewm_mean(alpha=2/13, adjust=False).over(g).alias("ema12"),
            pl.col("close").ewm_mean(alpha=2/27, adjust=False).over(g).alias("ema26")
        ])
        .with_columns([
            (pl.col("ema12") - pl.col("ema26")).alias("macd")
        ])
        .with_columns([
            pl.col("macd").ewm_mean(alpha=2/10, adjust=False).over(g).alias("macd_signal")
        ])
        .with_columns([
            (pl.col("macd") - pl.col("macd_signal")).alias("macd_hist")
        ])
        # 8) 변동성 지표들
        .with_columns([
            pl.col("logret_1").rolling_std(10).over(g).alias("vol10"),
            pl.col("logret_1").rolling_std(20).over(g).alias("vol20")
        ])
        .with_columns([
            (((pl.col("high") / pl.col("low")).log() ** 2) / (4.0 * 0.6931471805599453)).alias("parkinson_raw")
        ])
        .with_columns([
            pl.col("parkinson_raw").rolling_mean(20).over(g).alias("parkinson20")
        ])
        .with_columns([
            (0.5 * ((pl.col("high") / pl.col("low")).log() ** 2) 
             - (2.0 * 0.6931471805599453 - 1.0) * ((pl.col("close") / pl.col("open")).log() ** 2)).alias("gk_raw")
        ])
        .with_columns([
            pl.col("gk_raw").rolling_mean(20).over(g).alias("gk20")
        ])
        # 9) True Range 관련
        .with_columns([
            _true_range().alias("tr")
        ])
        .with_columns([
            pl.col("tr").rolling_mean(14).over(g).alias("tr14")
        ])
        # 10) OBV 계산
        .with_columns([
            pl.when(pl.col("close") > pl.col("close").shift(1).over(g)).then(pl.col("volume"))
             .when(pl.col("close") < pl.col("close").shift(1).over(g)).then(-pl.col("volume"))
             .otherwise(0).alias("obv_delta")
        ])
        .with_columns([
            pl.col("obv_delta").cum_sum().over(g).alias("obv")
        ])
        # 11) MFI 계산
        .with_columns([
            (pl.col("tp") * pl.col("volume")).alias("raw_flow")
        ])
        .with_columns([
            pl.when(pl.col("tp") > pl.col("tp").shift(1).over(g)).then(pl.col("raw_flow")).otherwise(0).alias("pos_flow"),
            pl.when(pl.col("tp") < pl.col("tp").shift(1).over(g)).then(pl.col("raw_flow")).otherwise(0).alias("neg_flow")
        ])
        .with_columns([
            pl.col("pos_flow").rolling_sum(14).over(g).alias("pos_flow_sum14"),
            pl.col("neg_flow").rolling_sum(14).over(g).alias("neg_flow_sum14")
        ])
        .with_columns([
            _safe_div(pl.col("pos_flow_sum14"), pl.col("neg_flow_sum14")).alias("mfr14")
        ])
        .with_columns([
            (100 - 100/(1 + pl.col("mfr14"))).alias("mfi14")
        ])
        # 12) CMF 계산
        .with_columns([
            _safe_div((pl.col("close")-pl.col("low")) - (pl.col("high")-pl.col("close")),
                     (pl.col("high")-pl.col("low"))).alias("mfm")
        ])
        .with_columns([
            (pl.col("mfm") * pl.col("volume")).rolling_sum(20).over(g).alias("cmf_num"),
            pl.col("volume").rolling_sum(20).over(g).alias("cmf_den")
        ])
        .with_columns([
            _safe_div(pl.col("cmf_num"), pl.col("cmf_den")).alias("cmf20")
        ])
        # 13) Multi-window VWAP
        .with_columns([
            _safe_div((_tp() * pl.col("volume")).rolling_sum(w).over(g),
                     pl.col("volume").rolling_sum(w).over(g)).alias(f"vwap{w}")
            for w in W_VWAP
        ])
        # 14) Volume 지표들
        .with_columns([
            pl.col("volume").rolling_mean(20).over(g).alias("vol_ma20"),
            pl.col("volume").rolling_std(20).over(g).alias("vol_std20")
        ])
        .with_columns([
            _safe_div(pl.col("volume") - pl.col("vol_ma20"), pl.col("vol_std20")).alias("vol_z20"),
            (_safe_div(pl.col("volume"), pl.col("volume").shift(5).over(g)) - 1).alias("vol_roc5")
        ])
        # 15) Donchian 채널
        .with_columns([
            pl.col("high").rolling_max(20).over(g).alias("don20_hi"),
            pl.col("low").rolling_min(20).over(g).alias("don20_lo"),
            pl.col("high").rolling_max(55).over(g).alias("don55_hi"),
            pl.col("low").rolling_min(55).over(g).alias("don55_lo")
        ])
        .with_columns([
            _safe_div(pl.col("close") - pl.col("don20_lo"), pl.col("don20_hi") - pl.col("don20_lo")).alias("pos_in_don20")
        ])
        # 16) 가격 구조 지표들
        .with_columns([
            (pl.col("high") - pl.col("low")).alias("day_range"),
            (pl.col("close") - pl.col("open")).alias("body"),
            (pl.col("open") - pl.col("close").shift(1).over(g)).alias("gap")
        ])
        .with_columns([
            _safe_div(pl.col("day_range"), pl.col("close")).alias("rel_range")
        ])
        # 17) 상관관계 계산을 위한 중간 컬럼들
        .with_columns([
            pl.col("close").rolling_mean(20).over(g).alias("close_mean20"),
            pl.col("volume").rolling_mean(20).over(g).alias("vol_mean20_corr"),
            pl.col("close").rolling_std(20).over(g).alias("close_std20"),
            pl.col("volume").rolling_std(20).over(g).alias("vol_std20_corr")
        ])
        .with_columns([
            pl.col("obv").rolling_mean(20).over(g).alias("obv_mean20"),
            pl.col("vwap20").rolling_mean(20).over(g).alias("vwap20_mean20"),
            pl.col("obv").rolling_std(20).over(g).alias("obv_std20"),
            pl.col("vwap20").rolling_std(20).over(g).alias("vwap20_std20")
        ])
        # 18) 상관관계 계산
        .with_columns([
            ((pl.col("close") - pl.col("close_mean20")) * (pl.col("volume") - pl.col("vol_mean20_corr"))).rolling_mean(20).over(g).alias("close_vol_cov20"),
            ((pl.col("obv") - pl.col("obv_mean20")) * (pl.col("vwap20") - pl.col("vwap20_mean20"))).rolling_mean(20).over(g).alias("obv_vwap_cov20")
        ])
        .with_columns([
            _safe_div(pl.col("close_vol_cov20"), pl.col("close_std20") * pl.col("vol_std20_corr")).alias("corr_close_vol20"),
            _safe_div(pl.col("obv_vwap_cov20"), pl.col("obv_std20") * pl.col("vwap20_std20")).alias("corr_obv_vwap20")
        ])
        # 19) 곡률 계산
        .with_columns([
            (pl.col("close") - pl.col("close").shift(1).over(g)).alias("d1")
        ])
        .with_columns([
            (pl.col("d1") - pl.col("d1").shift(1).over(g)).alias("d2")
        ])
        .with_columns([
            pl.col("d2").rolling_mean(5).over(g).alias("curv5")
        ])
        # 20) 파생 지표들
        .with_columns([
            _safe_div(pl.col("close") - pl.col("sma20"), pl.col("sma20")).alias("dev_sma20"),
            _safe_div(pl.col("close") - pl.col("ema20"), pl.col("ema20")).alias("dev_ema20"),
            _safe_div(pl.col("sma20") - pl.col("sma50"), pl.col("sma50")).alias("ma_squeeze_20_50"),
            _safe_div(pl.col("ema12") - pl.col("ema26"), pl.col("close")).alias("ema12_26_rel")
        ])
        # 21) 중간 컬럼들 정리
        .drop([
            "price_change_6", "gain6_raw", "loss6_raw", "gain6", "loss6",
            "lowest5", "highest5", "lowest14", "highest14", "stochk14_raw", "willr14_raw", "willr5",
            "tp", "sma_tp10", "sma_tp20", "dev_tp10", "dev_tp20", "md10", "md20", "cci10", "cci20_raw",
            "ema12", "ema26", "parkinson_raw", "gk_raw", "tr", "obv_delta",
            "raw_flow", "pos_flow", "neg_flow", "pos_flow_sum14", "neg_flow_sum14", "mfr14",
            "mfm", "cmf_num", "cmf_den", "vol_ma20", "vol_std20", "day_range", "body", "gap",
            "close_mean20", "vol_mean20_corr", "close_std20", "vol_std20_corr",
            "obv_mean20", "vwap20_mean20", "obv_std20", "vwap20_std20",
            "close_vol_cov20", "obv_vwap_cov20", "d1", "d2"
        ])
        # 22) NaN/Null 처리 완화 (forward fill + 0 대체)
        .with_columns([
            pl.col(pl.FLOAT_DTYPES).fill_nan(None).forward_fill().over(g).fill_null(0),
            pl.col(pl.INTEGER_DTYPES).fill_null(0)
        ])
    )

def add_v3_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    g = "ticker"

    return (
        lf
        # ADX, +DI, -DI 계산
        .with_columns([
            _true_range().alias("tr_v3"),
            (pl.col("high") - pl.col("high").shift(1).over(g)).alias("up_move"),
            (pl.col("low").shift(1).over(g) - pl.col("low")).alias("down_move")
        ])
        .with_columns([
            pl.when((pl.col("up_move") > pl.col("down_move")) & (pl.col("up_move") > 0))
              .then(pl.col("up_move")).otherwise(0).alias("plus_dm"),
            pl.when((pl.col("down_move") > pl.col("up_move")) & (pl.col("down_move") > 0))
              .then(pl.col("down_move")).otherwise(0).alias("minus_dm")
        ])
        .with_columns([
            pl.col("tr_v3").rolling_mean(5).over(g).alias("atr5_v3"),
            pl.col("tr_v3").rolling_mean(14).over(g).alias("atr14_v3"),
            pl.col("plus_dm").rolling_sum(5).over(g).alias("plus_dm_sum5"),
            pl.col("plus_dm").rolling_sum(14).over(g).alias("plus_dm_sum14"),
            pl.col("minus_dm").rolling_sum(5).over(g).alias("minus_dm_sum5"),
            pl.col("minus_dm").rolling_sum(14).over(g).alias("minus_dm_sum14")
        ])
        .with_columns([
            (100 * _safe_div(pl.col("plus_dm_sum5"), pl.col("atr5_v3"))).alias("plus_di5"),
            (100 * _safe_div(pl.col("minus_dm_sum5"), pl.col("atr5_v3"))).alias("minus_di5"),
            (100 * _safe_div(pl.col("plus_dm_sum14"), pl.col("atr14_v3"))).alias("plus_di14_raw"),
            (100 * _safe_div(pl.col("minus_dm_sum14"), pl.col("atr14_v3"))).alias("minus_di14_raw")
        ])
        .with_columns([
            pl.col("plus_di14_raw").fill_null(pl.col("plus_di5")).alias("plus_di14"),
            pl.col("minus_di14_raw").fill_null(pl.col("minus_di5")).alias("minus_di14")
        ])
        .with_columns([
            (_safe_div((pl.col("plus_di14") - pl.col("minus_di14")).abs(), 
                      (pl.col("plus_di14") + pl.col("minus_di14"))) * 100).alias("dx")
        ])
        .with_columns([
            pl.col("dx").rolling_mean(5).over(g).alias("adx5"),
            pl.col("dx").rolling_mean(14).over(g).alias("adx14_raw")
        ])
        .with_columns([
            pl.col("adx14_raw").fill_null(pl.col("adx5")).alias("adx14")
        ])
        # V / V2 패턴 계산
        .with_columns([
            _safe_div((((pl.col("close")-pl.col("low")) - (pl.col("high")-pl.col("close"))) 
                      / (pl.col("high")-pl.col("low")+1e-9)) * pl.col("volume"), 1).alias("mfm_vol")
        ])
        .with_columns([
            pl.col("mfm_vol").rolling_sum(10).over(g).alias("mfm_vol_sum10"),
            pl.col("volume").rolling_sum(10).over(g).alias("vol_sum10_v3"),
            pl.col("mfm_vol").rolling_sum(20).over(g).alias("mfm_vol_sum20"),
            pl.col("volume").rolling_sum(20).over(g).alias("vol_sum20_v3")
        ])
        .with_columns([
            _safe_div(pl.col("mfm_vol_sum10"), pl.col("vol_sum10_v3")).alias("cmf10_v3"),
            _safe_div(pl.col("mfm_vol_sum20"), pl.col("vol_sum20_v3")).alias("cmf20_v3_raw")
        ])
        .with_columns([
            pl.col("cmf20_v3_raw").fill_null(pl.col("cmf10_v3")).alias("cmf20_v3")
        ])
        .with_columns([
            pl.col("cmf20_v3").alias("v_pattern")
        ])
        # OBV와 VWAP20 계산
        .with_columns([
            pl.when(pl.col("close") > pl.col("close").shift(1).over(g)).then(pl.col("volume"))
             .when(pl.col("close") < pl.col("close").shift(1).over(g)).then(-pl.col("volume"))
             .otherwise(0).alias("obv_delta_v3")
        ])
        .with_columns([
            pl.col("obv_delta_v3").cum_sum().over(g).alias("obv_v3")
        ])
        .with_columns([
            (_tp() * pl.col("volume")).rolling_sum(10).over(g).alias("vwap_num10_v3"),
            pl.col("volume").rolling_sum(10).over(g).alias("vwap_den10_v3"),
            (_tp() * pl.col("volume")).rolling_sum(20).over(g).alias("vwap_num_v3"),
            pl.col("volume").rolling_sum(20).over(g).alias("vwap_den_v3")
        ])
        .with_columns([
            _safe_div(pl.col("vwap_num10_v3"), pl.col("vwap_den10_v3")).alias("vwap10_v3"),
            _safe_div(pl.col("vwap_num_v3"), pl.col("vwap_den_v3")).alias("vwap20_v3_raw")
        ])
        .with_columns([
            pl.col("vwap20_v3_raw").fill_null(pl.col("vwap10_v3")).alias("vwap20_v3")
        ])
        .with_columns([
            ((pl.col("obv_v3") - pl.col("obv_v3").shift(5).over(g)) * 
             (pl.col("close") - pl.col("vwap20_v3"))).alias("v2_pattern")
        ])
        # RSI5 계산 (null 처리 강화)
        .with_columns([
            (pl.col("close") - pl.col("close").shift(1).over(g)).alias("price_change_5")
        ])
        .with_columns([
            pl.when(pl.col("price_change_5") > 0).then(pl.col("price_change_5")).otherwise(0.0).alias("gain5_raw"),
            pl.when(pl.col("price_change_5") < 0).then(-pl.col("price_change_5")).otherwise(0.0).alias("loss5_raw")
        ])
        .with_columns([
            pl.col("gain5_raw").rolling_mean(5).over(g).fill_null(pl.col("gain5_raw")).alias("gain5_v3"),
            pl.col("loss5_raw").rolling_mean(5).over(g).fill_null(pl.col("loss5_raw")).alias("loss5_v3")
        ])
        .with_columns([
            (100 - 100 / (1 + _safe_div(pl.col("gain5_v3"), pl.col("loss5_v3")))).alias("rsi5")
        ])
        # EMA와 ATR 계산
        .with_columns([
            pl.col("close").ewm_mean(alpha=2/6, adjust=False).over(g).alias("ema5"),
            pl.col("close").ewm_mean(alpha=2/15, adjust=False).over(g).alias("ema14"),
            pl.col("tr_v3").rolling_mean(5).over(g).alias("atr5")
        ])
        # Aroon 프록시 (pos_in_don20 필요)
        .with_columns([
            (pl.col("pos_in_don20") * 100).alias("aroon_up20"),
            ((1 - pl.col("pos_in_don20")) * 100).alias("aroon_dn20")
        ])
    # PSAR/KAMA 프록시
        .with_columns([
            pl.col("close").rolling_mean(5).over(g).alias("psar_proxy"),
            pl.col("close").ewm_mean(alpha=0.2, adjust=False).over(g).alias("kama10")
        ])
        # 중간 컬럼들 정리
        .drop([
            "tr_v3", "up_move", "down_move", "plus_dm", "minus_dm", 
            "atr5_v3", "atr14_v3", "plus_dm_sum5", "plus_dm_sum14", "minus_dm_sum5", "minus_dm_sum14",
            "plus_di5", "minus_di5", "plus_di14_raw", "minus_di14_raw", "adx5", "adx14_raw", "dx", 
            "mfm_vol", "mfm_vol_sum10", "vol_sum10_v3", "mfm_vol_sum20", "vol_sum20_v3", 
            "cmf10_v3", "cmf20_v3_raw", "cmf20_v3", "obv_delta_v3", "obv_v3", 
            "vwap_num10_v3", "vwap_den10_v3", "vwap10_v3", "vwap_num_v3", "vwap_den_v3", "vwap20_v3_raw", "vwap20_v3", 
            "price_change_5", "gain5_raw", "loss5_raw", "gain5_v3", "loss5_v3"
        ])
        # NaN/Null 처리 완화 (forward fill + 0 대체)
        .with_columns([
            pl.col(pl.FLOAT_DTYPES).fill_nan(None).forward_fill().over(g).fill_null(0),
            pl.col(pl.INTEGER_DTYPES).fill_null(0)
        ])
    )

def add_m002_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    """
    Feature block tailored for the M002 architecture.
    Builds upon v3 features and injects event heuristics, curvature/slope metrics, and context signals.
    """
    g = "ticker"
    curvature_window = 11

    lf = add_v3_features(add_v2_features(add_core_features(lf)))

    # Core derivatives and spreads
    lf = lf.with_columns([
        (pl.col("close") / pl.col("close").shift(1).over(g) - 1).alias("ret1"),
        (pl.col("volume") / pl.col("volume").shift(1).over(g) - 1).alias("vol_roc1"),
        (pl.col("rsi14") - pl.col("rsi14").shift(1).over(g)).alias("delta_rsi"),
        (pl.col("rsi14") - pl.col("rsi14").shift(3).over(g)).alias("delta_rsi_3"),
        (pl.col("macd_hist") - pl.col("macd_hist").shift(1).over(g)).alias("delta_macd"),
        (pl.col("ema5") - pl.col("ema20")).alias("ema_spread"),
        (pl.col("atr14") - pl.col("atr14").shift(1).over(g)).alias("delta_atr"),
    ])

    # Local volatility ratio
    lf = lf.with_columns([
        pl.col("ret1").rolling_std(5).over(g).alias("ret_std5"),
        pl.col("ret1").rolling_std(20).over(g).alias("ret_std20"),
    ])
    lf = lf.with_columns([
        _safe_div(pl.col("ret_std5"), pl.col("ret_std20")).alias("local_vol_index"),
    ])

    # Bollinger adaptions
    lf = lf.with_columns([
        pl.col("close").rolling_mean(20).over(g).alias("bb_mid"),
        pl.col("close").rolling_std(20).over(g).alias("bb_std"),
        (pl.col("close").rolling_mean(20).over(g) + 2 * pl.col("close").rolling_std(20).over(g)).alias("bb_upper"),
        (pl.col("close").rolling_mean(20).over(g) - 2 * pl.col("close").rolling_std(20).over(g)).alias("bb_lower"),
        ((pl.col("close").rolling_mean(20).over(g) + 2 * pl.col("close").rolling_std(20).over(g)) -
         (pl.col("close").rolling_mean(20).over(g) - 2 * pl.col("close").rolling_std(20).over(g))).alias("bb_band_width"),
    ])
    lf = lf.with_columns([
        _safe_div(pl.col("close") - pl.col("bb_mid"), pl.col("bb_band_width")).clip(0.0, 1.0).alias("pos_in_band_rel"),
        _safe_div(pl.col("bb_band_width"), pl.col("bb_mid").abs()).alias("bb_width"),
    ])

    # Spread relatives and slopes
    lf = lf.with_columns([
        _safe_div(pl.col("ema_spread"), pl.col("ema20")).alias("ema_spread_rel"),
        (pl.col("ema_spread") - pl.col("ema_spread").shift(1).over(g)).alias("delta_ema_spread"),
    ])
    lf = lf.with_columns([
        (pl.col("ema_spread_rel") - pl.col("ema_spread_rel").shift(1).over(g)).alias("ema_spread_rel_slope"),
    ])

    # ATR relatives
    lf = lf.with_columns([
        _safe_div(pl.col("atr14"), pl.col("atr14").rolling_mean(20).over(g)).alias("atr_rel"),
        _safe_div(pl.col("delta_atr"), pl.col("atr14").rolling_std(10).over(g)).alias("delta_atr_rel"),
    ])
    lf = lf.with_columns([
        (pl.col("atr_rel") - pl.col("atr_rel").shift(1).over(g)).alias("atr_slope"),
    ])

    # Momentum smoothers
    lf = lf.with_columns([
        pl.col("vol_z20").alias("volume_z"),
        (pl.col("ret1") - pl.col("ret1").shift(1).over(g)).alias("price_accel"),
        pl.col("ret1").alias("price_slope"),
        (pl.col("macd_signal") - pl.col("macd_signal").shift(1).over(g)).alias("macd_signal_slope"),
    ])

    # Normalised price for curvature measures
    lf = lf.with_columns([
        _safe_div(
            pl.col("close") - pl.col("close").rolling_mean(20).over(g),
            pl.col("close").rolling_std(20).over(g)
        ).alias("close_norm"),
    ])

    lf = lf.with_columns([
        pl.col("close_norm")
        .rolling_map(lambda s: _poly_second_derivative(s.to_numpy()), window_size=curvature_window, min_periods=5)
        .over(g)
        .alias("curv_2")
    ])

    lf = lf.with_columns([
        pl.col("rsi14").ewm_mean(alpha=2 / 6, adjust=False).over(g).alias("rsi_ewm"),
        pl.col("macd_hist").ewm_mean(alpha=2 / 5, adjust=False).over(g).alias("macd_hist_ewm"),
        pl.col("atr_rel").ewm_mean(alpha=2 / 4, adjust=False).over(g).alias("atr_rel_ewm"),
        pl.col("volume_z").ewm_mean(alpha=2 / 5, adjust=False).over(g).alias("volume_z_ewm"),
        pl.col("ema_spread_rel").ewm_mean(alpha=2 / 5, adjust=False).over(g).alias("ema_spread_rel_ewm"),
    ])

    lf = lf.with_columns([
        pl.col("rsi_ewm").rolling_median(3).over(g).fill_null(pl.col("rsi_ewm")).alias("rsi_smooth"),
        pl.col("macd_hist_ewm").rolling_mean(3).over(g).alias("macd_smooth"),
        pl.col("atr_rel_ewm").rolling_mean(3).over(g).alias("atr_smooth"),
        pl.col("volume_z_ewm").rolling_mean(3).over(g).alias("volume_z_smooth"),
        pl.col("ema_spread_rel_ewm").rolling_mean(3).over(g).alias("ema_spread_smooth"),
    ])

    # Event heuristics
    lf = lf.with_columns([
        pl.when(pl.col("local_vol_index") > 1.5).then(1).otherwise(0).cast(pl.Int8).alias("event_local_vol_spike"),
        pl.when(
            (pl.col("pos_in_band_rel") < 0.35)
            & (pl.col("delta_rsi_3") > 0)
            & (pl.col("delta_macd") > 0)
            & (pl.col("volume_z") > 0)
        ).then(1).otherwise(0).cast(pl.Int8).alias("event_rebound_candidate"),
        pl.when(
            (pl.col("volume_z") > 0.8)
            & (pl.col("vol_roc1") > 0)
            & (pl.col("delta_rsi") > 0)
        ).then(1).otherwise(0).cast(pl.Int8).alias("event_volume_regain"),
        pl.when(
            (pl.col("pos_in_band_rel") > 0.85)
            & (pl.col("delta_ema_spread") < 0)
            & (pl.col("delta_atr") > 0)
            & (pl.col("volume_z") < 0.5)
        ).then(1).otherwise(0).cast(pl.Int8).alias("event_exhaustion_candidate"),
        pl.when(
            (pl.col("local_vol_index") > 1.2)
            & (pl.col("macd_hist") < 0)
            & (pl.col("delta_rsi") < 0)
        ).then(1).otherwise(0).cast(pl.Int8).alias("event_breakdown_risk"),
    ])

    lf = lf.with_columns([
        (pl.col("event_volume_regain") & pl.col("event_local_vol_spike")).cast(pl.Int8).alias("I_vr_and_vs"),
        (pl.col("event_breakdown_risk") & (pl.col("curv_2") < 0) & (pl.col("ema_spread_rel") < 0) & (pl.col("delta_rsi") < 0))
        .cast(pl.Int8)
        .alias("I_bd_early"),
        (pl.col("event_breakdown_risk") & (pl.col("curv_2") > 0) & (pl.col("rsi14") < 40) & (pl.col("macd_signal_slope") > 0))
        .cast(pl.Int8)
        .alias("I_bd_late"),
    ])

    # Context metrics
    lf = lf.with_columns([
        pl.col("event_volume_regain").rolling_sum(20).over(g).alias("event_volume_regain_freq20"),
        pl.col("event_breakdown_risk").rolling_sum(20).over(g).alias("event_breakdown_freq20"),
        pl.col("event_local_vol_spike").rolling_sum(20).over(g).alias("event_local_vol_freq20"),
    ])

    lf = lf.with_columns([
        pl.col("close").rolling_max(60).over(g).alias("recent_high_60"),
        pl.col("close").rolling_min(60).over(g).alias("recent_low_60"),
    ])

    lf = lf.with_columns([
        _safe_div(pl.col("close") - pl.col("recent_low_60"), pl.col("recent_low_60")).alias("recovery_from_low_60"),
        _safe_div(pl.col("close") - pl.col("recent_high_60"), pl.col("recent_high_60")).alias("distance_from_high_60"),
    ])

    lf = lf.with_columns([
        pl.when(pl.col("event_volume_regain") > 0).then(pl.col("date")).otherwise(None).alias("_last_vr_date"),
        pl.when(pl.col("event_breakdown_risk") > 0).then(pl.col("date")).otherwise(None).alias("_last_bd_date"),
    ])

    lf = lf.with_columns([
        pl.col("_last_vr_date").forward_fill().over(g).alias("_last_vr_ff"),
        pl.col("_last_bd_date").forward_fill().over(g).alias("_last_bd_ff"),
    ])

    lf = lf.with_columns([
        (pl.col("date") - pl.col("_last_vr_ff")).dt.total_days().cast(pl.Int32).fill_null(1_000).alias("days_since_volume_regain"),
        (pl.col("date") - pl.col("_last_bd_ff")).dt.total_days().cast(pl.Int32).fill_null(1_000).alias("days_since_breakdown"),
    ])

    # Clean up temporary helpers
    lf = lf.drop([
        "ret_std5", "ret_std20", "_last_vr_date", "_last_bd_date", "_last_vr_ff", "_last_bd_ff", "rsi_ewm",
        "macd_hist_ewm", "atr_rel_ewm", "volume_z_ewm", "ema_spread_rel_ewm"
    ])

    return lf

def add_feature_set(lf: pl.LazyFrame, feature_set: str="v1") -> pl.LazyFrame:
    if feature_set == "v1":
        return add_core_features(lf)
    if feature_set == "v2":
        return add_v2_features(add_core_features(lf))
    if feature_set == "v3":
        return add_v3_features(add_v2_features(add_core_features(lf)))
    if feature_set == "m002":
        return add_m002_features(lf)
    raise ValueError(f"unknown feature_set: {feature_set}")
