# features/feature_sets.py
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
            # ATR14: True Range의 14일 이동평균
            pl.col("tr_raw").rolling_mean(14).over(g).alias("atr14"),

            # VWAP20: Volume Weighted Average Price
            (pl.col("typical") * vol).rolling_sum(20).over(g).alias("vwap_num"),
            vol.rolling_sum(20).over(g).alias("vwap_den"),
        ])
        .with_columns([
            # VWAP20 최종 계산
            (pl.col("vwap_num") / (pl.col("vwap_den") + 1e-9)).alias("vwap20"),
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
            # 14일 이동평균 적용
            pl.col("gain_raw").rolling_mean(14).over(g).alias("gain14"),
            pl.col("loss_raw").rolling_mean(14).over(g).alias("loss14"),
        ])
        .with_columns([
            # RSI14 최종 계산
            (100 - 100/(1 + (pl.col("gain14")/(pl.col("loss14")+1e-9)))).alias("rsi14")
        ])
        .drop(["tr_raw", "typical", "vwap_num", "vwap_den", "price_change", "gain_raw", "loss_raw", "gain14", "loss14"])
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

# ==== v2 확장 ====
def add_v2_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    g = "ticker"

    W_S = [5, 10, 20, 50, 100, 200]
    W_VWAP = [10, 20, 50]

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
        # 3) RSI6 계산을 위한 중간 단계
        .with_columns([
            (pl.col("close") - pl.col("close").shift(1).over(g)).alias("price_change_6")
        ])
        .with_columns([
            pl.when(pl.col("price_change_6") > 0).then(pl.col("price_change_6")).otherwise(0.0).alias("gain6_raw"),
            pl.when(pl.col("price_change_6") < 0).then(-pl.col("price_change_6")).otherwise(0.0).alias("loss6_raw")
        ])
        .with_columns([
            pl.col("gain6_raw").rolling_mean(6).over(g).alias("gain6"),
            pl.col("loss6_raw").rolling_mean(6).over(g).alias("loss6")
        ])
        .with_columns([
            (100 - 100/(1 + _safe_div(pl.col("gain6"), pl.col("loss6")))).alias("rsi6")
        ])
        # 4) Stochastic 지표들
        .with_columns([
            pl.col("low").rolling_min(14).over(g).alias("lowest14"),
            pl.col("high").rolling_max(14).over(g).alias("highest14")
        ])
        .with_columns([
            (_safe_div(pl.col("close") - pl.col("lowest14"), pl.col("highest14") - pl.col("lowest14")) * 100).alias("stochk14")
        ])
        .with_columns([
            pl.col("stochk14").rolling_mean(3).over(g).alias("stochd14")
        ])
        # 5) Williams %R
        .with_columns([
            (-_safe_div(pl.col("highest14") - pl.col("close"), pl.col("highest14") - pl.col("lowest14")) * 100).alias("willr14")
        ])
        # 6) CCI(20) 계산
        .with_columns([
            _tp().alias("tp")
        ])
        .with_columns([
            pl.col("tp").rolling_mean(20).over(g).alias("sma_tp20")
        ])
        .with_columns([
            (pl.col("tp") - pl.col("sma_tp20")).abs().alias("dev_tp")
        ])
        .with_columns([
            pl.col("dev_tp").rolling_mean(20).over(g).alias("md20")
        ])
        .with_columns([
            _safe_div(pl.col("tp") - pl.col("sma_tp20"), 0.015 * pl.col("md20")).alias("cci20")
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
            "lowest14", "highest14", "tp", "sma_tp20", "dev_tp", "md20",
            "ema12", "ema26", "parkinson_raw", "gk_raw", "tr", "obv_delta",
            "raw_flow", "pos_flow", "neg_flow", "pos_flow_sum14", "neg_flow_sum14", "mfr14",
            "mfm", "cmf_num", "cmf_den", "vol_ma20", "vol_std20", "day_range", "body", "gap",
            "close_mean20", "vol_mean20_corr", "close_std20", "vol_std20_corr",
            "obv_mean20", "vwap20_mean20", "obv_std20", "vwap20_std20",
            "close_vol_cov20", "obv_vwap_cov20", "d1", "d2"
        ])
        # 22) NaN/Null 처리
        .with_columns([
            pl.col(pl.FLOAT_DTYPES).fill_nan(None),
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
            pl.col("tr_v3").rolling_mean(14).over(g).alias("atr14_v3"),
            pl.col("plus_dm").rolling_sum(14).over(g).alias("plus_dm_sum14"),
            pl.col("minus_dm").rolling_sum(14).over(g).alias("minus_dm_sum14")
        ])
        .with_columns([
            (100 * _safe_div(pl.col("plus_dm_sum14"), pl.col("atr14_v3"))).alias("plus_di14"),
            (100 * _safe_div(pl.col("minus_dm_sum14"), pl.col("atr14_v3"))).alias("minus_di14")
        ])
        .with_columns([
            (_safe_div((pl.col("plus_di14") - pl.col("minus_di14")).abs(), 
                      (pl.col("plus_di14") + pl.col("minus_di14"))) * 100).alias("dx")
        ])
        .with_columns([
            pl.col("dx").rolling_mean(14).over(g).alias("adx14")
        ])
        # V / V2 패턴 계산
        .with_columns([
            _safe_div((((pl.col("close")-pl.col("low")) - (pl.col("high")-pl.col("close"))) 
                      / (pl.col("high")-pl.col("low")+1e-9)) * pl.col("volume"), 1).alias("mfm_vol")
        ])
        .with_columns([
            pl.col("mfm_vol").rolling_sum(20).over(g).alias("mfm_vol_sum20"),
            pl.col("volume").rolling_sum(20).over(g).alias("vol_sum20_v3")
        ])
        .with_columns([
            _safe_div(pl.col("mfm_vol_sum20"), pl.col("vol_sum20_v3")).alias("cmf20_v3")
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
            (_tp() * pl.col("volume")).rolling_sum(20).over(g).alias("vwap_num_v3"),
            pl.col("volume").rolling_sum(20).over(g).alias("vwap_den_v3")
        ])
        .with_columns([
            _safe_div(pl.col("vwap_num_v3"), pl.col("vwap_den_v3")).alias("vwap20_v3")
        ])
        .with_columns([
            ((pl.col("obv_v3") - pl.col("obv_v3").shift(5).over(g)) * 
             (pl.col("close") - pl.col("vwap20_v3"))).alias("v2_pattern")
        ])
        # RSI5 계산
        .with_columns([
            (pl.col("close") - pl.col("close").shift(1).over(g)).alias("price_change_5")
        ])
        .with_columns([
            pl.when(pl.col("price_change_5") > 0).then(pl.col("price_change_5")).otherwise(0.0).alias("gain5_raw"),
            pl.when(pl.col("price_change_5") < 0).then(-pl.col("price_change_5")).otherwise(0.0).alias("loss5_raw")
        ])
        .with_columns([
            pl.col("gain5_raw").rolling_mean(5).over(g).alias("gain5"),
            pl.col("loss5_raw").rolling_mean(5).over(g).alias("loss5")
        ])
        .with_columns([
            (100 - 100 / (1 + _safe_div(pl.col("gain5"), pl.col("loss5")))).alias("rsi5")
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
            "tr_v3", "up_move", "down_move", "plus_dm", "minus_dm", "atr14_v3",
            "plus_dm_sum14", "minus_dm_sum14", "dx", "mfm_vol", "mfm_vol_sum20", 
            "vol_sum20_v3", "cmf20_v3", "obv_delta_v3", "obv_v3", "vwap_num_v3", 
            "vwap_den_v3", "vwap20_v3", "price_change_5", "gain5_raw", "loss5_raw", 
            "gain5", "loss5"
        ])
        # NaN/Null 처리
        .with_columns([
            pl.col(pl.FLOAT_DTYPES).fill_nan(None),
            pl.col(pl.INTEGER_DTYPES).fill_null(0)
        ])
    )

def add_feature_set(lf: pl.LazyFrame, feature_set: str="v1") -> pl.LazyFrame:
    if feature_set == "v1":
        return add_core_features(lf)
    if feature_set == "v2":
        return add_v2_features(add_core_features(lf))
    if feature_set == "v3":
        return add_v3_features(add_v2_features(add_core_features(lf)))
    raise ValueError(f"unknown feature_set: {feature_set}")