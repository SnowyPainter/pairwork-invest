# features/feature_sets.py
import polars as pl

def add_core_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    g = ["ticker"]
    close = pl.col("close")
    high  = pl.col("high")
    low   = pl.col("low")
    vol   = pl.col("volume").fill_null(0.0)

    # 날짜 정렬된 시퀀스로 만든 뒤 롤링/시프트
    close_s = close.sort_by("date")
    high_s  = high.sort_by("date")
    low_s   = low.sort_by("date")
    vol_s   = vol.sort_by("date")

    tr_raw = pl.max_horizontal(
        (high_s - low_s),
        (high_s - close_s.shift(1)).abs(),
        (low_s  - close_s.shift(1)).abs()
    )

    typical = (high_s + low_s + close_s) / 3

    return (
        lf
        .with_columns([
            # logret_1: diff(log close)
            (close_s.log().diff()).over(g).alias("logret_1"),

            # ret_5
            ((close_s / close_s.shift(5)) - 1).over(g).alias("ret_5"),

            # ATR14
            tr_raw.over(g).rolling_mean(14).alias("atr14"),

            # VWAP20
            (
                ((typical * vol_s).over(g).rolling_sum(20)) /
                (vol_s.over(g).rolling_sum(20) + 1e-9)
            ).alias("vwap20"),
        ])
        .with_columns([
            ( (close_s - close_s.shift(1)).clip(lower_bound=0.0) ).over(g).rolling_mean(14).alias("gain14"),
            ( (close_s.shift(1) - close_s).clip(lower_bound=0.0) ).over(g).rolling_mean(14).alias("loss14"),
        ])
        .with_columns([
            (100 - 100/(1 + (pl.col("gain14")/(pl.col("loss14")+1e-9)))).alias("rsi14")
        ])
        .drop(["gain14","loss14"])
    )


# ==== 유틸 ====
def _tp() -> pl.Expr:
    return (pl.col("high") + pl.col("low") + pl.col("close")) / 3

def _true_range() -> pl.Expr:
    prev_close = pl.col("close").shift(1)
    return pl.max_horizontal(
        pl.col("high") - pl.col("low"),
        (pl.col("high") - prev_close).abs(),
        (pl.col("low")  - prev_close).abs(),
    )

def _safe_div(a: pl.Expr, b: pl.Expr, eps: float=1e-9) -> pl.Expr:
    return a / (b + eps)

# ==== v2 확장 ====
def add_v2_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    g = ["ticker"]

    W_S = [5, 10, 20, 50, 100, 200]
    W_VWAP = [10, 20, 50]

    # 1) 트렌드/이평
    sma_cols = []
    ema_cols = []
    for w in W_S:
        sma_cols.append((pl.col("close").over(g).rolling_mean(w)).alias(f"sma{w}"))
        ema_cols.append((pl.col("close").over(g).ewm_mean(alpha=2/(w+1), adjust=False)).alias(f"ema{w}"))

    # 2) 모멘텀 & 오실레이터
    # ROC, RSI(6), Stoch, Williams %R, CCI, MACD
    roc_cols = [
        ((pl.col("close")/pl.col("close").shift(w)-1).over(g)).alias(f"roc{w}") for w in [5,10,20]
    ]
    # RSI6
    rsi6 = (
        pl.when((pl.col("close")-pl.col("close").shift(1))>0)
          .then(pl.col("close")-pl.col("close").shift(1)).otherwise(0.0)
          .over(g).rolling_mean(6)
    ).alias("gain6_tmp")
    loss6 = (
        pl.when((pl.col("close")-pl.col("close").shift(1))<0)
          .then((pl.col("close")-pl.col("close").shift(1)).abs()).otherwise(0.0)
          .over(g).rolling_mean(6)
    ).alias("loss6_tmp")

    # Stochastic
    def stoch_kd(n: int=14, d: int=3):
        lowest_n  = pl.col("low").over(g).rolling_min(n)
        highest_n = pl.col("high").over(g).rolling_max(n)
        k = _safe_div(pl.col("close") - lowest_n, highest_n - lowest_n) * 100
        dline = k.over(g).rolling_mean(d)
        return k.alias(f"stochk{n}"), dline.alias(f"stochd{n}")

    k14, d14 = stoch_kd(14, 3)

    # Williams %R
    willr14 = ( - _safe_div( (highest_n := pl.col("high").over(g).rolling_max(14)) - pl.col("close"),
                              highest_n - pl.col("low").over(g).rolling_min(14) ) * 100 ).alias("willr14")

    # CCI(20)
    tp_col = _tp().alias("tp")
    sma_tp20 = tp_col.over(g).rolling_mean(20)
    md20 = (tp_col - sma_tp20).abs().over(g).rolling_mean(20)
    cci20 = _safe_div(tp_col - sma_tp20, 0.015 * md20).alias("cci20")

    # MACD(12,26,9) (EMA 기반)
    ema12 = pl.col("close").over(g).ewm_mean(alpha=2/(12+1), adjust=False)
    ema26 = pl.col("close").over(g).ewm_mean(alpha=2/(26+1), adjust=False)
    macd = (ema12 - ema26).alias("macd")
    signal = macd.over(g).ewm_mean(alpha=2/(9+1), adjust=False).alias("macd_signal")
    macd_hist = (pl.col("macd") - pl.col("macd_signal")).alias("macd_hist")

    # 3) 변동성
    vol_10 = pl.col("logret_1").over(g).rolling_std(10).alias("vol10")
    vol_20 = pl.col("logret_1").over(g).rolling_std(20).alias("vol20")
    # Parkinson (고저 변동성)
    parkinson = (
        ((pl.col("high")/pl.col("low")).log() ** 2) / (4 * pl.lit(pl.Series([2.0]).log()[0]))  # 4*ln(2)
    ).over(g).rolling_mean(20).alias("parkinson20")
    # Garman–Klass (고저 + 시가/종가)
    gk = (
        0.5 * (pl.col("high")/pl.col("low")).log()**2
        - (2*pl.lit(pl.Series([2.0]).log()[0]) - 1) * (pl.col("close")/pl.col("open")).log()**2
    ).over(g).rolling_mean(20).alias("gk20")
    # True Range 파생
    tr = _true_range().alias("tr")
    tr14 = pl.col("tr").over(g).rolling_mean(14).alias("tr14")

    # 4) 거래량 & 자금흐름: OBV, MFI, CMF, VWAPs, Volume zscore
    # OBV
    price_up = (pl.col("close") > pl.col("close").shift(1))
    price_dn = (pl.col("close") < pl.col("close").shift(1))
    obv_delta = (pl.when(price_up).then(pl.col("volume"))
                   .when(price_dn).then(-pl.col("volume"))
                   .otherwise(0)).alias("obv_delta")
    obv = pl.col("obv_delta").over(g).cum_sum().alias("obv")

    # MFI(14)
    raw_flow = (pl.col("tp") * pl.col("volume")).alias("raw_flow")
    pos_flow = (pl.when(pl.col("tp") > pl.col("tp").shift(1)).then(pl.col("raw_flow")).otherwise(0)).alias("pos_flow")
    neg_flow = (pl.when(pl.col("tp") < pl.col("tp").shift(1)).then(pl.col("raw_flow")).otherwise(0)).alias("neg_flow")
    mfr14 = _safe_div(pl.col("pos_flow").over(g).rolling_sum(14),
                      pl.col("neg_flow").over(g).rolling_sum(14)).alias("mfr14")
    mfi14 = (100 - 100/(1 + pl.col("mfr14"))).alias("mfi14")

    # CMF(20)
    mfm = _safe_div( (pl.col("close")-pl.col("low")) - (pl.col("high")-pl.col("close")),
                     (pl.col("high")-pl.col("low")) ).alias("mfm")
    cmf20 = _safe_div(
        (pl.col("mfm")*pl.col("volume")).over(g).rolling_sum(20),
        pl.col("volume").over(g).rolling_sum(20)
    ).alias("cmf20")

    # Multi-window VWAP
    vwap_exprs = []
    for w in W_VWAP:
        vwap_exprs.append( _safe_div(
            ((_tp()*pl.col("volume")).over(g).rolling_sum(w)),
            (pl.col("volume").over(g).rolling_sum(w))
        ).alias(f"vwap{w}") )

    # Volume zscore/roc
    vol_ma20 = pl.col("volume").over(g).rolling_mean(20).alias("vol_ma20")
    vol_std20 = pl.col("volume").over(g).rolling_std(20).alias("vol_std20")
    vol_z20 = _safe_div(pl.col("volume") - pl.col("vol_ma20"), pl.col("vol_std20")).alias("vol_z20")
    vol_roc5 = _safe_div(pl.col("volume"), pl.col("volume").shift(5)) - 1
    vol_roc5 = vol_roc5.over(g).alias("vol_roc5")

    # 5) 가격 구조/브레이크아웃: Donchian, 위치, 갭/바디
    don20_hi = pl.col("high").over(g).rolling_max(20).alias("don20_hi")
    don20_lo = pl.col("low").over(g).rolling_min(20).alias("don20_lo")
    don55_hi = pl.col("high").over(g).rolling_max(55).alias("don55_hi")
    don55_lo = pl.col("low").over(g).rolling_min(55).alias("don55_lo")
    pos_in_don20 = _safe_div(pl.col("close")-pl.col("don20_lo"),
                             pl.col("don20_hi")-pl.col("don20_lo")).alias("pos_in_don20")
    day_range = (pl.col("high")-pl.col("low")).alias("day_range")
    body = (pl.col("close")-pl.col("open")).alias("body")
    gap = (pl.col("open")-pl.col("close").shift(1)).alias("gap")
    rel_range = _safe_div(pl.col("day_range"), pl.col("close")).alias("rel_range")

    # 6) 상관/패턴 힌트: rolling corr (close, volume), (OBV, VWAP20), 곡률(2차차분)
    # Polars rolling_corr가 없다면: 표준화-곱의 평균으로 근사
    def rolling_corr(x: pl.Expr, y: pl.Expr, w: int, name: str) -> pl.Expr:
        mx = x.over(g).rolling_mean(w)
        my = y.over(g).rolling_mean(w)
        sx = (x - mx).over(g).rolling_std(w)
        sy = (y - my).over(g).rolling_std(w)
        z = _safe_div( ( (x - mx) * (y - my) ).over(g).rolling_mean(w),
                       (sx * sy) )
        return z.alias(name)

    corr_close_vol20 = rolling_corr(pl.col("close"), pl.col("volume"), 20, "corr_close_vol20")
    corr_obv_vwap20  = rolling_corr(pl.col("obv"), pl.col("vwap20"), 20, "corr_obv_vwap20")

    # 곡률(2차 차분) 근사: V-curve 탐지의 힌트 (close 기반)
    d1 = (pl.col("close") - pl.col("close").shift(1)).alias("d1")
    d2 = (pl.col("d1") - pl.col("d1").shift(1)).alias("d2")
    curv5 = pl.col("d2").over(g).rolling_mean(5).alias("curv5")  # 양수↑ 음수↓ 곡률 힌트

    return (
        lf
        # 1차 모듈들
        .with_columns(sma_cols + ema_cols + roc_cols)
        .with_columns([rsi6, loss6])  # 임시
        .with_columns([
            (100 - 100/(1 + _safe_div(pl.col("gain6_tmp"), pl.col("loss6_tmp")))).alias("rsi6")
        ]).drop(["gain6_tmp","loss6_tmp"])
        .with_columns([tp_col])
        .with_columns([k14, d14, willr14, cci20, macd, signal])
        .with_columns([macd_hist])
        # 변동성: tr과 tr14를 분리
        .with_columns([vol_10, vol_20, parkinson, gk, tr])
        .with_columns([tr14])
        # MFI: tp, raw_flow -> pos/neg -> mfr -> mfi 순차 적용
        .with_columns([raw_flow])
        .with_columns([pos_flow, neg_flow])
        .with_columns([mfr14])
        .with_columns([mfi14])
        # CMF: mfm 후 cmf20
        .with_columns([mfm])
        .with_columns([cmf20])
        # VWAPs
        .with_columns(vwap_exprs)
        # OBV: delta 후 누적합
        .with_columns([obv_delta])
        .with_columns([obv])
        # Volume zscore: 평균/표준편차 후 zscore
        .with_columns([vol_ma20, vol_std20])
        .with_columns([vol_z20, vol_roc5])
        # Donchian: 경계 후 포지션
        .with_columns([don20_hi, don20_lo, don55_hi, don55_lo])
        .with_columns([day_range])
        .with_columns([body, gap, rel_range])
        .with_columns([pos_in_don20])
        # 상관
        .with_columns([corr_close_vol20, corr_obv_vwap20])
        # 곡률: d1 -> d2 -> curv5
        .with_columns([d1])
        .with_columns([d2])
        .with_columns([curv5])
        # 파생: 이동평균乖離, 크로스 근접도
        .with_columns([
            _safe_div(pl.col("close")-pl.col("sma20"), pl.col("sma20")).alias("dev_sma20"),
            _safe_div(pl.col("close")-pl.col("ema20"), pl.col("ema20")).alias("dev_ema20"),
            _safe_div(pl.col("sma20")-pl.col("sma50"), pl.col("sma50")).alias("ma_squeeze_20_50"),
            _safe_div(ema12 - ema26, pl.col("close")).alias("ema12_26_rel")
        ])
        .with_columns([
            pl.col(pl.FLOAT_DTYPES).fill_nan(None),
            pl.col(pl.INTEGER_DTYPES).fill_null(0)
        ])
    )

def add_v3_features(lf: pl.LazyFrame) -> pl.LazyFrame:
    g = ["ticker"]

    # ADX, +DI, -DI (동일)
    tr = _true_range()
    up_move = pl.col("high") - pl.col("high").shift(1)
    down_move = pl.col("low").shift(1) - pl.col("low")
    plus_dm = pl.when((up_move > down_move) & (up_move > 0)).then(up_move).otherwise(0)
    minus_dm = pl.when((down_move > up_move) & (down_move > 0)).then(down_move).otherwise(0)
    atr14 = tr.over(g).rolling_mean(14)
    plus_di14  = (100 * _safe_div(plus_dm.over(g).rolling_sum(14), atr14)).alias("plus_di14")
    minus_di14 = (100 * _safe_div(minus_dm.over(g).rolling_sum(14), atr14)).alias("minus_di14")
    dx = _safe_div((plus_di14 - minus_di14).abs(), (plus_di14 + minus_di14)) * 100
    adx14 = dx.over(g).rolling_mean(14).alias("adx14")

    # V / V2 (동일)
    cmf20 = (
        _safe_div(((( (pl.col("close")-pl.col("low")) - (pl.col("high")-pl.col("close")) )
                   / (pl.col("high")-pl.col("low")+1e-9)) * pl.col("volume")
                  ).over(g).rolling_sum(20),
                 pl.col("volume").over(g).rolling_sum(20))
    ).alias("cmf20")
    v_pattern = pl.col("cmf20").alias("v_pattern")

    obv = (pl.when(pl.col("close") > pl.col("close").shift(1)).then(pl.col("volume"))
             .when(pl.col("close") < pl.col("close").shift(1)).then(-pl.col("volume"))
             .otherwise(0)).over(g).cum_sum()
    vwap20 = _safe_div((_tp()*pl.col("volume")).over(g).rolling_sum(20),
                       pl.col("volume").over(g).rolling_sum(20))
    v2_pattern = ((obv - obv.shift(5)) * (pl.col("close") - vwap20)).alias("v2_pattern")

    # RSI5 (수정된 부분)
    gain5 = (
        pl.when(pl.col("close") - pl.col("close").shift(1) > 0)
          .then(pl.col("close") - pl.col("close").shift(1)).otherwise(0.0)
    ).over(g).rolling_mean(5)
    loss5 = (
        pl.when(pl.col("close") - pl.col("close").shift(1) < 0)
          .then((pl.col("close") - pl.col("close").shift(1)).abs()).otherwise(0.0)
    ).over(g).rolling_mean(5)
    rs = _safe_div(gain5, loss5)
    rsi5 = (100 - 100 / (1 + rs)).alias("rsi5")

    ema5  = pl.col("close").over(g).ewm_mean(alpha=2/(5+1),  adjust=False).alias("ema5")
    ema14 = pl.col("close").over(g).ewm_mean(alpha=2/(14+1), adjust=False).alias("ema14")
    atr5  = tr.over(g).rolling_mean(5).alias("atr5")

    # Aroon 프록시 (v2의 pos_in_don20 이용)
    aroon_up20 = (pl.col("pos_in_don20") * 100).alias("aroon_up20")
    aroon_dn20 = ((1 - pl.col("pos_in_don20")) * 100).alias("aroon_dn20")

    # PSAR/KAMA 프록시
    psar  = pl.col("close").over(g).rolling_mean(5).alias("psar_proxy")
    kama10 = pl.col("close").over(g).ewm_mean(alpha=0.2, adjust=False).alias("kama10")

    return (lf.with_columns([
        plus_di14, minus_di14, adx14,
        v_pattern, v2_pattern,
        rsi5, ema5, ema14, atr5,
        aroon_up20, aroon_dn20, psar, kama10
    ]).with_columns([
            pl.col(pl.FLOAT_DTYPES).fill_nan(None),
            pl.col(pl.INTEGER_DTYPES).fill_null(0)
        ]))

def add_feature_set(lf: pl.LazyFrame, feature_set: str="v1") -> pl.LazyFrame:
    if feature_set == "v1":
        return add_core_features(lf)
    if feature_set == "v2":
        return add_v2_features(add_core_features(lf))
    if feature_set == "v3":
        return add_v3_features(add_v2_features(add_core_features(lf)))
    raise ValueError(f"unknown feature_set: {feature_set}")