# 모델 설계

![ex](./example.png)

# 0) 목표/가정

* **호라이즌:** 5거래일(±2일 변형 실험)
* **타깃:** (a) `P(5D Rebound ≥ +x%)`, (b) 기대수익 `E[ret_5D]`, (c) 최대낙폭 `E[dd_5D]`
* **의사결정:** 확률·기대수익·리스크 3요소를 결합한 스코어로 Long/Flat/Short

---

# 1) 라벨링(정답) 정의

* `ret_5D = (Close[t+5] / Close[t] - 1) * 100`
* `dd_5D = min_{τ∈[t, t+5]} (Close[τ]/Close[t]-1)*100`
* **Rebound 라벨(분류):**
  `y_bin = 1 if ret_5D ≥ +1.0% AND dd_5D ≥ -3.0% else 0`
  (임계값은 이벤트별 최적화: Volume Regain은 +1.5%, Local Vol Spike는 +0.5% 등)
* **회귀 라벨:** `y_ret = ret_5D`, `y_dd = dd_5D`

> 이유: “U형 하단 중앙”에서 **올라갈 확률**과 **올라간다면 얼마나** + **중간에 얼마나 까먹는지**가 핵심.

---

# 2) 특징(Features)

## 2.1 이벤트 원천 신호

* One-hot: `I_volume_regain`, `I_local_vol_spike`, `I_breakdown_risk`, `I_rebound_candidate`, `I_exhaustion_candidate`
* 이벤트 조합: `I_vr_and_vs` (VR+VS 동시), `I_bd_late`(아래 2.3 참고)

## 2.2 곡률·형태(핵심)

* **국소 곡률(2차미분):** Savitzky–Golay로 `curv_2 = d²(close_norm)/dt²`

  * `curv_2 > 0`이면 U형(오목), `< 0`이면 ∩형(볼록)
  * “U형 하단 중앙” 판별: `curv_2` 양수 & `curv_2` 피크 근처, `slope ≈ 0`
* **대역 내 위치:** `pos_in_band_rel` (이미 보유)
* **EMA 스프레드 기울기:** `d(ema_spread_rel)/dt`
* **Price slope/accel:** 1차·2차 차분

## 2.3 ‘초기/후기 Breakdown’ 세분화 태그

* `I_bd_early`: BD 발생 & `curv_2 < 0` & `ema_spread_rel<0` & `RSI_slope<0`
* `I_bd_late`: BD 발생 & `curv_2 > 0` & `RSI<40` & `MACD_signal ↑`
  → 우리가 말한 **“U형 하단 중앙”**은 거의 `I_bd_late = 1`

## 2.4 모멘텀·변동성·수급

* RSI level & slope, MACD & signal, `volume_z`, `ATR_smooth` & slope
* 캔들형 요약: 최근 N일 **하단침투 비율**, 꼬리 길이 평균 등

## 2.5 맥락

* 전일 이벤트 이후 경과일수, 최근 N일 이벤트 빈도
* 전고/전저 대비 괴리, 최근 낙폭 회복률

---

# 3) 모델 아키텍처 (3계층)

## (A) 상태/국면 추정기 (Regime Classifier)

* 입력: 최근 20~40일의 시계열(곡률/모멘텀/볼륨/밴드)
* 모델: **TCN**(1D dilated conv) 또는 LSTM (경량 시작 가능)
* 출력: `state ∈ {Accumulation, EarlyUp, Peak, Distribution, LateDown(U-bottom)}`
  → 특히 **LateDown(U-bottom)** 확률은 우리가 말한 “U형 하단 중앙” 확률

## (B) 이벤트-강화 예측기 (Multi-task Head)

* 입력: (A)의 state 확률 + 현재 시점 features + 이벤트 One-hot/조합
* 모델: **LightGBM**(탭형 강함) 또는 XGBoost
* 출력: `P_up = P(y_bin=1)`, `E_ret = E[y_ret]`, `E_dd = E[y_dd]` (멀티태스크)

> 왜 분리? — 시퀀스(형태) 이해는 TCN이, 임계·조합 의사결정은 GBDT가 강함.

## (C) 정책(Policy) 레이어

* 스코어: `Score = P_up * E_ret - λ * max(0, -E_dd)`

  * λ는 위험기피(예: 0.3~0.7 튜닝)

* 의사결정:

  * `Score ≥ θ_long` → **Long**
  * `Score ∈ (θ_flat_low, θ_long)` → **Flat**
  * `Score ≤ θ_short` & `state≈Peak or BD early` → 선택적으로 **Short**

* 포지션 사이징(예):
  `size = clip( k * Score / ex_ante_vol, 0, size_max )`
  (ex_ante_vol = 최근 10일 ATR/가격)

---

# 4) 학습·검증

* **시계열 CV:** expanding/walk-forward (누수 방지)

* 메트릭:

  * 분류: PR-AUC, Recall@TopK(알파는 희귀)
  * 회귀: MAE(ret, dd)
  * 전략: **E[ret]/stdev, Calmar(E[ret]/MDD), HitRatio, PnL(거래비용 포함)**

* **이벤트별 가중학습:** Volume Regain, BD_late의 중요도를 class_weight로 보정

---

# 5) 실행 로직(의사코드)

```python
# ── A) Regime (TCN)
state_prob = tcn.predict(seq_features[-40:])  # dict: {Accum, EarlyUp, Peak, Dist, LateDown}

# ── B) Features
x = concat([
  event_onehots, event_combos,  # I_volume_regain, I_bd_late, I_vr_and_vs...
  curvature, slopes, rsi/macd, volume_z, atr_slope,
  band_position, ema_spread_rel, context_feats,
  state_prob.values()
])

P_up, E_ret, E_dd = lgbm.predict(x)  # multi-task outputs

# ── C) Policy
Score = P_up * E_ret - lam * max(0, -E_dd)

if Score >= theta_long:
    side = "LONG"
    size = min(k * Score / ex_ante_vol, size_max)
elif Score <= theta_short and state_prob["Peak"]+state_prob["Accum"]<0.2:
    side = "SHORT"
else:
    side = "FLAT"

# Risk
stop = -min( max_stop, dynamic_from(E_dd, ATR) )
take = +dynamic_take_from(E_ret, skew=1.5)

# Conflict rule: VR+VS 동시면 단기 과열 → 이익실현 강화
if I_vr_and_vs:
    take *= 0.8
    hold_days = min(hold_days, 3)

# BD_late면 최초 1봉은 half-size, VR 동반시 full-size
if I_bd_late:
    size *= 0.5
    if I_volume_regain:
        size *= 2.0
```

---

# 6) 규칙 오버레이(우리 통찰 반영)

* **U형 하단 중앙(BD_late) ⇒ 롱 바이어스**

  * 단, `RSI<25` & `ATR↑↑` 급변동이면 1봉 지켜본 뒤 진입(“probe entry”).
* **VR+VS 동시 ⇒ 과열/천장 위험**

  * 롱 보유 시 **강제 트레일링 타이트닝**(ATR*1.0 → ATR*0.6)
  * 신규 진입 지양, 숏은 Regime=Peak일 때만 선택.
* **BD_early ⇒ 숏 또는 회피**

  * `curv_2<0 & EMA<0 & RSI_slope<0` 3조건 동시 충족 시만 액션.

---

# 7) 백테스트 체크리스트

* 거래비용(왕복 4~10bp), 슬리피지(ATR 기반) 반영
* **동일 이벤트 클러스터** 중복 진입 방지(쿨다운 3~5일)
* 레버리지 시 펀딩/이자 비용 포함
* **Drawdown constraint:** 월별 MDD > 10% 시 자동 리스크다운(사이즈 50%)

---

# 8) 우선순위 로드맵

1. **Baseline:** LightGBM 단일 모델(이벤트+곡률+모멘텀)로 `P_up, E_ret, E_dd` 멀티태스크
2. **+Regime(TCN):** state 확률 피쳐 추가 → 성능 갱신
3. **세분 BD:** `I_bd_early / I_bd_late` 태깅 정확도 개선(라벨 QA)
4. **Policy 튜닝:** θ/λ/k, hold_days, 트레일링 규칙 Bayesian Optimization
5. **포트폴리오:** 종목 확장 + 동시 포지션 상관/버짓 관리

---

# 9) 기대 효과

* “U형 하단 중앙=BD_late 롱 바이어스” 통찰을 **정식 피쳐/정책으로 고정**
* `Volume Regain`을 **핵심 알파 시그널**로 채택, `VR+VS`는 **청산 강화 규칙**으로 반영
* 이벤트-시퀀스-정책의 **분리 설계**로 해석력·안정성↑