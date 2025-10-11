# M002 아키텍처 개요

**M002_FullArchitecture**는 다음 세 계층으로 구성된 파이프라인을 구현합니다.

1. **Feature & Label Builder**  
   `build_dataset(feature_set="m002")` → `features/feature_sets.py:add_m002_features()`  
   - 가격/볼륨 기반 기본 피처에 더해 곡률(`curv_2`), 밴드 포지션(`pos_in_band_rel`), 이벤트 헐리스틱(`event_*`, `I_bd_early/late`, `I_vr_and_vs`), 컨텍스트(`days_since_*`, `event_*_freq20`) 등을 생성합니다.  
   - `_compute_future_returns()`로 5거래일 후 수익률·드로다운(`ret_5d_pct`, `dd_5d_pct`)을 계산하고 `label_rebound_bin`을 부여합니다.

2. **Regime Classifier (LSTM)** – `models/M002_RegimeClassifier.py`  
   - 입력: 최근 `seq_len(기본 20)`일 동안의 시계열 피처(곡률·모멘텀·밴드·거래량).  
   - 라벨: `_assign_regime_labels()`가 규칙 기반으로 `Accumulation/EarlyUp/Peak/Distribution/LateDown`을 부여. LateDown은 `I_bd_late`와 곡률·모멘텀 조합으로 확대하여 U형 하단 후보를 더 넓게 포착합니다.  
   - 학습: PyTorch LSTM + FC. 클래스 불균형을 완화하기 위해  
     * `balance_classes=True`일 때 inverse-frequency 가중치(`class_weight_power`)를 CrossEntropyLoss와 `WeightedRandomSampler`에 적용  
     * `label_smoothing`(기본 0.05)으로 노이즈를 줄입니다.  
   - 출력: `predict_probabilities()`가 티커/날짜별 `state_prob_{name}` 벡터를 생성.

3. **Multi-task Head + Policy Layer** – `models/M002_FullArchitecture.py`

   ### 3.1 Head Dataset 준비
   - `_prepare_head_dataset()`이 m002 피처 + state 확률을 합쳐 pandas DataFrame을 만듭니다.  
   - 80/20 시계열 분할로 학습/검증 세트를 구성합니다.

   ### 3.2 LightGBM Multi-task Head
   - 분류기: rebound 확률(`P_up`)  
   - 회귀기: 5일 기대수익(`E_ret`), 최대 드로다운(`E_dd`)  
   - `_default_head_params()`로 초기 하이퍼파라미터를 정의하고, `_fit_head_models()`에서 LightGBM + `MultiOutputRegressor`를 학습합니다.
   - **Optuna 기반 AutoML**  
     * `tune_head_hyperparams(n_trials=...)` 호출 시 TPE/Random 샘플러로 학습  
     * 목적함수 = `valid_avg_precision + 0.1 * valid_policy_mean`  
     * 최적 파라미터는 `self.best_head_params`에 저장되어 이후 `train()`에 자동 적용됩니다.

   ### 3.3 Policy Layer
   - 기본 스코어: `Score = P_up * (E_ret/100) - λ * max(0, -(E_dd/100))` (λ=`risk_aversion`)  
   - `PolicyConfig` 주요 파라미터  
     * `theta_long=0.05`, `theta_short=-0.05`, `theta_flat_low=0.0`  
     * `rescale_window=60`, `score_scale=2.5`: 티커별 롤링 Z-score로 `policy_score_rescaled`를 생성해 분포를 확대  
     * `restrict_short_on_peak_prob`: Peak 확률이 높을 때만 숏 허용  
   - `_apply_policy()`  
     * `atr_smooth` 기반 ex-ante volatility로 포지션 사이징  
     * `I_vr_and_vs`는 과열 시그널로 0.8배 축소, `I_bd_late`가 있으면 1.1배 확대  
     * 최종 `action ∈ {LONG, FLAT, SHORT}`, `position_size`는 `[-size_max, size_max]`로 클리핑.

---

## 데이터 흐름 요약

```
build_dataset("m002") ─┬─> add_m002_features ──> labeled pl.DataFrame
                       └─> RegimeClassifier.train() ──> state_prob_*

state_prob_* + m002 features ──> LightGBM Head (Optuna 튜닝 가능) ──> P_up, E_ret, E_dd
                                                         │
                                                         ▼
                                  Policy Layer (z-score rescale, thresholds, rules)
                                                         ▼
                              action / position_size / policy_score / probabilities
```

---

## 사용 예시

```python
from models.M002_FullArchitecture import M002FullArchitecture, FullArchitectureConfig

config = FullArchitectureConfig(verbose=True)
arch = M002FullArchitecture(config)

# (선택) Optuna 튜닝
arch.tune_head_hyperparams(n_trials=50)

# 학습
metrics = arch.train()

# 추론
pred = arch.predict(feature_polars_df)
```

---

## 체크포인트
- Regime LSTM: 클래스 분포·가중치가 `metrics['class_counts']`, `metrics['class_weights']`로 로그됩니다.  
- Head 튜닝: `tune_head_hyperparams` 결과가 `best_head_params` / `best_head_metrics`에 저장됩니다.  
- Policy Score: `_apply_policy`가 리스케일 후 스코어 분포를 자동 조절하므로, 필요하면 `score_scale`·`theta_*` 값을 조정하세요.

이 구조를 통해 이벤트 시퀀스 → 탭형 예측 → 정책 의사결정이 명확히 분리되어, 각 계층을 독립적으로 튜닝하거나 교체할 수 있습니다.
