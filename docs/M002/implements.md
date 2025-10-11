# M002 3계층 아키텍처 구현 방식

본 문서는 `features/feature_sets.py`, `models/M002_RegimeClassifier.py`, `models/M002_FullArchitecture.py` 등으로 구성된 M002 파이프라인이 **저수준 데이터 처리**부터 **정책 의사결정**까지 어떻게 동작하는지를 단계별로 정리합니다.

---

## 1. 데이터 기반 레이어 (Feature Pipeline)

### 1.1 Silver → LazyFrame 로딩
- `data/dataset_builder.py`의 `build_dataset()`이 로컬 Silver 데이터(티커별 OHLCV)를 `polars.LazyFrame` 형태로 스캔합니다.
- 사용자는 `feature_set="m002"`로 호출하여, 아래 커스텀 파이프라인을 적용합니다.

### 1.2 `add_m002_features()` (features/feature_sets.py:516)
- 기존 v1/v2/v3 피처 위에 다음을 추가:
  - **곡률/모멘텀**: `curv_2`, `price_accel`, `ema_spread_rel`, `atr_rel` 등.
  - **이벤트 헐리스틱**: `event_local_vol_spike`, `event_rebound_candidate`, `event_breakdown_risk` 등.
  - **조합 태그**: `I_bd_early`, `I_bd_late`, `I_vr_and_vs`.
  - **컨텍스트**: `event_volume_regain_freq20`, `days_since_volume_regain`.
- 모든 계산은 `polars` 표현식으로 처리 → 그룹별 shift/rolling, `rolling_map`을 통한 2차미분(Savitzky-Golay 근사), `over("ticker")` 기반 익스펙트, `fill_null` 로 누락값 처리.
- 결과물: 컬럼이 풍부한 `pl.DataFrame` (or `LazyFrame`) – 이후 모든 모델의 입력 기반.

---

## 2. 1계층: Regime Classifier (상태/국면 추정)

### 2.1 라벨링 (models/M002_RegimeClassifier.py:93)
- `_assign_regime_labels()`:
  - **우선순위 기반 병렬 평가**: 겹치는 이벤트가 많을 때 순서 의존성 제거.
  - 각 상태별 조건을 개별적으로 평가 후 우선순위 점수 부여:
    - Peak(5), Accumulation(4), EarlyUp(3), LateDown(2), Distribution(1)
  - `polars.when().then()`으로 각 상태의 점수 계산 → `pl.struct()`와 `map_elements()`로 최대 점수 상태 선택.
  - `STATE_TO_ID` 매핑으로 Int 라벨 (`regime_state_id`) 생성.
  - **조건 완화**: 엄격한 threshold를 완화하여 클래스 균형 개선 (예: curv_2 > -0.5, rsi_smooth > 40 등).

### 2.2 시퀀스 구축
- `_prepare_dataframe()`:
  - `build_dataset()` 호출 → `m002` 피처 포함 DataFrame 획득.
  - 티커별 정렬 후 `fill_null(forward)`로 feature 누락값 보정.
  - `_assign_regime_labels()` 적용, 유효 샘플만 필터링.
- `_build_sequences()`:
  - `pandas`로 변환 후, 티커별 20일(window) 슬라이딩 → `(N, 20, feature_dim)` numpy 배열과 라벨 vector 생성.

### 2.3 LSTM 학습
- `RegimeLSTM`: PyTorch `nn.LSTM` + FC layer.
- `RegimeSequenceDataset`으로 시퀀스/라벨을 `DataLoader`에 공급.
- **균형 배치 샘플링**: 희소 클래스의 gradient variance 문제를 해결하기 위해 `balanced_collate_fn()` 적용.
  - 각 배치 내에서 클래스별 oversampling: 소수 클래스를 repetition으로 균형 맞춤.
  - `WeightedRandomSampler` 대신 배치 수준 균형으로 학습 안정성 향상.
- `train()`:
  - AdamW, CrossEntropyLoss (weight=None, label_smoothing=0.15), max_epochs=15.
  - 최적 validation loss snapshot 저장.
  - 반환: train/valid loss, accuracy, 시퀀스 수.

### 2.4 추론
- `predict_probabilities(pl.DataFrame)`:
  - 입력 DataFrame을 티커별 정렬, feature null 제거.
  1. 티커 기준으로 연속 구간 확보 → 20일(seq_len) 미만 구간은 패스.
  2. 각 window를 tensor로 변환 후 softmax.
  3. 원래 date index에 맞춰 state_prob_* 컬럼 생성 → `pl.DataFrame` 반환.

---

## 3. 2계층: Multi-task Head (LightGBM)

### 3.1 데이터 준비
- `M002FullArchitecture._load_base_dataframe()`:
  - `build_dataset()`으로 `pl.DataFrame` 획득 후 `_compute_future_returns()`로 미래 수익률/드로다운(±5일) 계산.
  - `label_rebound_bin` = 미래 ret ≥ +1%, drawdown ≥ -3%.

### 3.2 상태 확률 조인
- `train()` 내부에서 RegimeClassifier 추론 → `state_prob_*` 컬럼 join (`pl.DataFrame.join`).
- `fill_null({state_prob_col:0})`로 미측정 행 보정.

### 3.3 학습 입력 준비
- 학습 컬럼: `HEAD_FEATURES` (기존 baseline features + Regime Prob + 이벤트/곡률/컨텍스트).
- `polars` → `pandas` 변환 (`select` 후 `.to_pandas()`).
- `dropna`, inf 제거.
- 80/20 time-based split.

### 3.4 LightGBM 학습
- 분류기: `lgb.LGBMClassifier` (binary logloss, 600 estimators).
- 회귀기: `MultiOutputRegressor(lgb.LGBMRegressor)` – 두 타깃 (ret, dd) 동시 학습.
- 검증: 평균 정밀도, MAE(ret), MAE(dd), Policy Score 평균 계산.
- `trained_state` 사전에 Regime/Head 메트릭 저장.

---

## 4. 3계층: Policy Layer

### 4.1 Policy Score 계산
- `_soft_policy_score(probs, ret, dd, λ)`:
  - `Score = P_up * (ret/100) - λ * max(0, -dd/100)` (퍼센트 → 소수 변환 주의).
  - λ는 `PolicyConfig.risk_aversion` (기본 0.5).

### 4.2 추가 규칙
- `_apply_policy()`:
  - `atr_smooth` 기반 `ex_ante_vol` (rolling mean 10일).
  - 포지션 사이징: `size = clip(k * score / ex_ante_vol, [-size_max, size_max])`.
  - 이벤트 기반 보정:
    - `I_vr_and_vs` → score 0.8배 (과열).
    - `I_bd_late` → score 1.1배 (롱 강화).
  - 의사결정 분기:
    - Score ≥ θ_long → LONG
    - Score ≤ θ_short & (`I_bd_early` or Peak prob ≥ threshold) → SHORT
    - 나머지 → FLAT
  - 최종 `position_size`는 양수 롱 / 음수 숏 / 0 플랫.

### 4.3 추론 출력 (`predict`)
- 입력: 이미 `m002` 피처를 갖고 있는 `pl.DataFrame`.
- 단계:
  1. RegimeClassifier로 state_prob column 생성·조인.
  2. (예측 시점용이라면 미래 ret/dd 없음) – `HEAD_FEATURES`만 추출.
  3. LightGBM head로 `pred_rebound_prob`, `pred_expected_ret_pct`, `pred_expected_dd_pct`, `policy_score`.
  4. `_apply_policy` → `action`, `position_size`, `ex_ante_vol` 등 포함한 final `pd.DataFrame`.

---

## 5. 종합 플로우

```
Raw OHLCV ──> build_dataset(feature_set="m002") ──> add_m002_features (Polars)
   │                                       │
   │ feed                                 └> rich features w/ event flags, curvature, context
   ▼
RegimeClassifier (LSTM w/ balanced batch sampling) ──> state_prob_* ─┐
   │ (priority-based labeling, oversampling)                         │ join
   ▼                                                                 ▼
   └─> regime_state_id ──> sequences ──> LSTM training ─────────────┘
LightGBM Head (P_up, E_ret, E_dd) ──> PolicyScore ──> Policy Layer
                                                          │
                                                          └> action, size, score, supporting columns
```

- **Polars**는 전체 파이프라인에서 주된 ETL 엔진 → LazyFrame으로 필요 시 가속.
- **PyTorch** LSTM은 상태 추정 (국면 파악), **LightGBM**은 이벤트+상태+컨텍스트를 결합한 회귀/분류 헤드.
- **Policy layer**는 도메인 규칙과 결합하여 최종 트레이딩 지시 신호 생성.

---

## 6. 확장 포인트

- **데이터 샘플링 전략**: 현재 `balanced_collate_fn` 기반 oversampling 외에 SMOTE, ADASYN 등 고급 샘플링 기법 실험.
- **라벨링 개선**: 우선순위 기반 → Semi-supervised (pseudo-labeling) or HMM 기반 상태 추정으로 발전.
- **모델 아키텍처**: LightGBM Multi-task 대신 XGBoost, TabNet, Transformer 기반 시퀀스 모델 실험.
- **학습 안정화**: curriculum learning (쉬운 샘플부터 어려운 샘플로) 또는 focal loss로 클래스 불균형 추가 대응.
- **Policy 튜닝**: 파라미터(θ, λ, k) 자동 최적화 (Bayesian Optimization) 및 Cooldown/포트폴리오 리스크 조정 로직 추가.
- **모니터링**: 상태 확률을 `features/M002_FeatureExplorer.py`에서 시각화하여 모델 해석/품질 모니터링.

---

위 구조를 통해 M002 모델은 이벤트-시퀀스-정책을 일관된 코드베이스에서 재활용 가능하며, 각 계층의 책임이 명확히 구분됩니다. 실제 백테스트/라이브 파이프라인에서는 **1→2→3 단계 결과를 캐싱**하여 재학습 비용을 줄이고, 정책 파라미터 튜닝을 반복 하는 방식으로 운영할 수 있습니다.
