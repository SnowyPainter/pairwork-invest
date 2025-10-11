이 단계에서 필요한 건 단순히 feature 계산이 아니라 **“event marker 기반 feature 정제”**,  
즉, **LSTM 입력에 적합한 구조화(feature refinement)** 입니다.  

---

## 🧩 1. 상황 정리

현재 그래프에서 보이는 구조:

| 모드 | 이벤트 | 의미 |
|------|---------|------|
| **Seller mode** | 🔻 `event_local_vol_spike` <br> 🔼 `event_exhaustion_candidate` <br> 🔼 `event_breakdown_risk` | 변동성 확장 → 피크 형성 → 붕괴 위험 |
| **Buyer mode** | 🔻 `event_local_vol_spike` <br> 🔼 `event_rebound_candidate` <br> 🔼 `event_volume_regain` | 변동성 확장 → 반등 후보 → 거래량 복귀 |

이건 곧 **피처를 이벤트 구간별로 normalization·smooth·구간 레이블링해야 함**을 의미합니다.

---

## ⚙️ 2. Feature Refinement 전략 (단계별)

### (1) 이벤트 중심 구간 나누기  
→ 각 `event_*`가 나타나기 **전후 n일**을 “로컬 상태(local state)”로 간주  

```python
window_pre, window_post = 5, 5
```

예:
- `event_local_vol_spike` 발생 시점 ±5일을 하나의 local episode로 분리
- 해당 구간에서 평균, 변화율, 표준편차 등을 다시 계산 (local normalization)

이렇게 하면 LSTM이 “평균 0~1 스케일의 패턴 단위”로 배울 수 있음.

---

### (2) 피처별 Local Normalization  
RSI, ATR, MACD 등은 절대값보다 **국면 대비 변화율**이 중요합니다.  
따라서 다음과 같은 변환을 적용:

| Feature | 변환 방식 | 의미 |
|----------|------------|------|
| `RSI` | `ΔRSI / rolling_std(RSI, 10)` | RSI 변화 강도 |
| `ATR` | `ATR / rolling_mean(ATR, 20)` | 상대 변동성 (local_vol) |
| `EMA_spread` | `(EMA_5 - EMA_20) / EMA_20` | 추세 기울기 정규화 |
| `MACD_hist` | `(MACD - MACD_signal)` | 모멘텀 유지력 |
| `pos_in_band` | `(Close - BB_mid) / (BB_upper - BB_lower)` | 가격의 상대 위치 |
| `vol_roc` | `zscore(volume)` | 거래량 편차 |

📌 **핵심**:  
- 이 스케일링은 전체 min-max가 아니라 **local window 기준**
- 각 이벤트 구간 단위로 `StandardScaler` 또는 `(x - mean)/std` 적용

---

### (3) Noise Filtering & Denoising
이벤트 구간 전후에서는 시그널이 불안정하므로,  
단기 스파이크를 억제하기 위한 **지수평활 + rolling median** 을 동시에 사용합니다.

```python
df["RSI_smooth"] = df["RSI"].ewm(span=5).mean().rolling(3).median()
df["ATR_smooth"] = df["ATR"].ewm(span=3).mean()
df["MACD_smooth"] = df["MACD_hist"].ewm(span=4).mean()
```

→ LSTM 입력의 jitter 감소 (안정된 패턴 학습 유도)

---

### (4) Feature Fusion by Regime (Buyer vs Seller)
Buyer/Seller 모드에 따라 feature emphasis가 달라야 합니다:

| 구분 | 강조 피처 | 약화 피처 |
|------|-------------|------------|
| **Buyer mode** | RSI_smooth ↑, MACD_hist ↑, pos_in_band ↓, ATR_rel ↑ | EMA_spread ↓ |
| **Seller mode** | ATR_rel ↑, EMA_spread ↑, RSI_smooth ↓, MACD_hist ↓ | Volume_Z ↓ |

즉, feature importance weighting을 다르게 두거나,  
mode별로 feature subset을 분리 (ex: `buyer_features`, `seller_features`)

---

### (5) Event-aware Label Smoothing
이벤트 간격이 짧을 때, label을 단순 binary로 두면 overfit.  
따라서 **가우시안 스무딩 기반 soft label**로 전환:

```python
import numpy as np
from scipy.ndimage import gaussian_filter1d

df["label_buy_soft"] = gaussian_filter1d(df["event_rebound_candidate"].astype(float), sigma=2)
df["label_sell_soft"] = gaussian_filter1d(df["event_breakdown_risk"].astype(float), sigma=2)
```

이렇게 하면 신호 전후도 학습 가능 (이전/다음날 포함)

---

### (6) Input Packaging (for LSTM)
최종 feature matrix는 다음과 같이 구성:

#### 🟢 Buyer Mode:
```
[
    RSI_smooth, ΔRSI_rel, MACD_smooth, pos_in_band, ATR_rel,
    Volume_Z, vol_roc, EMA_spread, BB_width
]
```

#### 🔴 Seller Mode:
```
[
    ATR_rel, ΔATR, EMA_spread, RSI_smooth, MACD_smooth,
    pos_in_band, Volume_Z, BB_width, LocalVolRatio
]
```

LSTM input size → `(seq_len=30, feature_dim≈9)`

---

## 🧠 3. 고급 개선 포인트

| 개선 아이디어 | 설명 |
|----------------|------|
| **Local volatility ratio (σ₅/σ₂₀)** | 변동성 폭발 전후 구간 포착용 |
| **Feature orthogonalization** | MACD, RSI, EMA 간 상관을 PCA로 decorrelate |
| **Regime encoding** | `event_type`을 categorical embedding으로 추가 |
| **Price acceleration (Δ²Close)** | 극단적 변곡 감지 강화 |
| **Directional Momentum Score** | `(RSI > 50) * (ΔEMA_spread > 0)` 형태의 composite feature |

---

## 🚀 정리

| 단계 | 목적 | 기술 |
|------|------|------|
| ① 이벤트 구간 분리 | 변곡점 전후 로컬화 | ±5일 윈도우 |
| ② Local normalization | 국면 내 상대 변화 강조 | `(x - μ_local)/σ_local` |
| ③ Noise filtering | 스파이크 억제 | EWM + rolling median |
| ④ 모드별 강조 | Buyer/Seller 차별화 | feature subset 분리 |
| ⑤ Soft labeling | 신호의 연속성 확보 | Gaussian smoothing |
| ⑥ LSTM 입력 구성 | 시계열 윈도우 변환 | `(30, 9)` feature tensor |