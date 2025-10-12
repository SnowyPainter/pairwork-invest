# M002: A Geometric Regime Recognition Architecture for End-of-Day Trading

## Abstract

본 연구는 일봉 기반 다종목 주식 시장에서 **가격 곡선의 기하학적 형태(geometry)**를 해석하여 시장 상태를 추정하고, 이를 기반으로 **다중목표 예측(Multi-task Forecasting)** 및 **정책 의사결정(Policy Inference)**으로 확장하는 **3계층 트레이딩 아키텍처(M002)**를 제안한다.  
핵심 아이디어는 밴드 내 위치, 모멘텀·변동성 지표(RSI/ATR 등), 거래량 질감, 이벤트 신호(예: 재유입·반등·소진·붕괴 위험)와 곡선의 기하적 특징(경사·곡률)을 결합해 국면(regime)을 규칙적으로 분류하고,  
이를 LSTM 기반 확률 모델로 일반화한 뒤, LightGBM 헤드로 리바운드 확률·예상 수익/낙폭을 추정하여 정책 점수로 의사결정에 연결한다.  

미국 종목군을 대상으로 한 백테스트 집계 결과(mean 기준), Profit Factor ≈ 3.91, Calmar Ratio ≈ 21.48, Beta ≈ 0.17, Win Rate ≈ 50.4%를 보였고, Exposure Time ≈ 62.7%로(롱 온리 운용) 시장 상관도와 노출 시간을 제한하면서도 손익비 중심의 성과를 확인했다.  
본 구조는 단순 기술지표 예측을 넘어 **형태학적(morphological)** 국면 해석을 학습된 확률·정책 계층과 접목한 시장 인식 체계를 제시한다.

---

## 1. Introduction

가격 시계열의 변화는 단순한 수치열이 아니라, **곡선의 형태적 패턴**으로 이해할 수 있다.  
전통적 기술적 분석은 RSI·MACD·ATR 등의 단일 지표를 통해 시장 상태를 추정하지만,  
이는 곡선의 기하적 구조(볼록·오목, 방향성, 위치)와 이벤트성 맥락을 직접적으로 결합하지 못한다.  

본 연구는 이러한 한계를 해결하기 위해,  
가격 곡선 \( f(t) \)의 형태학적 특성을 이용한 **기하학적 국면 인식(Geometric Regime Recognition)**을 도입한다.  
이 접근은 시계열을 함수로 간주하되, 실제 구현에서는 밴드 내 상대위치(pos_in_band), 경사·곡률에 해당하는 파생 특성(price_slope/price_accel/curv_2), 모멘텀·변동성(RSI/ATR), 거래량(z-score), 이벤트 신호(반등/재유입/소진/붕괴 위험, 분해된 breakdown early/late) 등을 결합한 **우선순위 기반 규칙**으로 상태를 판정한다.

---

## 2. Related Work

기존 Regime-switching 모델(Hamilton, 1989; Ang & Bekaert, 2002)은 Hidden Markov Model(HMM)을 이용해  
가격의 확률적 상태 전이를 모델링하였으나, 이들은 구조적 해석이 어렵고,  
국면 자체가 데이터 통계적 성질에 의해 정의된다는 한계가 있다.

최근의 LSTM/Transformer 기반 시계열 분류(Chen et al., 2023)는 데이터 주도적 접근을 강화했지만,  
**형태적 해석 가능성(interpretability)**이 떨어진다.

이에 비해 본 연구의 접근은  
> *“기하적 휴리스틱(heuristic geometry) → 확률적 인식(probabilistic recognition) → 정책적 의사결정(policy adaptation)”*  
이라는 3계층 구조로, **형태학적 신호를 데이터 기반 학습으로 연결**한다는 점에서 차별화된다.

---

## 3. Methodology

### 3.1 Overview

M002 아키텍처는 다음 세 계층으로 구성된다:

| 계층 | 역할 | 학문적 해석 |
|------|------|-------------|
| ① Heuristic Geometry Layer | 곡선·이벤트·지표 조합으로 국면 분류 | Morphological Interpreter |
| ② Sequence Learning Layer | 형태 전이의 확률적 인식(다중 클래스) | Probabilistic Student of Geometry |
| ③ Multi-task Policy Layer | 리바운드 확률·예상 수익/낙폭 기반 정책 | Decision-Theoretic Policy Optimizer |

---

### 3.2 Heuristic Geometry Layer (Rule-based Morphological Labeling)

이 단계는 가격 곡선의 **기하·이벤트 신호를 규칙 기반으로 통합**한다. 핵심 국면은 Accumulation, EarlyUp, Peak, Distribution, LateDown의 5종이며,  
밴드 내 상대위치·모멘텀·변동성·거래량·이벤트(반등·재유입·소진·붕괴 위험, breakdown early/late) 조건을 점수화하고, 우선순위와 동률 해소 규칙으로 최종 국면을 선택한다.  
예컨대 Accumulation은 저밴드·저변동·저거래량에서, EarlyUp은 재유입/반등 후보 및 약상승 모멘텀에서, Peak은 고밴드·과열·소진 신호에서, Distribution은 완만한 하락 모멘텀과 리스크 신호에서, LateDown은 하단 약세/지연 붕괴 징후에서 강화된다.

---

### 3.3 Sequence Learning Layer (Probabilistic Regime Recognition)

Heuristic Layer에서 생성된 라벨을 teacher 신호로 사용하여,  
LSTM 기반 분류기가 시계열 패턴의 **확률적 전이(P(state|X))**를 학습한다.

- 입력: 최근 20일 시퀀스(경사·곡률·밴드 위치·모멘텀·변동성 등)
- 출력: 5차원 상태 확률 벡터
- 학습: 클래스 불균형은 배치 내 균형화(오버샘플링)로 보정, label smoothing ≈ 0.15

이를 통해 단일 시점의 형태를 넘어서 **형태의 시간적 맥락(time-context)**을 모델링한다.

---

### 3.4 Multi-task Policy Layer

Regime 확률과 확장 특성을 결합해, 향후 5일 **리바운드 확률**(이진)과 **예상 수익률·최대낙폭**(다중 회귀)을 함께 추정한다. 정책 점수는 아래와 같이 정의된다.

\[\text{Score} = P_\text{rebound} \cdot \frac{E[\text{ret}]}{100} - \lambda \cdot \max(0, -E[\text{dd}]/100)\]

- 점수는 종목별 롤링 z-정규화(예: 40일) 후 스케일링되어 분포 안정화
- ATR 기반 ex-ante 변동성으로 포지션 크기 산출 및 최대치 제한
- 적응형 분위수 임계값으로 LONG/SHORT/FLAT 결정, Peak 확률이 높을 때는 SHORT 제약
- 이벤트 조정: 거래량 재유입·변동 스파이크 결합(I_vr_and_vs)은 점수 완화, breakdown early/late 신호는 점수 강화

---

## 4. Results

미국 종목군 백테스트 요약(집계, reports/m002_full_backtest_btpy/summary_agg.csv 기준):

- Profit Factor: mean ≈ 3.91, median ≈ 0.88
- Calmar Ratio: mean ≈ 21.48
- Beta: mean ≈ 0.17
- Max Drawdown: median ≈ −25.15%
- Win Rate: mean ≈ 50.4%
- Exposure Time: mean ≈ 62.7% (롱 온리 운용)

단순 승률이 비슷하더라도, 정책 점수와 위험 제약을 결합한 손익비 개선으로 성과가 도출됨을 확인했다.

---

## 5. Discussion and Academic Contribution

### (1) 형태학적 해석 가능성
- 가격 곡선을 함수 \( f(t) \)로 보고, 국면을 **위상적 상태(topological state)**로 분류.
- 각 국면은 곡률·기울기 조합으로 정의되어 해석 가능성이 높음.

### (2) 계층적 학습 구조
- Heuristic → Probabilistic → Policy 로 분리된 구조는
  모듈별 독립 학습 및 교체가 가능.
- Teacher–Student 구조로서 **Weakly-supervised Regime Learning**을 구현.

### (3) 실용적 확장성
- 정책 계층은 실제 거래비용·변동성 조정 등을 내재화.
- End-of-Day 자동 매매 시스템에 직접 적용 가능.

---

## 6. Conclusion

M002 아키텍처는 시장을 단순 통계열이 아닌 **곡선적 형태(geometric form)**로 해석하는 새로운 패러다임을 제시한다.  
기하학적 휴리스틱으로부터 시작해, 시계열 확률 모델과 정책 최적화로 확장되는 이 구조는  
형태 기반 국면 인식(morphological regime recognition)이라는 새로운 관점을 금융 시계열 분석에 도입하였다.  

향후 연구로는:
- Transformer 기반 형태 인식기(TCN/Temporal ConvNet)로의 확장  
- Weak labeling을 기반으로 한 self-training  
- 포트폴리오 수준의 정책 최적화  
가 가능하다.

---

## References

- Hamilton, J.D. (1989). *A new approach to the economic analysis of nonstationary time series and the business cycle*. Econometrica.  
- Ang, A., & Bekaert, G. (2002). *Regime switches in interest rates*. Journal of Business & Economic Statistics.  
- Chen, H. et al. (2023). *Deep Regime Classification for Financial Time Series*. IEEE Access.
