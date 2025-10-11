# Backtester (vectorbt 기반)

기존 백테스터를 삭제하고 vectorbt로 전환했습니다.

## 설치

```bash
pip install -r requirements.txt
```

vectorbt가 추가되었습니다.

## 백테스트 파일

### 1. M001_backtest.py

M001 모델 (EventDetector + DirectionClassifier) 백테스트

**매매 로직:**
- EventDetector가 이벤트 발생을 예측 (확률 ≥ 0.5)
- DirectionClassifier가 상승 방향을 예측하면 Long
- 일별 상위 20개 시그널만 선택

**실행:**
```bash
python backtester/M001_backtest.py
```

**필요 모델:**
- `models/saved/tcn_event_detector_KR_2018_2019_2020_100pct_L60.pth`
- `models/saved/direction_classifier_KR_2018_2019_2020.txt`

**파라미터 조정:**
```python
backtester = M001Backtester(
    event_model_path="...",
    direction_model_path="...",
    event_threshold=0.5,      # 이벤트 확률 임계값
    top_n_events=20,          # 일별 최대 시그널 수
    commission=0.001,         # 수수료율 (0.1%)
    initial_cash=10_000
)
```

### 2. M002_baseline_backtest.py

M002 Baseline (MultiTask 모델) 백테스트

**매매 로직:**
- Trigger 확률 ≥ 0.5
- 기대 수익률 ≥ 1.0%
- 기대 낙폭 ≥ -3.0%
- 일별 상위 20개 시그널만 선택

**실행:**
```bash
python backtester/M002_baseline_backtest.py
```

**파라미터 조정:**
```python
backtester = M002BaselineBacktester(
    model=model,
    trigger_threshold=0.5,       # 트리거 확률 임계값
    top_n_signals=20,            # 일별 최대 시그널 수
    min_expected_return=1.0,     # 최소 기대 수익률 (%)
    max_drawdown=-3.0,           # 최대 허용 낙폭 (%)
    commission=0.001,
    initial_cash=10_000
)
```

### 3. M002_full_backtest.py

M002 Full Architecture (RegimeClassifier + MultiTask + Policy) 백테스트

**매매 로직:**
- Policy score > 0 (Long 포지션)
- 일별 상위 20개 시그널만 선택
- Position size에 따른 크기 조정

**실행:**
```bash
python backtester/M002_full_backtest.py
```

**파라미터 조정:**
```python
backtester = M002FullBacktester(
    model=model,
    top_n_signals=20,   # 일별 최대 시그널 수
    commission=0.001,
    initial_cash=10_000
)
```

## 출력

각 백테스트는 다음을 생성합니다:

1. **콘솔 출력**: 통계 요약
   - Total Return
   - Sharpe Ratio
   - Max Drawdown
   - Win Rate
   - 등등

2. **CSV 파일**: `reports/{model_name}_backtest/stats.csv`
   - 전체 통계 저장

3. **HTML 차트**: `reports/{model_name}_backtest/equity_curve.html`
   - 자산 곡선 시각화
   - 브라우저에서 인터랙티브 차트

## vectorbt 장점

1. **속도**: 기존 백테스터보다 훨씬 빠름 (NumPy 벡터화)
2. **기능**: 다양한 성과 지표 자동 계산
3. **시각화**: 인터랙티브 차트 제공
4. **유연성**: 다양한 매매 전략 지원
5. **검증됨**: 널리 사용되는 라이브러리

## 기존 백테스터와 차이점

### 기존 (삭제됨)
- Custom implementation (981 lines)
- Polars 기반 데이터 처리
- 느린 속도
- 제한적인 기능

### 새로운 (vectorbt)
- 검증된 라이브러리 사용
- Pandas 기반 (vectorbt 요구사항)
- 빠른 속도 (벡터화)
- 풍부한 성과 지표
- 인터랙티브 시각화

## 주의사항

1. **데이터 형식**: vectorbt는 Pandas를 사용하므로 Polars → Pandas 변환 필요
2. **메모리**: 큰 데이터셋의 경우 메모리 사용량 주의
3. **모델 경로**: 저장된 모델 파일 경로 확인 필요

## 예시

```python
# M001 백테스트
from backtester.M001_backtest import M001Backtester

backtester = M001Backtester(
    event_model_path="models/saved/...",
    direction_model_path="models/saved/..."
)

portfolio = backtester.run(
    market="KR",
    years=[2021, 2022],
    max_tickers=100
)

# 결과 확인
print(portfolio.stats())
portfolio.plot().show()
```

