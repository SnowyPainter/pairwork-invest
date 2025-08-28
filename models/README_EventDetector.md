# Event Detector - FT-Transformer

주식 이벤트 탐지를 위한 FT-Transformer 기반 모델

## 📊 개요

**Event Detector**는 주식 데이터에서 큰 가격 변동 이벤트(±5%)를 예측하는 이진 분류 모델입니다.

### 🎯 목적
- **이벤트 탐지**: 다음날 ±5% 이상 변동할 가능성이 높은 주식 식별
- **변동성 예측**: 거래량/변동성 지표를 통한 극단적 움직임 포착
- **리스크 관리**: 고변동성 구간 사전 탐지로 포지션 조절

### 🏗️ 아키텍처
- **FT-Transformer**: Feature Tokenizer + Multi-Head Attention
- **입력**: 12개 변동성/거래량 기반 기술적 지표
- **출력**: 이벤트 확률 (0~1)

## 🔧 설치 및 설정

```bash
# 프로젝트 루트에서
cd /home/snowypainter/pairwork-invest

# 필요 패키지 설치 (이미 설치되어 있다면 생략)
pip install torch torchvision torchaudio
pip install scikit-learn
```

## 🚀 사용법

### 1. 모델 훈련

#### 간단한 실행:
```bash
# 기본 설정으로 훈련
./train_event_detector.sh
```

#### 상세 설정으로 훈련:
```bash
python models/train_event_detector.py \
    --market KR \
    --years 2018,2019,2020 \
    --max_tickers 100 \
    --epochs 50 \
    --batch_size 512 \
    --lr 0.001 \
    --model_name "my_event_detector"
```

#### 주요 매개변수:
- `--market`: 시장 선택 (KR/US)
- `--years`: 학습 데이터 연도 (쉼표로 구분)
- `--max_tickers`: 최대 종목 수
- `--epochs`: 학습 에포크 수
- `--batch_size`: 배치 크기
- `--lr`: 학습률
- `--d_model`: 모델 차원 (기본: 192)
- `--n_heads`: 어텐션 헤드 수 (기본: 8)
- `--dropout`: 드롭아웃 비율 (기본: 0.1)

### 2. 이벤트 예측

```bash
python models/predict_events.py \
    --model_path models/checkpoints/event_detector_final.pth \
    --data_path your_data.csv \
    --threshold 0.5 \
    --output_path predictions.parquet
```

#### 매개변수:
- `--model_path`: 훈련된 모델 경로
- `--data_path`: 예측할 데이터 파일
- `--threshold`: 예측 임계값 (기본: 0.5)
- `--output_path`: 결과 저장 경로

### 3. Python 코드에서 사용

```python
from models.event_detector import EventDetector, EventDetectorTrainer, create_event_detector
from data.dataset_builder import build_dataset

# 데이터 로드
df = build_dataset(
    years=[2020, 2021],
    market="KR",
    max_tickers=50,
    feature_set="v2",
    label_task="classification"
)

# 모델 생성
model = create_event_detector(n_features=12)
trainer = EventDetectorTrainer(model)

# 훈련
history = trainer.train(df, epochs=30)

# 평가
results = trainer.evaluate(df)
print(f"AUC: {results['auc']:.4f}")

# 예측
features, labels, _ = trainer.prepare_data(df)
predictions, probabilities = trainer.predict(features)
```

## 📈 모델 성능

### 입력 피처 (12개)
분석 결과 이벤트 탐지에 가장 유효한 지표들:

1. **`rel_range`**: 상대적 변동폭 - **7% 성공률** ⭐
2. **`vol10`**: 10일 변동성 - **6% 성공률**
3. **`obv`**: On-Balance Volume - **6% 성공률**
4. **`parkinson20`**: Parkinson 변동성 - **6% 성공률**
5. **`gk20`**: Garman-Klass 변동성 - **5% 성공률**
6. **`vol20`**: 20일 변동성 - **5% 성공률**
7. **`atr5/10/14`**: Average True Range - 방향성 50%
8. **`tr14`**: True Range 14일
9. **`vol_roc5`**: 거래량 변화율
10. **`vol_z20`**: 거래량 Z-score

### 예상 성능 지표
- **AUC**: 0.65-0.75 (랜덤: 0.5)
- **정확도**: 60-70%
- **이벤트 재현율**: 40-60%
- **이벤트 정밀도**: 15-25%

### 성능 해석
- **극단값 탐지에 특화**: 일반적인 작은 변동보다 큰 이벤트 예측에 유리
- **변동성 증가 신호**: `rel_range`, `vol10` 등이 강한 신호
- **방향성 한계**: 이벤트 발생은 예측하지만 상승/하락 방향은 별도 모델 필요

## 📁 파일 구조

```
models/
├── event_detector.py           # 메인 모델 클래스
├── train_event_detector.py     # 훈련 스크립트
├── predict_events.py           # 추론 스크립트
├── checkpoints/                # 모델 체크포인트
│   ├── event_detector_best.pth
│   ├── event_detector_final.pth
│   └── event_detector_history.csv
└── README_EventDetector.md     # 이 파일

train_event_detector.sh         # 간편 실행 스크립트
```

## 🔄 워크플로우

### 1단계: 이벤트 탐지
```
입력 데이터 → Event Detector → 이벤트 확률
```

### 2단계: 방향 분류 (추후 개발)
```
이벤트 감지된 데이터 → Direction Classifier → 상승/하락 예측
```

### 3단계: 트레이딩 시스템
```
이벤트 + 방향 → 포지션 결정 + 리스크 관리
```

## ⚠️ 주의사항

1. **클래스 불균형**: 이벤트는 전체 데이터의 10-20%만 차지
2. **과적합 위험**: 과거 데이터에만 최적화될 수 있음
3. **시장 환경**: 학습 기간과 다른 시장 상황에서 성능 저하 가능
4. **방향성 부재**: 이벤트 발생만 예측, 상승/하락 방향은 예측 안 함

## 📊 모니터링

### 학습 과정 모니터링
- **Validation AUC**: 0.7 이상 목표
- **Early Stopping**: Validation AUC 기준 15 에포크
- **Learning Rate Scheduling**: AUC 개선 정체 시 자동 감소

### 운영 모니터링
- **일일 이벤트 예측률**: 5-15% 범위 유지
- **고신뢰도 예측** (prob ≥ 0.8): 전체의 1-5%
- **실제 이벤트 적중률**: 월별 추적

## 🔮 향후 개발 계획

1. **Direction Classifier**: 이벤트 방향 예측 모델 추가
2. **Multi-Timeframe**: 1일, 3일, 5일 다중 시간대 예측
3. **Ensemble Model**: 여러 모델 조합으로 성능 향상
4. **Real-time Pipeline**: 실시간 데이터 처리 및 예측
5. **Backtesting Framework**: 전략 검증 시스템

## 📞 문의

모델 관련 문의나 개선 제안은 이슈 등록 또는 직접 연락 바랍니다.

---

*Created: $(date)*  
*Model: FT-Transformer Event Detector*  
*Version: 1.0*
