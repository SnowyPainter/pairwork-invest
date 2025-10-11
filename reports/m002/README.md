# M002 Feature Analysis - 변곡점 매매 전략

이 폴더는 M002 피처 탐색기의 시각화 결과물을 포함합니다.

## 📊 전략 개요

**변곡점 매매 (Inflection Point Trading)**:
- **🟢 BUY**: 그래프가 오목한 곳 → 볼록하게 변하는 지점 (가격 저점)
- **🔴 SELL**: 그래프가 볼록한 곳 → 오목하게 변하는 지점 (가격 고점)

## 📁 폴더 구조

```
reports/m002/
├── buyer/                    # 매수 모드 분석 결과
├── seller/                   # 매도 모드 분석 결과
├── [TICKER]_combined_events.html    # Buyer/Seller 동시 비교 차트
├── [TICKER]_episodes.html           # 에피소드 시각화
├── combined_info.json              # 통합 메타데이터 + 에피소드 요약
├── episodes_detailed.json          # 상세 에피소드 정보
└── README.md                       # 이 파일
```

## 🎯 생성되는 시각화 파일들

### 1. Feature Panels (`[TICKER]_feature_panels.html`)
**변곡점 매매 분석을 위한 핵심 시각화**

- **가격 & 추세 지표**: 정규화된 가격과 기술적 지표들을 겹쳐서 표시
  - 검은색: 기본 가격선
  - 다양한 색상: EMA, MACD, RSI 등 추세 지표들 (스무딩 적용)
  - 🟢 삼각형: BUY 신호 (오목→볼록 변곡점)
  - 🔴 역삼각형: SELL 신호 (볼록→오목 변곡점)

- **거래량 지표**: 거래량 관련 피처들
- **모멘텀 지표**: RSI, 스토캐스틱 등
- **변동성 지표**: ATR, 볼린저 밴드 등
- **이벤트 신호 & 레이블**: 매매 이벤트 신호들
  - 이진 이벤트: `event_*` (0/1 값, 단계 그래프로 표시)
  - 소프트 레이블: `label_*_soft` (가우시안 스무딩된 연속값, 채워진 영역으로 표시)

### 2. Events Visualization (`[TICKER]_events.html`)
**이벤트 기반 매매 신호 표시**

- 가격 차트에 이벤트 마커 표시
- Buyer 모드: `event_rebound_candidate`, `event_volume_regain`, `event_local_vol_spike`
- Seller 모드: `event_exhaustion_candidate`, `event_breakdown_risk`, `event_local_vol_spike`
- 공통: `label_buy_soft`, `label_sell_soft` (소프트 레이블)

### 3. Correlation Heatmap (`[TICKER]_correlations.html`)
**피처 간 상관관계 분석**

### 4. Combined Event Analysis (`[TICKER]_combined_events.html`) - Combined 모드 전용
**Buyer와 Seller 이벤트의 동시 비교**

- **가격 차트 with 이벤트 구간**: 녹색/빨강 영역으로 이벤트 구간 표시
- **Buyer 이벤트 패널**: 매수 이벤트 신호들
- **Seller 이벤트 패널**: 매도 이벤트 신호들
- **색상 구분**: 🟢 녹색 계열 = Buyer 이벤트 | 🔴 빨강 계열 = Seller 이벤트

### 5. Episode Visualization (`[TICKER]_episodes.html`) - Combined 모드 전용
**로컬 에피소드 분석**

- **Buyer Mode Episodes**: 매수 이벤트 주변 지역 분석
- **Seller Mode Episodes**: 매도 이벤트 주변 지역 분석
- **색칠된 구간**: 각 에피소드별로 구분하여 표시
- **에피소드 라벨**: 이벤트 타입과 ID 표시

### 6. Combined Info JSON (`combined_info.json`) - Combined 모드 전용
**통합 메타데이터 및 에피소드 요약 통계**

```json
{
  "tickers": ["AAPL"],
  "episode_summary": {
    "total_episodes": 72,
    "by_mode": {"buyer": 34, "seller": 38},
    "by_event_type": {
      "event_rebound_candidate": {"buyer": 5, "seller": 0},
      "event_breakdown_risk": {"buyer": 0, "seller": 13}
    },
    "avg_episode_length": {"buyer": 5.4, "seller": 5.1}
  }
}
```

### 7. Episodes Detailed JSON (`episodes_detailed.json`) - Combined 모드 전용
**개별 에피소드의 상세 분석 데이터**

- **기본 정보**: 에피소드 ID, 티커, 모드, 이벤트 타입, 기간, 가격 변동
- **기술적 지표 통계**: 각 피처의 평균, 표준편차, 최솟값/최댓값, 시작/종료 값
- **이벤트 신호 분석**: 에피소드 내 이벤트 활성화 기간, 최대/평균 강도
- **에피소드별 데이터 포인트**: 시계열 데이터 구조화

### 8. Episode Analysis JSON (`episode_analysis.json`) - analyze 명령어 실행 시
**종합 에피소드 전략 분석 결과**

- **metadata**: 분석 메타정보 (총 에피소드 수, 날짜 범위, 분석 대상)
- **overall_statistics**: 전체 통계 (승률, 평균 수익률, 최고/최저 성과)
- **best_performing_episode**: 최고 성과 에피소드 상세 정보 + 피처 값들
- **worst_performing_episode**: 최저 성과 에피소드 상세 정보 + 피처 값들
- **event_performance**: Buyer/Seller 이벤트별 성과 비교
  - `buyer_events`: 매수 이벤트별 통계 (event_rebound_candidate, event_volume_regain 등)
  - `seller_events`: 매도 이벤트별 통계 (event_exhaustion_candidate, event_breakdown_risk 등)
- **top_performing_episodes**: 상위 5개 에피소드 피처 패턴
- **strategy_recommendations**: AI 기반 전략 추천 결과
- **risk_metrics**: 각 이벤트별 리스크 지표 (샤프 비율, 최대 손실 등)

## 🚀 사용법

```bash
# Buyer 모드 시각화 생성
pnpm run features:m002:buyer -- --tickers AAPL MSFT --start 2024-01-01

# Seller 모드 시각화 생성
pnpm run features:m002:seller -- --tickers TSLA NVDA --start 2024-01-01

# Combined 모드: Buyer와 Seller 이벤트를 동시에 비교 + JSON 데이터
pnpm run features:m002:combined -- --tickers AAPL MSFT --start 2024-01-01

# 에피소드 전략 분석: 성과/리스크/추천 전략 생성
pnpm run features:m002:analyze

# 옵션들
--tickers: 분석할 티커들 (기본값: AAPL MSFT GOOGL)
--start: 시작 날짜 (기본값: 2023-01-01)
--end: 종료 날짜 (선택사항)
--progress: 진행상황 표시
```

## 📈 분석 포인트

### 변곡점 식별 방법
1. **가격 곡선의 2차 미분**을 통해 곡률 변화 감지
2. **스무딩 적용**으로 노이즈 제거 및 추세 파악
3. **정규화된 스케일**에서 다양한 지표들을 비교 분석

### 전략적 활용
- **오목 구간 진입**: 가격이 떨어지다가 상승 반전 시점
- **볼록 구간 진입**: 가격이 오르다가 하락 반전 시점
- **거래량 확인**: 변곡점 근처의 거래량 변화 관찰
- **모멘텀 지표**: RSI, MACD 등의 과매수/과매도 신호 확인

## 📊 데이터 파일들

- `summary_statistics.csv`: 각 티커의 기본 통계
- `event_summary.csv`: 이벤트 발생 빈도
- `feature_info.json`: 분석 메타데이터
- `combined_info.json`: 통합 메타데이터 + 에피소드 요약 (Combined 모드)
- `episodes_detailed.json`: 상세 에피소드 분석 데이터 (Combined 모드)
- `episode_analysis.json`: 종합 전략 분석 결과 (analyze 명령어 실행 시)

## 🔧 기술적 세부사항

- **스무딩 윈도우**: 5일 이동평균 (기본값)
- **정규화**: Min-Max 스케일링 [0,1] 범위
- **변곡점 감지**: 2차 미분의 제로 크로싱
- **색상 코딩**: 각 피처 그룹별 고유 색상 사용
