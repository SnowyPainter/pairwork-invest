#!/usr/bin/env python3
"""
빠른 드리프트 분석 스크립트
2018-2020 vs 2021 데이터 비교
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# 프로젝트 경로 추가
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.M001_EventDetector import EventDetectorLGBM
import polars as pl

def check_concept_drift():
    """
    2018-2020 vs 2021 데이터 드리프트 분석
    """
    print("🔍 개념 드리프트 분석 (2018-2020 vs 2021)")

    try:
        # 학습 데이터 (2018-2020)
        print("\n[학습 데이터 로드: 2018-2020]")
        train_detector = EventDetectorLGBM(threshold=0.05, use_enhanced_features=True)
        train_df = train_detector.load_data(
            market="KR", years=[2018, 2019, 2020], max_tickers=50
        )

        # 테스트 데이터 (2021)
        print("\n[테스트 데이터 로드: 2021]")
        test_detector = EventDetectorLGBM(threshold=0.05, use_enhanced_features=True)
        test_df = test_detector.load_data(
            market="KR", years=[2021], max_tickers=50
        )

        if len(train_df) == 0 or len(test_df) == 0:
            print("❌ 데이터 로드 실패")
            return

        print(f"\n📊 데이터 크기 비교:")
        print(f"  학습 데이터: {len(train_df):,} 행")
        print(f"  테스트 데이터: {len(test_df):,} 행")
        print(f"  비율: {len(test_df)/len(train_df):.2f}")

        # 이벤트 빈도 비교
        train_events = train_df.select([
            pl.col("big_move_event").sum(),
            pl.col("big_up_event").sum(),
            pl.col("big_down_event").sum(),
            pl.len()
        ]).row(0)

        test_events = test_df.select([
            pl.col("big_move_event").sum(),
            pl.col("big_up_event").sum(),
            pl.col("big_down_event").sum(),
            pl.len()
        ]).row(0)

        print(f"\n📈 이벤트 빈도 비교:")
        print(f"  대형 변동: 학습 {train_events[0]/train_events[3]:.3f} vs 테스트 {test_events[0]/test_events[3]:.3f}")
        print(f"  상승 이벤트: 학습 {train_events[1]/train_events[3]:.3f} vs 테스트 {test_events[1]/test_events[3]:.3f}")
        print(f"  하락 이벤트: 학습 {train_events[2]/train_events[3]:.3f} vs 테스트 {test_events[2]/test_events[3]:.3f}")

        # 주요 피처 분포 비교
        key_features = ['vwap20', 'rsi5', 'atr14', 'cmf20', 'obv']
        print(f"\n📉 주요 피처 분포 비교:")

        for feature in key_features:
            if feature in train_df.columns and feature in test_df.columns:
                train_mean = train_df.select(pl.col(feature).mean()).row(0)[0]
                test_mean = test_df.select(pl.col(feature).mean()).row(0)[0]
                train_std = train_df.select(pl.col(feature).std()).row(0)[0]
                test_std = test_df.select(pl.col(feature).std()).row(0)[0]

                print(f"  {feature}:")
                print(f"   학습 평균: {train_mean:.4f}, 표준편차: {train_std:.4f}")
                print(f"   테스트 평균: {test_mean:.4f}, 표준편차: {test_std:.4f}")
                print(f"   변동률: {abs(train_mean - test_mean) / abs(train_mean):.1%}")

            # 변동폭 비교
        print(f"\n💹 변동폭 패턴 비교:")
        train_volatility = train_df.select(pl.col("intraday_range").mean()).row(0)[0]
        test_volatility = test_df.select(pl.col("intraday_range").mean()).row(0)[0]

        print(f"   학습 변동폭: {train_volatility:.4f}, 테스트 변동폭: {test_volatility:.4f}")
        print(f"   변동률: {abs(train_volatility - test_volatility) / train_volatility:.1%}")

        # 드리프트 지표 계산
        drift_score = 0
        reasons = []

        # 이벤트 빈도 드리프트
        event_drift = abs(train_events[0]/train_events[3] - test_events[0]/test_events[3])
        if event_drift > 0.02:
            drift_score += 1
            reasons.append(f"이벤트 빈도 차이: {event_drift:.3f}")

        # 변동성 드리프트
        vol_drift = abs(train_volatility - test_volatility) / train_volatility
        if vol_drift > 0.1:
            drift_score += 1
            reasons.append(f"변동성 변화: {vol_drift:.1%}")

        # 피처 분포 드리프트
        feature_drift_count = 0
        for feature in key_features:
            if feature in train_df.columns and feature in test_df.columns:
                train_mean = train_df.select(pl.col(feature).mean()).row(0)[0]
                test_mean = test_df.select(pl.col(feature).mean()).row(0)[0]
                if abs(train_mean - test_mean) / abs(train_mean) > 0.15:
                    feature_drift_count += 1

        if feature_drift_count >= 2:
            drift_score += 1
            reasons.append(f"{feature_drift_count}개 피처 분포 변화")

        print(f"\n🎯 드리프트 분석 결과:")
        if drift_score == 0:
            print("  ✅ 큰 드리프트 없음 - 모델 재사용 가능")
        elif drift_score == 1:
            print("  ⚠️ 약한 드리프트 - 모델 성능 모니터링 필요")
        else:
            print("  ❌ 강한 드리프트 - 모델 재학습 권장")
            print(f"  이유: {', '.join(reasons)}")

        return drift_score, reasons
    except Exception as e:
        print(f"❌ 드리프트 분석 실패: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    print("🚀 빠른 드리프트 분석 시작")
    print("=" * 60)

    # 드리프트 분석 실행
    drift_score, reasons = check_concept_drift()

    print("\n" + "=" * 60)
    print("📋 분석 결과 요약:")

    if drift_score is None:
        print("❓ 드리프트 분석 실패 - 수동 검토 필요")
    elif drift_score == 0:
        print("✅ 2021년 데이터는 2018-2020과 유사 - 기존 모델 사용 가능")
    elif drift_score == 1:
        print("⚠️ 2021년 데이터에 약한 변화 - 성능 모니터링 필요")
        if reasons:
            print(f"   발견된 변화: {', '.join(reasons)}")
    elif drift_score >= 2:
        print("❌ 2021년 데이터가 크게 다름 - 모델 재학습 필요")
        if reasons:
            print(f"   주요 차이점: {', '.join(reasons)}")

    print("\n💡 추천 액션:")
    if drift_score is None:
        print("   • 드리프트 분석 실패로 인한 수동 검토 필요")
        print("   • 데이터 로딩 및 모델 import 확인")
    elif drift_score == 0:
        print("   • 기존 모델 그대로 사용")
        print("   • 정기적인 모니터링만 수행")
    elif drift_score == 1:
        print("   • 2021년 데이터로 성능 테스트")
        print("   • 필요시 파인튜닝 고려")
    else:
        print("   • 2021년 데이터를 학습에 포함")
        print("   • 피처 재설계 검토")
        print("   • 모델 아키텍처 재검토")