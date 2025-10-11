#!/usr/bin/env python3
"""
M002 Episode Analysis - 변곡점 매매 전략 평가 및 분석

이 스크립트는 에피소드 데이터를 활용하여:
1. 이벤트 타입별 성과 분석
2. 전략 백테스트 시뮬레이션
3. 리스크 분석
4. 매매 타이밍 최적화 제안
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# 스타일 설정
plt.style.use('default')
sns.set_palette("husl")


class M002EpisodeAnalyzer:
    """M002 에피소드 데이터를 분석하는 클래스"""

    def __init__(self, episodes_json_path: str = "reports/m002/episodes_detailed.json"):
        self.episodes_json_path = Path(episodes_json_path)
        self.episodes_data = self._load_episodes_data()
        self.episodes_df = self._create_episodes_dataframe()

    def _load_episodes_data(self) -> Dict[str, Any]:
        """에피소드 JSON 데이터 로드"""
        with open(self.episodes_json_path, 'r') as f:
            return json.load(f)

    def _create_episodes_dataframe(self) -> pd.DataFrame:
        """에피소드 데이터를 DataFrame으로 변환"""
        episodes = self.episodes_data['episodes']
        df = pd.DataFrame(episodes)

        # 날짜 변환
        df['start_date'] = pd.to_datetime(df['start_date'])
        df['end_date'] = pd.to_datetime(df['end_date'])

        # mode 컬럼이 있다면 제거 (legacy 데이터 호환성)
        if 'mode' in df.columns:
            df = df.drop('mode', axis=1)

        return df

    def analyze_event_performance(self) -> pd.DataFrame:
        """이벤트 타입별 성과 분석"""
        print("📊 이벤트 타입별 성과 분석")
        print("=" * 50)

        performance_stats = []

        for event_type in self.episodes_df['event_type'].unique():
            event_episodes = self.episodes_df[self.episodes_df['event_type'] == event_type]

            stats = {
                'event_type': event_type,
                'total_episodes': len(event_episodes),
                'avg_price_change': event_episodes['price_change_pct'].mean(),
                'median_price_change': event_episodes['price_change_pct'].median(),
                'std_price_change': event_episodes['price_change_pct'].std(),
                'positive_episodes': (event_episodes['price_change_pct'] > 0).sum(),
                'success_rate': (event_episodes['price_change_pct'] > 0).mean() * 100,
                'avg_duration': event_episodes['duration_days'].mean(),
                'total_return': event_episodes['price_change_pct'].sum()
            }

            performance_stats.append(stats)

        # DataFrame 생성 및 정렬
        perf_df = pd.DataFrame(performance_stats)
        perf_df = perf_df.sort_values('avg_price_change', ascending=False)

        # 결과 출력
        print(perf_df.to_string(index=False, float_format='%.2f'))
        print()

        return perf_df

    def analyze_event_distribution(self) -> pd.DataFrame:
        """이벤트별 분포 및 성과 분석"""
        print("📊 이벤트 분포 분석")
        print("=" * 30)

        event_stats = []

        for event_type in self.episodes_df['event_type'].unique():
            event_episodes = self.episodes_df[self.episodes_df['event_type'] == event_type]

            stats = {
                'event_type': event_type,
                'total_episodes': len(event_episodes),
                'avg_price_change': event_episodes['price_change_pct'].mean(),
                'success_rate': (event_episodes['price_change_pct'] > 0).mean() * 100,
                'avg_duration': event_episodes['duration_days'].mean(),
                'total_return': event_episodes['price_change_pct'].sum()
            }

            event_stats.append(stats)

        event_df = pd.DataFrame(event_stats).sort_values('total_episodes', ascending=False)

        print(event_df.to_string(index=False, float_format='%.2f'))
        print()

        return event_df

    def simulate_trading_strategy(self, capital: float = 10000) -> Dict[str, Any]:
        """간단한 매매 전략 시뮬레이션"""
        print("🎯 매매 전략 시뮬레이션")
        print("=" * 30)

        # 각 이벤트 타입별로 개별 포지션 시뮬레이션
        strategy_results = {}

        for event_type in self.episodes_df['event_type'].unique():
            event_episodes = self.episodes_df[self.episodes_df['event_type'] == event_type].copy()

            # 각 에피소드에 동일 금액 투자 시뮬레이션
            position_size = capital / len(event_episodes) if len(event_episodes) > 0 else 0
            total_return = 0

            for _, episode in event_episodes.iterrows():
                # 가격 변동을 수익률로 적용
                episode_return = episode['price_change_pct'] / 100
                total_return += position_size * episode_return

            final_capital = capital + total_return

            strategy_results[event_type] = {
                'initial_capital': capital,
                'final_capital': final_capital,
                'total_return': total_return,
                'total_return_pct': (final_capital - capital) / capital * 100,
                'num_trades': len(event_episodes),
                'avg_return_per_trade': total_return / len(event_episodes) if len(event_episodes) > 0 else 0
            }

        # 결과 출력
        results_df = pd.DataFrame(strategy_results).T
        results_df = results_df.sort_values('total_return_pct', ascending=False)

        print(f"초기 자본: ${capital:,.0f}")
        print(results_df.to_string(float_format='%.2f'))
        print()

        return strategy_results

    def analyze_risk_metrics(self) -> Dict[str, Any]:
        """리스크 메트릭 분석"""
        print("⚠️ 리스크 분석")
        print("=" * 20)

        risk_metrics = {}

        for event_type in self.episodes_df['event_type'].unique():
            event_episodes = self.episodes_df[self.episodes_df['event_type'] == event_type]

            returns = event_episodes['price_change_pct']

            if len(returns) > 1:
                # 샤프 비율 계산 (무위험 수익률 0% 가정)
                avg_return = returns.mean()
                std_return = returns.std()
                sharpe_ratio = avg_return / std_return if std_return > 0 else 0

                # 최대 손실
                max_drawdown = returns.min()

                # 승률
                win_rate = (returns > 0).mean() * 100

                risk_metrics[event_type] = {
                    'sharpe_ratio': sharpe_ratio,
                    'max_drawdown': max_drawdown,
                    'win_rate': win_rate,
                    'volatility': std_return,
                    'best_trade': returns.max(),
                    'worst_trade': returns.min()
                }
            else:
                risk_metrics[event_type] = {
                    'sharpe_ratio': 0,
                    'max_drawdown': returns.iloc[0] if len(returns) > 0 else 0,
                    'win_rate': 100 if returns.iloc[0] > 0 else 0,
                    'volatility': 0,
                    'best_trade': returns.iloc[0] if len(returns) > 0 else 0,
                    'worst_trade': returns.iloc[0] if len(returns) > 0 else 0
                }

        # 결과 출력
        risk_df = pd.DataFrame(risk_metrics).T
        risk_df = risk_df.sort_values('sharpe_ratio', ascending=False)

        print(risk_df.to_string(float_format='%.2f'))
        print()

        return risk_metrics

    def generate_strategy_recommendations(self) -> Dict[str, Any]:
        """전략 추천 생성"""
        print("🎯 전략 추천")
        print("=" * 20)

        # 성과 분석
        perf_df = self.analyze_event_performance()
        risk_metrics = self.analyze_risk_metrics()

        # 추천 전략 선정 기준:
        # 1. 평균 수익률 > 0%
        # 2. 샤프 비율 > 0.5
        # 3. 승률 > 50%
        # 4. 최대 손실 > -5%

        recommendations = []

        for event_type in perf_df['event_type']:
            perf = perf_df[perf_df['event_type'] == event_type].iloc[0]
            risk = risk_metrics.get(event_type, {})

            score = 0
            reasons = []

            # 평균 수익률 체크
            if perf['avg_price_change'] > 0:
                score += 2
                reasons.append(f"평균 수익 양호 (+{perf['avg_price_change']:.2f}%)")
            else:
                reasons.append(f"평균 수익 부진 ({perf['avg_price_change']:+.2f}%)")
            # 샤프 비율 체크
            sharpe = risk.get('sharpe_ratio', 0)
            if sharpe > 0.5:
                score += 2
                reasons.append(f"샤프 비율 우수 ({sharpe:.2f})")
            elif sharpe > 0:
                score += 1
                reasons.append(f"샤프 비율 보통 ({sharpe:.2f})")

            # 승률 체크
            win_rate = risk.get('win_rate', 0)
            if win_rate > 60:
                score += 1
                reasons.append(f"승률 우수 ({win_rate:.1f}%)")
            elif win_rate > 50:
                score += 0.5
                reasons.append(f"승률 보통 ({win_rate:.1f}%)")

            # 최대 손실 체크
            max_dd = risk.get('max_drawdown', 0)
            if max_dd > -5:
                score += 1
                reasons.append(f"최대 손실 적정 ({max_dd:+.1f}%)")
            elif max_dd > -10:
                score += 0.5
                reasons.append(f"최대 손실 주의 ({max_dd:+.1f}%)")

            # 에피소드 수 체크
            if perf['total_episodes'] >= 5:
                score += 1
                reasons.append(f"충분한 샘플 수 ({perf['total_episodes']}개)")
            elif perf['total_episodes'] >= 3:
                score += 0.5
                reasons.append(f"샘플 수 보통 ({perf['total_episodes']}개)")
            recommendation = {
                'event_type': event_type,
                'recommendation_score': score,
                'recommendation': '강력 추천' if score >= 4 else '추천' if score >= 3 else '보류' if score >= 2 else '비추천',
                'reasons': reasons,
                'expected_return': perf['avg_price_change'],
                'win_rate': risk.get('win_rate', 0),
                'sharpe_ratio': risk.get('sharpe_ratio', 0),
                'max_drawdown': risk.get('max_drawdown', 0)
            }

            recommendations.append(recommendation)

        # 정렬 및 출력
        rec_df = pd.DataFrame(recommendations)
        rec_df = rec_df.sort_values('recommendation_score', ascending=False)

        print("전략 추천 결과:")
        for _, rec in rec_df.iterrows():
            print(f"{rec['event_type']}: {rec['recommendation']} (점수: {rec['recommendation_score']:.1f})")
            print(f"  → 기대 수익: {rec['expected_return']:+.2f}%, 승률: {rec['win_rate']:.1f}%")
            print(f"  → 샤프비율: {rec['sharpe_ratio']:.2f}, 최대손실: {rec['max_drawdown']:+.1f}%")
            for reason in rec['reasons']:
                print(f"     • {reason}")
            print()

        return {'recommendations': recommendations, 'summary': rec_df}

    def create_analysis_json(self) -> Dict[str, Any]:
        """JSON 형식으로 종합 분석 결과 생성"""
        print("📋 JSON 분석 결과 생성")
        print("=" * 30)

        # 모든 분석 실행 (출력 없이)
        perf_df = self.analyze_event_performance()
        event_df = self.analyze_event_distribution()
        strategy_results = self.simulate_trading_strategy()
        risk_metrics = self.analyze_risk_metrics()
        recommendations = self.generate_strategy_recommendations()

        # 가장 수익률 좋은 에피소드 찾기
        best_episode = self.episodes_df.loc[self.episodes_df['price_change_pct'].idxmax()].to_dict()
        worst_episode = self.episodes_df.loc[self.episodes_df['price_change_pct'].idxmin()].to_dict()

        # 모든 이벤트들을 통합하여 분석 후 재분배
        all_event_performance = {}

        # 모든 이벤트 타입에 대해 성과 분석
        for event_type in self.episodes_df['event_type'].unique():
            event_data = self.episodes_df[self.episodes_df['event_type'] == event_type]
            all_event_performance[event_type] = {
                'count': len(event_data),
                'avg_return': float(event_data['price_change_pct'].mean()),
                'win_rate': float((event_data['price_change_pct'] > 0).mean() * 100),
                'best_episode': float(event_data['price_change_pct'].max()),
                'worst_episode': float(event_data['price_change_pct'].min()),
                'volatility': float(event_data['price_change_pct'].std()),
                'total_return': float(event_data['price_change_pct'].sum())
            }

        # 성과 기반으로 Buyer/Seller 이벤트 재분배
        # 규칙: 평균 수익률 > 0.5% → Buyer, < -0.5% → Seller, 그 외 → Neutral
        buyer_event_performance = {}
        seller_event_performance = {}
        neutral_event_performance = {}

        for event_type, stats in all_event_performance.items():
            avg_return = stats['avg_return']
            if avg_return > 0.5:  # 양수 수익률이 좋은 이벤트 → 매수 신호
                buyer_event_performance[event_type] = stats
            elif avg_return < -0.5:  # 음수 수익률이 나쁜 이벤트 → 매도 신호
                seller_event_performance[event_type] = stats
            else:  # 중립 이벤트
                neutral_event_performance[event_type] = stats

        # 재분배 결과 요약 출력
        print("🎯 이벤트 재분배 결과:")
        print(f"  총 이벤트 수: {len(all_event_performance)}")
        print(f"  → Buyer 이벤트: {len(buyer_event_performance)}개")
        print(f"  → Seller 이벤트: {len(seller_event_performance)}개")
        print(f"  → Neutral 이벤트: {len(neutral_event_performance)}개")
        print()

        if buyer_event_performance:
            print("🟢 재분배된 Buyer 이벤트들:")
            for event, stats in buyer_event_performance.items():
                print(f"  • {event}: +{stats['avg_return']:.2f}% (승률: {stats['win_rate']:.1f}%)")
            print()

        if seller_event_performance:
            print("🔴 재분배된 Seller 이벤트들:")
            for event, stats in seller_event_performance.items():
                print(f"  • {event}: {stats['avg_return']:+.2f}% (승률: {stats['win_rate']:.1f}%)")
            print()

        if neutral_event_performance:
            print("⚪ Neutral 이벤트들:")
            for event, stats in neutral_event_performance.items():
                print(f"  • {event}: {stats['avg_return']:+.2f}% (승률: {stats['win_rate']:.1f}%)")
            print()

        # 에피소드별 피처 패턴 분석 (상위 5개 에피소드)
        top_episodes = self.episodes_df.nlargest(5, 'price_change_pct')[['episode_id', 'event_type', 'price_change_pct', 'feature_statistics']].to_dict('records')

        analysis_result = {
            "metadata": {
                "total_episodes": len(self.episodes_df),
                "date_range": f"{self.episodes_df['start_date'].min()} to {self.episodes_df['end_date'].max()}",
                "tickers_analyzed": list(self.episodes_df['ticker'].unique()),
                "analysis_date": pd.Timestamp.now().isoformat()
            },
            "overall_statistics": {
                "total_episodes": len(self.episodes_df),
                "overall_win_rate": float((self.episodes_df['price_change_pct'] > 0).mean() * 100),
                "avg_episode_duration": float(self.episodes_df['duration_days'].mean()),
                "avg_price_change": float(self.episodes_df['price_change_pct'].mean()),
                "median_price_change": float(self.episodes_df['price_change_pct'].median()),
                "best_episode_return": float(self.episodes_df['price_change_pct'].max()),
                "worst_episode_return": float(self.episodes_df['price_change_pct'].min())
            },
            "best_performing_episode": {
                "episode_id": int(best_episode['episode_id']),
                "ticker": best_episode['ticker'],
                "event_type": best_episode['event_type'],
                "price_change_pct": float(best_episode['price_change_pct']),
                "duration_days": int(best_episode['duration_days']),
                "start_date": best_episode['start_date'].isoformat(),
                "end_date": best_episode['end_date'].isoformat(),
                "feature_values": best_episode['feature_statistics']
            },
            "worst_performing_episode": {
                "episode_id": int(worst_episode['episode_id']),
                "ticker": worst_episode['ticker'],
                "event_type": worst_episode['event_type'],
                "price_change_pct": float(worst_episode['price_change_pct']),
                "duration_days": int(worst_episode['duration_days']),
                "start_date": worst_episode['start_date'].isoformat(),
                "end_date": worst_episode['end_date'].isoformat(),
                "feature_values": worst_episode['feature_statistics']
            },
            "event_performance": {
                "buyer_events": buyer_event_performance,
                "seller_events": seller_event_performance,
                "neutral_events": neutral_event_performance,
                "all_events_performance": all_event_performance,
                "redistribution_summary": {
                    "total_events": len(all_event_performance),
                    "redistributed_to_buyer": len(buyer_event_performance),
                    "redistributed_to_seller": len(seller_event_performance),
                    "remained_neutral": len(neutral_event_performance),
                    "redistribution_logic": "avg_return > 0.5% → Buyer, < -0.5% → Seller, else → Neutral"
                }
            },
            "top_performing_episodes": top_episodes,
            "strategy_recommendations": recommendations['recommendations'],
            "risk_metrics": risk_metrics
        }

        # JSON 파일로 저장
        with open("reports/m002/episode_analysis.json", "w", encoding="utf-8") as f:
            json.dump(analysis_result, f, indent=2, ensure_ascii=False)

        print("✅ 분석 결과가 reports/m002/episode_analysis.json에 저장되었습니다.")
        print(f"📊 총 {len(self.episodes_df)}개 에피소드 분석 완료")
        print(f"📉 가장 나쁜 에피소드: {worst_episode['event_type']} ({worst_episode['price_change_pct']:+.2f}%)")

        return analysis_result


def main():
    """메인 분석 함수"""
    print("🚀 M002 에피소드 분석 시작")
    print("=" * 40)

    try:
        analyzer = M002EpisodeAnalyzer()

        # JSON 분석 결과 생성
        analyzer.create_analysis_json()

        print("\n✅ 분석 완료! 자세한 결과는 reports/m002/episode_analysis.json을 확인하세요.")

    except FileNotFoundError:
        print("❌ 에피소드 데이터 파일을 찾을 수 없습니다.")
        print("   먼저 'pnpm run features:m002:combined'를 실행해서 데이터를 생성하세요.")
    except Exception as e:
        print(f"❌ 분석 중 오류 발생: {e}")


if __name__ == "__main__":
    main()
