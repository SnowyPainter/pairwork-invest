#!/usr/bin/env python3

import sys
from pathlib import Path

# 프로젝트 모듈 경로 (backtest_m001.py의 코드)
sys.path.append(str(Path(__file__).parent))

print("Testing imports...")

try:
    from models.M001_DirectionClassifier import DirectionClassifierLGBM
    print("✓ M001_DirectionClassifier import successful")
except Exception as e:
    print("✗ M001_DirectionClassifier import failed:", e)

try:
    from models.M001_EventDetector import EventDetectorManager
    print("✓ M001_EventDetector import successful")
except Exception as e:
    print("✗ M001_EventDetector import failed:", e)

try:
    from data.dataset_builder import build_dataset
    print("✓ data.dataset_builder import successful")
except Exception as e:
    print("✗ data.dataset_builder import failed:", e)

try:
    from backtester.backtester import BacktestConfig
    print("✓ backtester.backtester import successful")
except Exception as e:
    print("✗ backtester.backtester import failed:", e)
