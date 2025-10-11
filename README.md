# Pairwork Invest Workflow

`pnpm` 스크립트로 백테스트 · 학습 · 분석 파이프라인을 실행할 수 있도록 구성했습니다.  
Python 환경을 그대로 사용하면서도, 추후 다른 모델을 추가할 때 스크립트만 확장하면 되도록 설계되었습니다.

## 1. 준비

```bash
# Node 18+ / pnpm 8+ 권장
pnpm install            # 의존성 없음, pnpm-lock 생성만 수행
pnpm run py:deps        # Python 패키지 설치 (requirements.txt 기준)
```

> `pnpm run py` 스크립트는 현재 활성화된 Conda 가상환경(`CONDA_PYTHON_EXE`, `CONDA_PREFIX`)을 최우선으로 사용합니다.  
> WSL 기본값은 `/home/snowypainter/miniconda3/envs/research` 환경의 파이썬을 가리키며, 일반 가상환경을 켠 상태라면 해당 인터프리터가 자동 선택됩니다.  
> 실행 시 프로젝트 루트를 `PYTHONPATH`에 자동으로 추가하므로, `from backtester...` 같은 내부 모듈 임포트가 바로 동작합니다.

## 2. 주요 워크플로우

```bash
# 방향 분류 모델 학습 (LightGBM)
pnpm run train:direction

# M001 통합 백테스트 (EventDetector + DirectionClassifier)
pnpm run backtest:m001

# 방향 분류기 단독 백테스트
pnpm run backtest:direction

# 방향 피처 분석 리포트 생성
pnpm run analytics:direction
```

## 3. 커스텀 실행

임의의 파이썬 모듈/스크립트를 실행하고 싶다면 `pnpm run py`를 사용하세요.

```bash
# 예시: 빠른 백테스트 모드
pnpm run py backtester/backtest_m001.py -- --quick

# 예시: 데이터셋 빌더 모듈 실행 (인자 전달 가능)
pnpm run py -m data.dataset_builder
```

스크립트에 인자를 넘길 때는 `--` 이후에 작성하면 `pnpm`이 그대로 전달합니다.

## 4. 구조와 확장 가이드

- `package.json`의 `scripts` 섹션에 새로운 명령을 추가하면 다른 모델 워크플로우도 쉽게 연결할 수 있습니다.
- 공통 파이썬 런처는 `scripts/run-python.mjs`에 정의되어 있습니다. 필요한 경우 후보 인터프리터를 수정하거나, `process.env.PYTHON`에 원하는 경로를 지정하면 됩니다.
- 장기적으로 모델별 디렉터리에 CLI 엔트리포인트를 제공하면 `scripts` 항목만 추가하면서 통합 관리할 수 있습니다.

## 5. 참고

- Python 패키지 목록은 `requirements.txt`에서 관리합니다.
- 빌드/백테스트 결과물은 기존과 동일하게 `reports/` 하위에 생성됩니다.
- Node 의존성은 아직 없지만, 추후 필요한 도구(예: 문서/리포트 자동화)를 추가할 때 `pnpm add <pkg>`로 쉽게 확장 가능합니다.
