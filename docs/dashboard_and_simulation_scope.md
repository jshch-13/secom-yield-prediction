# 대시보드 / 시뮬레이션 범위 정의 (Scope Definition)

이 프로젝트는 성격이 다른 두 종류의 산출물을 포함합니다. **둘을 절대 같은 표, 같은 문장, 같은 화면에서
혼합해서 해석해서는 안 됩니다.**

1. **SECOM ML 분석** — 공개된 실제 SECOM 센서 데이터를 사용한 분석/모델링 결과
   (`src/`, `notebooks/`, `dashboard/`, `outputs/interactive/*.html` 중 SECOM 4종).
2. **Synthetic Wafer Map Demo** — `simulations/`가 코드로 직접 생성한 가상 좌표 데이터를 이용한
   시각화/리포팅 엔지니어링 데모 (`simulations/`, `outputs/simulations/`,
   `data/processed/synthetic_wafer_map_demo.csv`, `docs/synthetic_wafer_map_demo.html`).

## 범위 비교표

| 구분 | 데이터 출처 | 가능한 분석 | 불가능하거나 주장하면 안 되는 분석 |
|---|---|---|---|
| SECOM ML 분석 | 공개 익명 센서 데이터 (UCI/Kaggle SECOM) | 결측치 분석, 불균형 분류, feature importance, 불량 예측, threshold tuning | 센서의 물리적 의미 단정, 설비/챔버 traceback, wafer spatial pattern, 실제 Fab 수율 원인 규명 |
| Synthetic Wafer Demo | `simulations/generate_synthetic_wafer_data.py`가 코드로 생성한 가상 좌표 데이터 | wafer map 시각화, EDGE_RING/CENTER/SCRATCH/RANDOM/CLEAN 패턴 시각화, 규칙 기반 패턴 분류 데모, batch PDF 리포팅 | 실제 Fab 수율 분석, 실제 공정 원인 규명, SECOM 기반 결과라는 주장, 검증된 프로덕션 분류기라는 주장 |

## 데이터/산출물 위치가 섞이지 않도록 하는 규칙

- **폴더 분리**: SECOM 관련 코드는 `src/`, `notebooks/`, `dashboard/`에, 합성 웨이퍼 데모는 `simulations/`에만 둡니다.
- **출력 분리**: SECOM 산출물은 `outputs/figures/`, `outputs/metrics/`, `outputs/interactive/`에, 합성 데모
  산출물은 `outputs/simulations/`에만 저장합니다.
- **데이터 분리**: SECOM 정제 데이터는 `data/processed/secom_*.csv`, 합성 데모 데이터는
  `data/processed/synthetic_wafer_map_demo.csv`로 파일명 자체를 구분합니다.
- **표기 규칙**: 합성 데이터를 다루는 모든 화면/문서/파일은 "[SYNTHETIC DEMO]" 또는 "Synthetic data - Not
  derived from SECOM or production Fab data" 문구를 반드시 포함합니다 (그래프 제목, PDF 리포트 매 페이지,
  README 섹션 첫 문장 등).
- **스키마 검증**: `simulations/wafer_pattern_classifier.py`는 입력 데이터에 `synthetic_data_flag=True`가
  없으면 예외를 발생시켜, 실수로 실제 데이터를 합성 데모 코드에 흘려보내는 것을 코드 레벨에서 차단합니다.

## 각 산출물의 성격 요약

| 산출물 | 성격 |
|---|---|
| `outputs/interactive/interactive_feature_distribution.html` | SECOM 실제 데이터 |
| `outputs/interactive/interactive_scatter_matrix.html` | SECOM 실제 데이터 |
| `outputs/interactive/prediction_probability_dashboard.html` | SECOM 실제 데이터 |
| `outputs/interactive/feature_importance.html` | SECOM 실제 데이터 |
| `dashboard/app.py` (Dash) | SECOM 실제 데이터 |
| `outputs/simulations/wafer_map_*.png` | 코드 생성 합성 데이터, [SYNTHETIC DEMO] |
| `outputs/simulations/synthetic_wafer_demo_report.pdf` | 코드 생성 합성 데이터, [SYNTHETIC DEMO] |
| `simulations/wafer_pattern_classifier.py` 결과 | Rule-based demonstration classifier (검증된 모델 아님) |

## 결론

- SECOM ML 분석 결과(Recall/Precision/PR-AUC 등)를 인용할 때는 항상 `outputs/metrics/`, `data/processed/secom_*`
  출처를 명시합니다.
- Synthetic Wafer Demo 결과를 인용할 때는 항상 "합성 데이터/데모"임을 명시하고, SECOM 분석 결과와 나란히
  배치하거나 같은 성능 지표 표에 넣지 않습니다.
