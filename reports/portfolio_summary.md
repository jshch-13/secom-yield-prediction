# 포트폴리오 요약 (Portfolio Summary)

**프로젝트**: SECOM 기반 반도체 공정 센서 데이터 불량 예측 및 불균형 분류 분석
**데이터**: 공개·익명화된 UCI/Kaggle SECOM 데이터셋 (1,567건, 590개 익명 센서 feature, 불량률 6.64%)
**핵심 기술**: Python, pandas, scikit-learn, imbalanced-learn (SMOTE), XGBoost, Plotly/Dash, Jupyter

> 이 문서는 이력서/포트폴리오 제출용으로 프로젝트를 한 페이지로 요약한 것입니다. 전체 내용은
> [`README.md`](../README.md), 실험 상세는 [`model_experiment_log.md`](model_experiment_log.md)를 참고하세요.

## 문제 정의

반도체 제조 공정의 센서 데이터를 이용해 최종 검사 전 불량(Fail) 가능성이 높은 제품을 조기 식별하는 문제를,
공개 SECOM 데이터셋으로 재현했습니다. 불량 비율이 6.64%에 불과한 심각한 클래스 불균형 상황에서 단순
정확도가 아닌 실무적으로 방어 가능한 평가/선택 기준을 세우는 것이 핵심 목표였습니다.

## 접근 방법

1. **데이터 적재**: Kaggle 단일 CSV, UCI 2-파일 포맷을 모두 자동 인식하는 로더 구현 (`src/data_loader.py`).
2. **EDA**: 결측치(최대 91%까지 존재), 상수/저분산 변수(127개), 고상관 변수쌍(329쌍), PCA 2D 투영(설명 분산
   9.2%, 클래스 미분리 — 과장 해석하지 않음)을 체계적으로 확인.
3. **누수 방지 전처리**: 결측치 median 보정, 저분산 제거, 스케일링, SMOTE를 모두 train fold에서만 fit하는
   `imblearn.Pipeline` 설계.
4. **모델 비교**: Baseline(Logistic Regression, 불균형 처리 없음), SMOTE+RF, class_weight+RF, XGBoost(native
   missing handling + scale_pos_weight) 4개 후보를 `RandomizedSearchCV` + `StratifiedKFold`로 비교.
5. **모델 선택**: Recall 단독이 아니라 PR-AUC를 1차 기준으로, F1/Precision/False Positive 부담을 함께 검토.
6. **Threshold Tuning**: 0.05~0.95 전 구간에서 Recall/Precision/F1 트레이드오프를 계산하고, 비즈니스 목적별
   추천 threshold를 제시.
7. **산출물 연계**: Spotfire용 CSV 3종, Plotly 인터랙티브 리포트 4종, Dash 로컬 대시보드까지 확장.

## 결과 (실제 SECOM 데이터, hold-out test 314건)

| 항목 | 값 |
|---|---|
| 최종 선정 모델 | RandomForest_SMOTE (median impute + variance threshold + SMOTE) |
| 선정 기준 | Test PR-AUC 최고 (0.231), CV PR-AUC 0.212±0.046, ROC-AUC 0.829 |
| Threshold=0.5 | Recall 0.048 / Precision 0.167 / F1 0.074 |
| Threshold=0.35 (F1 최대화) | Recall 0.619 / Precision 0.283 / F1 0.388 |

**핵심 인사이트**: 기본 threshold(0.5) 성능만 보면 이 모델은 거의 쓸모없어 보이지만, threshold를
비즈니스 목적에 맞게 조정하면 Recall이 0.05→0.62 수준으로 크게 개선됩니다. 이는 "0.5 고정 정확도"만으로
불균형 분류 모델을 평가하면 안 된다는 것을 실제 데이터로 보여준 사례입니다.

## 산출물 목록

- Jupyter 노트북 2개 (`notebooks/01_eda.ipynb`, `02_modeling.ipynb`) — 실제 SECOM 데이터로 실행 완료
- 재사용 가능한 Python 파이프라인 (`src/`) — 타입 힌트, docstring, pytest 15개 테스트
- Spotfire 연계 CSV 3종, 대시보드 가이드 문서
- Plotly 인터랙티브 HTML 4종 + Dash 로컬 대시보드 (`dashboard/`)
- (부가) 합성 웨이퍼맵 시각화/분류/PDF 리포팅 데모 (`simulations/`) — SECOM과 무관, UI/엔지니어링 역량 시연용

## 면접용 30초 설명

> "공개된 SECOM 반도체 센서 데이터셋으로 불량(Fail) 예측 파이프라인을 만들었습니다. 590개 익명 센서 변수에
> 불량 비율이 6.64%밖에 안 되는 심한 불균형 문제라서, 정확도 대신 PR-AUC·Recall·Precision을 함께 보고
> 모델을 선택했고, SMOTE와 class-weight 두 가지 방식을 같은 Random Forest로 비교했습니다. 결측치 보정·
> 스케일링·SMOTE는 전부 학습 데이터에서만 fit해서 데이터 누수를 막았고, threshold를 하나로 고정하지 않고
> Recall/Precision 트레이드오프 표를 만들었더니 threshold=0.5에서 Recall 0.05였던 최종 모델이 threshold=0.35
> 에서는 Recall 0.62까지 올라간다는 걸 실제로 확인했습니다. 결과는 노트북, 재사용 가능한 파이프라인, Spotfire
> 연계 CSV, Plotly 대시보드까지 이어지도록 구성했습니다."

## 이력서용 1줄 요약

> SECOM 반도체 센서 데이터(590 feature, 불량률 6.64%)를 활용해 데이터 누수 없는 불균형 분류 파이프라인을
> 구축하고, PR-AUC 기반 모델 선택 및 threshold tuning으로 Recall을 0.05→0.62로 개선하는 트레이드오프를
> 정량화함.
