# 모델 실험 로그 (Model Experiment Log)

이 문서는 `python src/train.py` 실행(실제 SECOM 데이터, `random_state=42`)으로 생성된
`outputs/metrics/model_comparison.csv`, `outputs/metrics/final_metrics.json`을 기준으로 작성되었습니다.
숫자를 임의로 추정하지 않고, 실행 결과를 그대로 기록합니다.

## 실험 설정

- 데이터: SECOM 1,567건, 590개 익명 센서 feature, 불량률 6.64% (Fail 104 / Pass 1,463)
- 분할: Stratified train/test = 1,253 / 314 (train fail rate 6.62%, test fail rate 6.69%)
- 교차검증: `StratifiedKFold(n_splits=5)` + `RandomizedSearchCV`(작은 탐색 공간, `n_iter=20` 상한)
- 1차 선택 기준: Hold-out test PR-AUC (다른 지표는 함께 검토용으로 기록)

## 후보 모델 비교 (Hold-out Test, threshold=0.5)

| Model | Preprocessing | Sampling/Weighting | Recall | Precision | F1 | PR-AUC | ROC-AUC | FP | FN |
|---|---|---|---|---|---|---|---|---|---|
| Baseline_LogisticRegression | median_impute + variance_threshold + scale | none | 0.143 | 0.273 | 0.188 | 0.129 | 0.640 | 8 | 18 |
| RandomForest_SMOTE | median_impute + variance_threshold | SMOTE (train fold only) | 0.048 | 0.167 | 0.074 | **0.231** | 0.829 | 5 | 20 |
| RandomForest_ClassWeight | median_impute + variance_threshold | class_weight=balanced | 0.000 | 0.000 | 0.000 | 0.220 | 0.785 | 1 | 21 |
| XGBoost_ScalePosWeight | variance_threshold only (native missing handling) | scale_pos_weight | 0.000 | 0.000 | 0.000 | 0.189 | 0.691 | 3 | 21 |

## Stratified 5-Fold CV 요약 (train, PR-AUC 기준)

| Model | CV PR-AUC (mean ± std) | CV Recall (mean ± std) |
|---|---|---|
| Baseline_LogisticRegression | 0.161 ± 0.066 | 0.120 ± 0.085 |
| RandomForest_SMOTE | 0.212 ± 0.046 | 0.108 ± 0.044 |
| RandomForest_ClassWeight | 0.214 ± 0.065 | 0.000 ± 0.000 |
| XGBoost_ScalePosWeight | 0.201 ± 0.052 | 0.049 ± 0.046 |

CV 결과와 test 결과의 PR-AUC 순위가 완전히 일치하지는 않습니다 (RandomForest_ClassWeight의 CV PR-AUC가
근소하게 더 높음). 이는 314건에 불과한 test set 크기, 그리고 21건뿐인 test Fail 샘플 수에서 오는 분산이
클 수밖에 없다는 점을 보여주며, 실무에서는 이 정도 표본 크기에서의 순위 차이를 과신하지 않아야 함을
시사합니다.

## 최종 모델 선정: RandomForest_SMOTE

- 선정 근거: Hold-out test PR-AUC 0.231로 4개 후보 중 최고. CV PR-AUC(0.212)도 Baseline 대비 뚜렷하게
  높음. ROC-AUC(0.829)도 가장 높아, threshold에 무관하게 Pass/Fail을 분리하는 능력이 상대적으로 우수함을
  시사.
- 주의할 점: threshold=0.5에서의 Recall(0.048)만 보면 이 모델이 "거의 불량을 못 잡는" 모델처럼 보이지만,
  이는 SMOTE로 학습된 모델의 예측 확률 분포가 0.5보다 낮은 쪽에 몰려 있기 때문입니다. Threshold를 0.35로
  낮추면 Recall이 0.619까지 상승합니다 (아래 threshold tuning 표 참고). 이것이 바로 "0.5 고정 정확도만으로
  판단하지 않는다"는 이 프로젝트의 핵심 원칙이 실제로 드러난 지점입니다.

## SMOTE vs class_weight 비교 (동일 모델: Random Forest)

| 방식 | Test PR-AUC | CV PR-AUC | Test Recall @0.5 |
|---|---|---|---|
| SMOTE (train fold only) | 0.231 | 0.212 ± 0.046 | 0.048 |
| class_weight='balanced' | 0.220 | 0.214 ± 0.065 | 0.000 |

두 방식의 PR-AUC 차이는 크지 않으며(0.231 vs 0.220), SMOTE가 test set 기준으로는 근소하게 우위였습니다.
CV 기준으로는 오히려 class_weight가 근소하게 높아, 이 정도 표본 크기에서는 두 방식이 사실상 대등하다고
해석하는 것이 더 안전합니다. 어느 한쪽이 확실한 승자라고 결론짓지 않았습니다.

## Threshold Tuning 결과 (최종 모델: RandomForest_SMOTE, test set)

| 전략 | Threshold | Recall | Precision | F1 | FP | FN |
|---|---|---|---|---|---|---|
| F1 최대화 | 0.35 | 0.619 | 0.283 | 0.388 | 33 | 8 |
| Recall 우선 (Precision ≥ 0.2 유지) | 0.30 | 0.810 | 0.210 | 0.333 | 64 | 4 |
| Precision 우선 (Recall ≥ 0.5 유지) | 0.35 | 0.619 | 0.283 | 0.388 | 33 | 8 |

전체 threshold(0.05~0.95) 스윕 결과는 `outputs/metrics/threshold_tuning_RandomForest_SMOTE.csv`와
`outputs/figures/threshold_tuning_RandomForest_SMOTE.png`에 있습니다.

## Feature Importance

- 방법: `permutation_importance` (scoring=`average_precision`), SHAP이 설치되지 않은 환경에서는 자동으로
  이 방식으로 대체됩니다. (`outputs/metrics/final_metrics.json`의 `importance_method` 필드로 실행 시점에
  실제 사용된 방법을 확인할 수 있습니다.)
- 상위 변수: `feature_059`, `feature_247`, `feature_487`, `feature_112`, `feature_385` 등
  (`outputs/metrics/feature_importance.csv` 전체 참고).
- 해석 주의: 위 순위는 익명화된 feature ID에 대한 통계적 중요도이며, 실제 공정/장비의 물리적
  인과관계를 의미하지 않습니다.

## 한계 및 재현성 관련 메모

- Test Fail 샘플이 21건에 불과해, 지표(특히 Recall/Precision)의 표본 분산이 큽니다. 이 프로젝트의 결론은
  "이 정도 규모의 공개 데이터에서 관찰된 경향"으로 한정해서 해석해야 합니다.
- `random_state=42`로 고정되어 있어 동일한 raw 데이터·동일한 라이브러리 버전(`requirements.txt`)에서는
  동일한 수치가 재현됩니다.
