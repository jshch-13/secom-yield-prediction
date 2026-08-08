# Spotfire 대시보드 제작 가이드

이 문서는 `python src/train.py` (또는 `notebooks/02_modeling.ipynb`) 실행 후 `data/processed/`에 생성되는
CSV 파일을 TIBCO Spotfire에서 불러와 대시보드를 구성하는 방법을 설명합니다. Spotfire `.dxp` 파일 자체는
이 저장소에 포함되어 있지 않으며, 아래 절차를 따라 사용자의 Spotfire 환경에서 직접 구성해야 합니다.

아래는 이 가이드를 따라 실제로 구성한 결과 화면입니다 (KPI 카드, Feature Importance, Pass/Fail별 센서 분포,
Scatter Plot, Fail Probability 히스토그램).

<img src="spotfire_example.png" alt="Spotfire dashboard example" width="800">

## 0. 데이터 및 분석의 한계 (반드시 먼저 읽어주세요)

- 이 대시보드는 공개·익명화된 SECOM 데이터셋 기반이며, 실제 반도체 Fab의 특정 장비/공정을 지칭하지 않습니다.
- **SECOM 데이터셋에는 Wafer X/Y 좌표, 장비 ID(Equipment ID), Chamber ID, Lot ID가 없습니다.**
  따라서 아래와 같은 분석/시각화는 만들지 마세요 (데이터에 없는 정보를 있는 것처럼 오해하게 만듭니다).
  - Wafer map (다이별 양불 지도)
  - Chamber traceback / 챔버별 불량 추적
  - 장비(Equipment) 간 비교 분석
  - Lot 단위 이력 추적
- feature importance는 익명화된 feature ID 기준의 통계적 중요도 순위이며, 실제 공정 인과관계로 해석하지 마세요.

## 1. 데이터 로드 방법

`data/processed/`에는 3개의 CSV가 생성됩니다.

| 파일 | 용도 |
|---|---|
| `secom_spotfire_master.csv` | 대시보드의 중심이 되는 단일 마스터 테이블 (행 단위 예측 결과) |
| `secom_feature_summary.csv` | Feature 단위 요약 (결측률/분산/중요도) |
| `secom_model_performance.csv` | 후보 모델 간 성능 비교 |

Spotfire에서:

1. `File > Add Data Tables...` 로 세 파일을 각각 불러옵니다.
2. Import 시 컬럼 타입을 확인합니다. `Time`은 Date/Time, `Pass_Fail`/`predicted_class`는 Integer(또는 카테고리로 다시 캐스팅),
   `fail_probability`/`prediction_threshold`는 Real(숫자)로 지정합니다.
3. 세 테이블은 관계(relation)를 설정할 필요가 없습니다. `secom_spotfire_master.csv`를 메인 데이터 테이블로 사용하고,
   `secom_feature_summary.csv`와 `secom_model_performance.csv`는 각각 별도의 시각화(피처 중요도 바 차트, 모델 비교 표)에서
   독립적으로 사용하면 충분합니다. 굳이 Data Relationship을 만들 필요가 없는 단순한 구조입니다.

## 2. `secom_spotfire_master.csv` 컬럼 설명

| 컬럼 | 설명 |
|---|---|
| `row_id` | 원본 데이터의 행 순번 |
| `Time` | 원본 타임스탬프 (원본 형태 보존) |
| `Pass_Fail` | 실제 라벨 (0=Pass, 1=Fail) |
| `Pass_Fail_Label` | 실제 라벨 텍스트 (Pass/Fail) |
| `dataset_split` | `train` 또는 `test`. 성능을 판단할 때는 반드시 `test`로 필터링하세요. `train` 행의 예측은 학습에 사용된 데이터에 대한 것이라 낙관적으로 나타납니다. |
| `model_used` | 예측에 사용된 최종 선정 모델 이름 |
| `fail_probability` | 모델이 예측한 불량 확률 (0~1) |
| `prediction_threshold` | 예측 클래스 판정에 사용된 임계값 (기본 0.5) |
| `predicted_class` | 예측 클래스 (0/1) |
| `model_prediction` | 예측 클래스 텍스트 (Pass/Fail) |
| `feature_XXX` (다수) | 중요도 상위 feature들의 원본 센서 값 (전체 590개가 아닌 상위 N개만 포함되어 있습니다) |

## 3. 권장 KPI 카드

Spotfire의 KPI Chart(또는 텍스트 영역 + 계산식)로 아래 지표를 상단에 배치합니다. `dataset_split = "test"`로
필터링한 뒤 계산하세요.

- 전체 샘플 수: `Count([row_id])`
- 불량 수: `Sum(If([Pass_Fail]=1, 1, 0))`
- 불량률: `Sum(If([Pass_Fail]=1, 1, 0)) / Count([row_id])`
- Recall / Precision / PR-AUC: `secom_model_performance.csv`의 최종 선정 모델 행 값을 텍스트 영역에 표시합니다.
  Spotfire 계산식으로 재계산하지 말고 `src/train.py`가 계산한 값을 그대로 인용하세요. Spotfire 내부 재계산은
  threshold 정의가 어긋나 값이 달라질 수 있습니다.

## 4. 권장 시각화

1. Bar Chart, Feature Importance Top 20
   - 데이터: `secom_feature_summary.csv`
   - X축: `feature_name`, Y축: `importance`, `rank <= 20`으로 필터링, `rank` 오름차순 정렬.
   - 부제목에 "익명화된 feature ID 기준 통계적 중요도"라고 명시하세요.

2. Box Plot, Pass/Fail별 상위 중요 센서 분포
   - 데이터: `secom_spotfire_master.csv` (`dataset_split = "test"` 권장, 분포 확인 목적이면 전체 사용도 가능)
   - X축: `Pass_Fail_Label`, Y축: 중요도 1~2위 `feature_XXX` 컬럼, Trellis by feature로 여러 센서를 나란히 비교.

3. Scatter Plot, 상위 2개 중요 센서
   - X축/Y축: 중요도 1위, 2위 `feature_XXX`
   - Color by: `Pass_Fail_Label`
   - Size by: `fail_probability`
   - `dataset_split = "test"`로 필터링하여 실제 일반화 성능 맥락에서 보세요.

4. Histogram, Fail Probability 분포
   - 데이터: `fail_probability` (`dataset_split = "test"`)
   - Color by: `Pass_Fail_Label`로 나누면 정상/불량 그룹의 확률 분포 차이를 볼 수 있습니다.
   - `prediction_threshold` 위치에 참조선(Reference Line)을 추가하면 임계값의 의미가 명확해집니다.

5. Table, 오분류 유형(FP/FN) 필터링
   - 데이터: `secom_spotfire_master.csv` (`dataset_split = "test"`)
   - 계산 컬럼 추가 (Spotfire Calculated Column):
     ```
     Case
       When [Pass_Fail]=0 and [predicted_class]=1 Then 'False Positive'
       When [Pass_Fail]=1 and [predicted_class]=0 Then 'False Negative'
       When [Pass_Fail]=1 and [predicted_class]=1 Then 'True Positive'
       Else 'True Negative'
     End
     ```
   - 이 컬럼으로 필터링/색상 구분하여 오분류 사례만 따로 조회할 수 있는 표를 만듭니다.

6. Cross Table, Threshold별 Recall/Precision/F1 비교
   - `outputs/metrics/threshold_tuning_<model_name>.csv`를 추가로 불러와 사용합니다.
   - 행: `threshold`, 값: `recall`, `precision`, `f1`을 Cross Table로 배치하면 임계값에 따른 트레이드오프를 한눈에 볼 수 있습니다.

## 5. 대시보드 구성 팁

- 상단: KPI 카드 (전체 샘플/불량 수/불량률/Recall/Precision/PR-AUC)
- 중단 좌: Feature Importance Bar Chart, 중단 우: Box Plot
- 하단 좌: Scatter Plot, 하단 우: Fail Probability Histogram
- 별도 탭: 오분류 Table, Threshold Cross Table
- 모든 페이지 상단에 "익명화된 공개 데이터 기반 시뮬레이션" 배지를 텍스트 영역으로 고정 배치할 것을 권장합니다.
