# 데이터 다운로드 및 배치 안내

이 저장소는 데이터 원본을 포함하지 않습니다 (`data/raw/`는 `.gitignore`에 의해 Git 추적에서 제외됩니다).
아래 방법 중 하나로 SECOM 데이터셋을 받아 `data/raw/` 아래에 배치한 뒤 파이프라인을 실행하세요.

## 방법 1: Kaggle (`kagglehub`, 권장)

```python
import kagglehub

path = kagglehub.dataset_download("paresh2047/uci-semcom")
print("Path to dataset files:", path)
```

다운로드된 폴더 안의 `uci-secom.csv` 파일을 프로젝트의 `data/raw/uci-secom.csv`로 복사합니다.

```bash
cp "<kagglehub 다운로드 경로>/uci-secom.csv" data/raw/uci-secom.csv
```

이 파일은 헤더 행 하나에 `Time`, 센서 컬럼(0~589), `Pass/Fail` 라벨을 모두 포함한 단일 CSV입니다.
`src/data_loader.py`가 이 레이아웃을 자동으로 인식합니다.

## 방법 2: UCI 원본 (2-파일 포맷)

[UCI Machine Learning Repository - SECOM](https://archive.ics.uci.edu/dataset/179/secom)에서
`secom.data`, `secom_labels.data`를 내려받아 다음 위치에 배치합니다.

```
data/raw/secom.data
data/raw/secom_labels.data
```

- `secom.data`: 공백으로 구분된 1567행 x 590열 센서 값 (결측치는 `NaN` 문자열)
- `secom_labels.data`: 각 행이 `<label>\t<date> <time>` 형식 (label: `-1`=정상, `1`=불량)

## 배치 후 확인

```bash
python -c "
import sys; sys.path.insert(0, '.')
from src.data_loader import load_raw_secom, get_feature_columns
from src import config
df = load_raw_secom(config.DATA_RAW_DIR)
print(df.shape)
print(df[config.LABEL_TEXT_COL].value_counts())
"
```

`rows=1567`, feature 컬럼 590개, `Pass`/`Fail` 라벨 분포가 출력되면 정상적으로 배치된 것입니다.

## 데이터 한계 (중요)

- 이 데이터셋은 **익명화(anonymized)된 공개 데이터**이며, 590개 센서 feature는 실제 어떤 장비/챔버/공정 단계에
  대응하는지 공개되어 있지 않습니다.
- 따라서 이 프로젝트의 모든 분석과 결과는 **실제 Fab 공정의 원인을 규명한 것이 아니라, 공개 데이터에서
  불량 판별에 통계적으로 기여하는 센서 패턴을 모델링한 가상 시뮬레이션**입니다.
- Wafer X/Y 좌표, 장비 ID, Chamber ID, Lot ID 등 실제 공정 추적에 필요한 정보는 이 데이터셋에 없습니다.
