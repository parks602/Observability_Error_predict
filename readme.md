# 시스템 리소스 예측 시스템

서버의 CPU, Memory, Disk 사용률을 실시간으로 모니터링하고, 60분 후의 리소스 사용량을 예측하여 장애를 사전에 방지하는 시스템입니다.

## 목차

- [개요](#개요)
- [주요 기능](#주요-기능)
- [시스템 구조](#시스템-구조)
- [설치 방법](#설치-방법)
- [사용 방법](#사용-방법)
- [설정](#설정)
- [디렉토리 구조](#디렉토리-구조)
- [트러블슈팅](#트러블슈팅)

## 개요

이 시스템은 Prometheus에서 수집한 메트릭 데이터를 기반으로 Random Forest 모델을 학습하여 시스템 리소스 사용률을 예측합니다. 임계치를 초과할 것으로 예측되면 Dooray를 통해 알림을 전송하고, MariaDB에 예측 결과를 저장합니다.

### 주요 특징

- **실시간 모니터링**: 1분 간격으로 자동 실행
- **사전 예측**: 60분 후의 리소스 사용률 예측
- **자동 알림**: 임계치 초과 예상 시 즉시 알림
- **자동 학습**: 주기적으로 모델 재학습
- **멀티 리소스**: CPU, Memory, Disk 통합 모니터링

## 주요 기능

### 1. 데이터 수집
- Prometheus에서 시스템 메트릭 수집
- 1분 단위 시계열 데이터 저장
- 자동 데이터 업데이트

### 2. 특성 엔지니어링
- 시간 기반 특성 (시간대, 요일, 업무시간)
- Lag 특성 (1분, 5분, 10분, 15분, 30분, 45분, 60분)
- 롤링 통계 (평균, 표준편차, 최대값, 최소값)
- 변화율 특성 (1분, 10분, 30분, 60분)
- 비율 및 추세 특성

### 3. 모델 학습
- Random Forest Classifier 사용
- 30일간의 히스토리 데이터로 학습
- 자동 임계값 최적화 (Recall 0.8 이상)
- 클래스 불균형 처리

### 4. 실시간 예측
- 최근 61분 데이터로 예측
- 확률 기반 임계치 판단
- 60분 후 리소스 상태 예측

### 5. 알림 및 저장
- Dooray 웹훅으로 알림 전송
- MariaDB에 예측 결과 저장
- 서버 정보 및 임계값 포함

## 시스템 구조

```
observability/
├── main.py                     # 메인 실행 파일
├── run.sh                      # 자동 실행 스크립트 (Linux/Mac)
├── run.bat                     # 자동 실행 스크립트 (Windows)
├── config/
│   ├── config.yaml            # 시스템 설정
│   └── server.yaml            # 서버 설정
├── src/
│   ├── core/
│   │   ├── runner.py          # 학습/예측 실행 로직
│   │   └── rf_model.py        # Random Forest 모델
│   ├── data/
│   │   ├── collector.py       # 데이터 수집
│   │   ├── query_handler.py   # Prometheus 쿼리 및 특성 생성
│   │   └── prometeus_collector.py
│   ├── database/
│   │   └── connector.py       # MariaDB 연결
│   ├── alert/
│   │   └── dooray.py         # Dooray 알림
│   └── utils/
│       ├── config_manager.py
│       └── checker.py
├── data/                       # 데이터 저장소
│   ├── train/                 # 학습 데이터
│   └── predict/               # 예측 데이터
├── models/                     # 학습된 모델
└── logs/                       # 로그 파일
```

## 설치 방법

### 1. 저장소 클론

```bash
git clone <repository-url>
cd observability
```

### 2. Python 가상환경 설정

```bash
# 가상환경 생성
python -m venv venv

# 가상환경 활성화
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

### 3. 패키지 설치

```bash
pip install -r requirements.txt
```

### 4. 설정 파일 구성

`config/config.yaml` 파일을 작성합니다:

```yaml
prometheus:
  url: "http://prometheus-server:9090"

database:
  host: "localhost"
  port: 3306
  user: "user"
  password: "password"
  database: "dbname"

models:
  cpu:
    model_path: "./models/cpu"
    threshold: 85
  memory:
    model_path: "./models/memory"
    threshold: 85
  disk:
    model_path: "./models/disk"
    threshold: 90

alert:
  dooray:
    webhook_url: "https://hook.dooray.com/..."

server_info:
  dir: "./server_info"
```

## 사용 방법

### 단일 실행

특정 리소스에 대해 한 번만 실행:

```bash
# CPU 예측
python main.py --object cpu

# Memory 예측
python main.py --object memory

# Disk 예측
python main.py --object disk
```

### 자동 실행 (1분 간격)

#### Linux/Mac

```bash
# 실행 권한 부여
chmod +x run.sh

# 실행
./run.sh
```

#### Windows

```cmd
run.bat
```

#### 중지

```
Ctrl + C
```

## 설정

### 임계값 설정

`config/config.yaml`에서 각 리소스별 임계값을 설정할 수 있습니다:

```yaml
models:
  cpu:
    threshold: 85      # CPU 85% 이상 예측 시 알림
  memory:
    threshold: 85      # Memory 85% 이상 예측 시 알림
  disk:
    threshold: 90      # Disk 90% 이상 예측 시 알림
```

### 모델 재학습 주기

모델은 다음 조건에서 자동으로 재학습됩니다:
- 모델이 존재하지 않는 경우
- 모델은 월 단위 업데이트 됩니다. 모델 경로 위치는 각 월별로 나뉘기에 모델이 존재하지 않게 됩니다.
- 마지막 학습 후 일정 기간이 지난 경우 (checker.py에서 설정)

## 디렉토리 구조

### 데이터 디렉토리

```
data/
├── train/
│   ├── cpu/
│   │   └── CODE_SVCTYPE_SVRTYPE_YYYY_MM_data.csv
│   ├── memory/
│   └── disk/
└── predict/
    ├── cpu/
    │   └── CODE_SVCTYPE_SVRTYPE_INSTANCE_data.csv
    ├── memory/
    └── disk/
```

### 모델 디렉토리

```
models/
├── cpu/
│   └── YYYY/
│       └── MM/
│           ├── CODE_SVCTYPE_SVRTYPE.pkl
│           └── CODE_SVCTYPE_SVRTYPE_config.json
├── memory/
└── disk/
```

## 동작 원리

### 1. 학습 단계

1. Prometheus에서 30일간의 메트릭 데이터 수집
2. 시계열 특성 생성 (Lag, Rolling, Change 등)
3. Random Forest 모델 학습
4. 최적 임계값 탐색 (F1-score 최대화, Recall >= 0.8)
5. 모델 및 설정 저장

### 2. 예측 단계

1. 최근 61분간의 메트릭 데이터 수집
2. 동일한 특성 생성 과정 적용
3. 학습된 모델로 60분 후 예측
4. 임계값 초과 여부 판단
5. 알림 전송 및 DB 저장

### 3. 특성 생성 예시 (CPU)

```python
# 시간 특성
- hour: 시간대 (0-23)
- day_of_week: 요일 (0-6)
- is_weekend: 주말 여부
- is_business_hour: 업무시간 여부 (9-18시)

# Lag 특성
- cpu_util_lag_1m: 1분 전 사용률
- cpu_util_lag_5m: 5분 전 사용률
- cpu_util_lag_60m: 60분 전 사용률

# 롤링 통계
- cpu_util_mean_30m: 30분 평균
- cpu_util_std_30m: 30분 표준편차
- cpu_util_max_30m: 30분 최대값

# 변화율
- cpu_util_change_1m: 1분간 변화량
- cpu_util_change_30m: 30분간 변화량

# 기타
- load_ratio_1m_5m: Load Average 비율
- cpu_user_system_ratio: User/System 비율
```

## 트러블슈팅

### 1. 인코딩 에러 (Windows)

**문제**: `UnicodeEncodeError: 'cp949' codec can't encode character`

**해결**: `main.py` 상단에 다음 코드 추가

```python
import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
```

### 2. 데이터 수집 실패

**문제**: Prometheus 연결 실패

**해결**:
- Prometheus URL 확인 (`config/config.yaml`)
- 네트워크 연결 확인
- Prometheus 서버 상태 확인

### 3. 모델 학습 실패

**문제**: `ValueError: Insufficient anomaly samples`

**해결**:
- 충분한 데이터가 수집되었는지 확인 (최소 30일)
- 임계값 초과 데이터가 있는지 확인
- 로그 파일 확인

### 4. 예측 실패 (NaN 문제)

**문제**: 데이터에 NaN 값이 많아 예측 불가

**해결**:
- 데이터 수집 시간 확인 (최소 61분 필요)
- Prometheus 메트릭 수집 상태 확인
- 로그에서 데이터 수집 실패 확인

### 5. 데이터베이스 연결 실패

**문제**: MariaDB 연결 오류

**해결**:
- DB 연결 정보 확인 (`config/config.yaml`)
- DB 서버 상태 확인
- 방화벽 설정 확인


## 성능 튜닝

### 모델 파라미터 조정

`src/core/rf_model.py`에서 Random Forest 파라미터 조정:

```python
self.model = RandomForestClassifier(
    n_estimators=400,      # 트리 개수
    max_depth=20,          # 최대 깊이
    min_samples_split=3,   # 분할 최소 샘플
    min_samples_leaf=3,    # 리프 최소 샘플
    class_weight=class_weight,
    random_state=42,
    n_jobs=-1,
)
```

### 임계값 조정

더 민감한 알림이 필요한 경우 `find_optimal_threshold` 함수의 파라미터 조정:

```python
self.best_threshold, _ = self.find_optimal_threshold(
    y_test, y_proba_test, 
    min_recall=0.8,  # 최소 Recall (높일수록 더 많은 알림)
    min_f1=0.3       # 최소 F1-score
)
```
