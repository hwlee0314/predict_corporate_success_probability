# 기업 성공 확률 예측 해커톤: 미래의 성공기업을 발굴하라!

### **2025-04-01 ~ 2025.05.30**
### [Competition & Data Link](https://dacon.io/competitions/open/236475/overview/description)
- 다양한 기업 데이터를 기반으로 AI 알고리즘을 개발하여, 기업의 성공 가능성을 예측

- 평가 산식: Weighted MAE  

- 4th out of 592 teams (top 0.7%) 

- 주최: 데이콘
---
### **개발 환경**

- **운영체제**: macos 26.0
- **언어**: Python 3.11

---

### **주요 특징**

- **피처 엔지니어링**: 기업나이, 1인당 투자금, 고객당 연매출 등 비즈니스 도메인 지식을 활용한 17개의 파생 변수 생성으로 기업 성과의 핵심 지표를 수치화했습니다.
- **수치형 문자열 데이터 정규화**: 쉼표가 포함된 금액 데이터와 범위형 데이터(예: "100-200")를 정교하게 파싱하여 평균값으로 변환하는 전처리 파이프라인을 구축했습니다.
- **앙상블 모델링**: Random Forest와 XGBoost의 Soft Voting 앙상블을 통해 각 모델의 장점을 결합하여 예측 안정성과 정확도를 동시에 확보했습니다.
- **피처 선택 최적화**: Random Forest 기반 SelectFromModel을 활용하여 중요도가 높은 피처만을 선별함으로써 모델 복잡도를 줄이고 일반화 성능을 향상시켰습니다.
- **확률 보정 (Calibration)**: Isotonic Regression을 통해 모델 출력값을 실제 확률 분포에 맞게 보정하여 예측 신뢰도를 높였습니다.
- **투자 단계 순서 인코딩**: Seed → Series A → Series B → Series C → IPO 순서로 투자 단계의 순서 정보를 보존하는 레벨 인코딩을 적용했습니다.

---



### **프로젝트 구조**

```
predict_corporate_success_probability/
├── Feature Engineering + RF + XGB Soft Voting.ipynb          
├── fiinal_submissionn.csv
├── README.md           
└── requirements.txt    
```

---

### **모델링 접근법**

기업의 다양한 정량적·정성적 데이터를 활용하여 성공 확률을 예측하는 회귀 모델을 구축했습니다. Random Forest와 XGBoost의 앙상블을 통해 안정적이고 정확한 예측을 달성했습니다.

***

### 1. 데이터 전처리 파이프라인
- **수치형 문자열 정규화**: 쉼표가 포함된 금액 데이터와 범위형 데이터("100-200") 파싱 후 평균값 변환
- **결측값 전략적 처리**: 분야는 최빈값, 수치형 변수는 중앙값으로 대체하여 도메인 특성 반영
- **범주형 변수 인코딩**: 국가, 분야, 투자단계 등을 Label Encoding으로 수치화
- **투자단계 순서 보존**: Seed(0) → Series A(1) → ... → IPO(4) 순서 정보를 유지하는 레벨 인코딩

### 2. 파생변수 생성
- **효율성 지표**: 1인당 투자금, 1인당 연매출, 고객당 연매출, 고객당 기업가치
- **수익성 지표**: 투자금 대비 연매출, 매출 대비 기업가치 
- **성장성 지표**: 기업나이 (2025 - 설립연도)
- **성공 신호**: 상장여부, 인수여부를 이진 플래그로 변환 및 통합 점수 생성
- **투자 성숙도**: 투자단계의 순서 정보를 활용한 성숙도 레벨링

### 3. 앙상블 모델링 전략
- **Random Forest Regressor**: 안정적인 기본 성능과 피처 중요도 기반 선택을 위한 베이스 모델
- **XGBoost Regressor**: 그래디언트 부스팅을 통한 세밀한 패턴 학습과 높은 예측 정확도
- **Soft Voting Ensemble**: 두 모델의 예측값을 평균하여 각각의 장점을 결합
- **피처 선택**: Random Forest 기반 SelectFromModel로 중요 피처만 선별하여 과적합 방지

### 4. 모델 최적화 및 보정
- **하이퍼파라미터 최적화**: Optuna를 통한 베이지안 최적화로 각 모델의 최적 파라미터 탐색
- **Isotonic Regression**: 모델 출력을 실제 확률 분포에 맞게 보정하여 신뢰도 향상
- **교차 검증**: K-Fold 교차 검증을 통한 모델 성능 안정성 확보

---

### **파일 설명**

#### `Feature Engineering + RF + XGB Soft Voting.ipynb`
- **데이터 로딩**: train.csv, test.csv, sample_submission.csv 불러오기
- **수치형 문자열 전처리**:
  - 쉼표 제거 및 범위형 데이터("100-200") 평균값 변환
  - 기업가치, 총 투자금, 연매출 컬럼 정규화
- **결측값 처리**:
  - 분야: 최빈값으로 대체
  - 직원 수, 고객수, 기업가치: 중앙값으로 대체
- **피처 엔지니어링** (17개 파생 변수):
  - `기업나이`: 2025 - 설립연도
  - `1인당_투자금`: 총 투자금 / 직원 수
  - `1인당_연매출`: 연매출 / 직원 수
  - `고객당_연매출`: 연매출 / 고객수
  - `투자금_대비_연매출`: 총 투자금 / 연매출
  - `매출_대비_기업가치`: 기업가치 / 연매출
  - `고객당_기업가치`: 기업가치 / 고객수
  - `상장여부_flag`, `인수여부_flag`, `flag_sum`: 이진 플래그 변수
  - `투자단계_level`: Seed(0) → Series A(1) → ... → IPO(4) 순서 인코딩
- **Label Encoding**: 국가, 분야, 투자단계, 인수여부, 상장여부 범주형 변수 인코딩
- **피처 선택**: Random Forest 기반 SelectFromModel로 중요 피처 선별
- **앙상블 모델링**:
  - XGBoost Regressor (최적화된 하이퍼파라미터)
  - Random Forest Regressor (최적화된 하이퍼파라미터)
  - Voting Regressor로 두 모델 결합
- **확률 보정**: Isotonic Regression으로 예측값 보정
- **예측 및 저장**: `fiinal_submissionn.csv` 생성

#### `data/`
- `train.csv`: 기업 학습 데이터 (ID, 국가, 분야, 설립연도, 직원수, 고객수, 기업가치, 투자금, 연매출, 투자단계, 인수여부, 상장여부, 성공확률)
- `test.csv`: 기업 테스트 데이터 (성공확률 제외)
- `sample_submission.csv`: 제출 파일 템플릿 (ID, 성공확률)

#### `fiinal_submissionn.csv`
- Feature Engineering + RF + XGB Soft Voting 모델의 최종 예측 결과 파일


