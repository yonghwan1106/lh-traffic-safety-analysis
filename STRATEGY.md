# LH 3기신도시 교통안전 데이터 분석 -- 최우수상 전략서

## 대회 정보
- **대회명**: (LH) 신도시 교통 인프라 데이터 분석을 통한 3기신도시 어린이 등 취약계층 교통안전 대안제시
- **주최**: LH 한국토지주택공사 (COMPAS 플랫폼)
- **상금**: 최우수 500만원, 우수 300만원, 장려 200만원 x 2
- **마감**: 2026.3.20(금) 18:00
- **참가팀**: 45팀

---

## 1. 차별화 전략: "45팀 중 1등이 되려면"

### 1.1 핵심 차별화 컨셉: "교통안전 취약성 지수(TSVI: Traffic Safety Vulnerability Index)"

대부분의 팀이 할 것:
- 단순 사고 통계 분석 → 위험 지역 표시 → "여기에 시설 설치하세요"
- 기존 신도시 분석 → 하남교산에 단순 대입

우리가 할 것:
- **100m x 100m 격자 단위의 다차원 취약성 지수(TSVI)** 개발
- 인구 취약성 + 도로 위험성 + 시설 부재성 + 시간대별 위험도를 결합한 복합 지표
- 기존 4개 신도시에서 검증 → 하남교산에 예측 적용
- **"사고가 이미 난 곳"이 아니라 "사고가 날 수밖에 없는 구조적 조건"을 찾는 것**

### 1.2 분석 프레임워크명: "SAFE-Grid Framework"

```
S - Spatial Risk Mapping (공간적 위험도 매핑)
A - Age-group Vulnerability Analysis (연령별 취약성 분석)
F - Facility Gap Detection (시설 공백 탐지)
E - Effect Estimation & Evidence-based Recommendation (효과 추정 & 근거 기반 제안)
```

이 프레임워크를 일관되게 사용하면:
- 보고서 구조가 명확해짐
- 발표 시 스토리텔링이 자연스러움
- 심사위원이 체계적이라고 평가함

### 1.3 왜 이것이 이기는가

| 일반적 접근 | 우리의 접근 |
|------------|-----------|
| 사고 다발 지역 분석 | 사고 발생 "구조적 조건" 모델링 |
| 기술 통계 중심 | 공간통계 + 머신러닝 + 인과추론 |
| 시설 위치 추천 | 격자 단위 최적 배치 + 비용편익 분석 |
| 기존 도시 분석만 | 기존 도시 모델 → 하남교산 예측 전이 |
| 단일 취약계층 | 어린이/고령자/임산부 맞춤 다계층 분석 |

### 1.4 학술적 기반: 3대 통계 축 (핵심 차별화)

우리의 분석은 세 가지 학술적 축으로 구성된다:

**축 1: 카운트 모형 + Empirical Bayes (EB)**
- 교통사고는 포아송/음이항(NB) 분포를 따르는 카운트 데이터 [A.1]
- NB-GLM Safety Performance Function(SPF)으로 기대 사고건수 추정
- EB 보정으로 소표본 격자의 극단값을 전체 평균 방향으로 수축(shrinkage)
- 핵심 수식: `EB_i = w_i * y_i + (1 - w_i) * mu_hat_i`, `w_i = 1/(1 + mu_hat_i/k_hat)`
- **차별화 포인트**: 대부분의 팀은 원시 사고건수를 직접 사용 → 우리는 EB 보정으로 통계적 노이즈 제거

**축 2: 다중스케일 공간분석 (MGWR)**
- GWR(지리가중회귀)의 확장: 변수마다 다른 bandwidth 허용 [A.2]
- 예: 교통량 영향 = 넓은 bandwidth(도시 수준), 학교 근접성 = 좁은 bandwidth(초지역)
- 핵심 수식: `y_i = beta_0(u_i,v_i) + sum_k beta_k(u_i,v_i) * x_ik + e_i` (변수별 bandwidth b_k)
- **차별화 포인트**: GWR만 사용하는 팀 대비 변수별 공간 스케일 차이를 포착

**축 3: 수리최적화 (ILP)**
- 시설 배치를 직관이 아닌 수학적 최적화로 해결 [A.4]
- p-Median: 수요 가중 거리 최소화
- Set Covering: 모든 고위험 격자 커버 최소 시설
- Max Coverage: 예산 제약 하 TSVI 감소 극대화
- **차별화 포인트**: "이 격자에 설치하세요" → "이 예산으로 이렇게 배치하면 최적"

---

## 2. 단계별 분석 파이프라인

### Phase 1: 데이터 수집 및 전처리 (Week 1-2)

#### Step 1.1: 공간 데이터 통합 기반 구축
```
[목표] 100m x 100m 격자를 기본 분석 단위로 설정하고 모든 데이터를 격자에 매핑

작업:
1. 5개 지역(동탄1·2, 위례, 하남미사, 판교, 하남교산) 격자 데이터 로드
2. 좌표계 통일: EPSG:4326 (WGS84)
3. 격자별 고유 ID 체계 구축 (지역코드_격자ID)
4. 모든 포인트/폴리곤 데이터를 격자에 Spatial Join
```

#### Step 1.2: 포인트 데이터 격자 매핑
```python
# 격자에 매핑할 데이터 목록:
- 어린이보호구역 → grid_child_protection_zone (boolean + 거리)
- 학교/유치원/어린이집 → grid_edu_facilities_count
- 횡단보도 → grid_crosswalk_count
- 버스정류장 → grid_busstop_count
- CCTV → grid_cctv_count
- 과속방지턱 → grid_speedbump_count
- 교통사고 이력 → grid_accident_count (유형별, 시간대별)
```

#### Step 1.3: 인구 데이터 통합 및 노출 변수(exposure) 산출
```python
# 상주인구 (File 03: 공개)
- 격자별 연령대별/성별 상주인구
- 고령인구 비율: (m_70g + m_80g + m_90g + m_100g + w_70g~w_100g) / 전체
- 가임기 여성 비율: (w_20g + w_30g) / 전체
- 주의: 0-19세 데이터 없음 → 유치원/어린이집 용량으로 아동인구 프록시 추정

# 유동인구 (File 04: 비공개 - COMPAS 노트북)
- 월별/시간대별 유동인구 밀도 (인/km²)

# 직장인구/방문인구 (Files 05/06: 비공개)
- TMST_00 ~ TMST_23: 24시간 시간대별 인구
- 시간대 변동계수(CV) = std / mean → TRI 핵심 입력
- 피크/비피크 비율 산출

# 서비스인구 (File 07: 비공개)
- hw(평일/주말), w_pop(직장), v_pop(방문)
```

#### Step 1.4: 도로 네트워크 데이터 통합
```python
# File 08: 도로 네트워크 (비공개)
- link_id, max_speed, road_rank, lanes, oneway, length

# File 09: 도로 속도 (비공개)
- v_link_id, velocity_AVRG, probe

# File 10: 추정교통량 (비공개)
- ALL_AADT, PSCR_AADT(승용), BUS_AADT(버스), FGCR_AADT(화물)

# File 11/12: 혼잡빈도/시간강도 (비공개)
- FRIN_CG, TI_CG
```

#### Step 1.5: 데이터 품질 검증
```python
# 필수 QA 체크리스트:
1. 결측값 비율 확인 및 처리 전략 수립
2. 이상치 탐지 (IQR 방법 + 도메인 지식)
3. 좌표 유효성 검증 (한국 영역 내)
4. 시간 범위 일관성 확인
5. 격자 매핑 커버리지 확인 (매핑 안 된 데이터 비율)
6. NULL 마스킹 값 탐지 (-999, 9999 등 센티널 값)
7. 과산포(overdispersion) 진단: Var(Y) / E(Y) > 1 → NB 모형 필요
```

### Phase 2: 탐색적 데이터 분석 (EDA) (Week 2-3)

#### Step 2.1: 기존 신도시 교통사고 패턴 분석
```python
# 분석 차원:
1. 시간적 패턴: 월별/요일별/시간대별 사고 추이
2. 공간적 패턴: 핫스팟 분석 (Getis-Ord Gi*)
3. 사고 유형별: 차대사람, 차대차, 차량단독
4. 피해자 특성별: 어린이(<13), 고령자(65+), 임산부
5. 위반 유형별: 신호위반, 과속, 보행자보호의무위반
6. 환경 조건별: 날씨, 도로유형, 노면상태
```

#### Step 2.2: 취약계층별 심층 분석
```
[어린이]
- 등하교 시간대(7-9시, 13-16시) 사고 집중도
- 학교 반경 300m/500m 내 사고율
- 어린이보호구역 내외 사고 비교

[고령자]
- 고령 인구 비율 상위 격자 사고 패턴
- 무단횡단 등 보행 사고 분석
- 의료시설/복지시설 접근 경로 위험도

[임산부 (차별화 포인트)]
- 가임기 여성 인구 + 산부인과/소아과 위치 기반 추정
- 대중교통 이용 동선의 안전성
```

#### Step 2.3: 교통안전 시설 효과성 기초 분석
```python
# 시설 존재 여부에 따른 사고율 비교
- CCTV 설치 격자 vs 미설치 격자 사고율
- 과속방지턱 설치 전후 (시계열 가용 시) 사고 변화
- 어린이보호구역 지정 격자의 사고 감소 효과
- 횡단보도 밀도와 보행자 사고의 관계
```

#### Step 2.4: 과산포(Overdispersion) 진단
```python
# 격자별 사고건수의 분포 특성 확인
# Var(Y) / E(Y) >> 1 → 포아송 부적합, NB 모형 채택 근거

from scipy.stats import chi2
mean_y = grid_df['accident_total'].mean()
var_y = grid_df['accident_total'].var()
dispersion = var_y / mean_y  # >> 1이면 과산포

# Cameron-Trivedi 검정으로 통계적 확인
# → NB-GLM SPF 채택의 학술적 정당화
```

### Phase 3: 핵심 분석 -- TSVI 지수 개발 (Week 3-4)

#### Step 3.0: EB 안전도 추정 (신규 -- 핵심 차별화)

```python
# NB-GLM Safety Performance Function 적합
# Y_i ~ NB(mu_i, k)
# log(mu_i) = beta_0 + beta_1*log(AADT_i) + beta_2*road_length_i + ...

from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod.families import NegativeBinomial

# EB 보정: 관측값과 모형 기대값의 가중평균
# EB_i = w_i * y_i + (1 - w_i) * mu_hat_i
# w_i = 1 / (1 + mu_hat_i / k_hat)

# 출력:
# - eb_estimate: EB 보정 사고건수
# - excess: eb_estimate - expected (구조적 결함 지표)
# - EB 수축 플롯: 원시 vs EB 산점도 → 심사위원 인상적
```

#### Step 3.1: TSVI (교통안전 취약성 지수) 설계

```
TSVI = w1*PVI + w2*RRI + w3*FGI + w4*TRI

여기서:
PVI (Population Vulnerability Index) = 취약계층 인구 노출도
RRI (Road Risk Index) = 도로 구조적 위험도
FGI (Facility Gap Index) = 안전시설 공백도
TRI (Temporal Risk Index) = 시간대별 위험 변동성
```

#### Step 3.2: 각 하위 지수 산출

```python
# PVI (인구 취약성 지수) -- 실제 컬럼 매핑
PVI = (child_pop_estimated * 0.40 +    # File 16/17 프록시
       elderly_ratio * 0.35 +            # File 03
       childbearing_ratio * 0.25         # File 03
      ) * (1 + floating_pop_density_norm)  # File 04

# RRI (도로 위험성 지수) -- 실제 컬럼 매핑
RRI = f(traffic_volume_all,       # ALL_AADT (File 10)
        avg_speed,                 # velocity_AVRG (File 09)
        road_length_total,         # length (File 08)
        intersection_count,        # File 08 노드 기반
        congestion_freq * congestion_time_intensity)  # Files 11/12

# FGI (시설 공백 지수) -- 거리 가중 점수 추가
facility_coverage = count_score * 0.60 + distance_score * 0.40
  - count_score: cctv_count, speed_bump_count, crosswalk_count 정규화 합
  - distance_score: 1/(1 + dist/300m) 역거리 함수
FGI = 1 - normalized(facility_coverage)

# TRI (시간대별 위험 지수) -- 시간대 변동성 반영
TRI = (pop_hourly_cv * 0.35 +       # TMST_00~23 변동계수
       peak_nonpeak_ratio * 0.35 +    # 피크/비피크 비율
       congestion_freq * 0.30)         # File 11
```

#### Step 3.2b: MGWR 다중스케일 분석 (신규 -- 학술적 참신함)

```python
# MGWR: 변수별 독립 bandwidth로 공간적 영향 범위 차이 포착
# y_i = beta_0(u_i,v_i) + sum_k beta_k(u_i,v_i) * x_ik + e_i

from mgwr.sel_bw import Sel_BW
from mgwr.gwr import MGWR

# 예상 결과:
# - 교통량(ALL_AADT): 넓은 bandwidth → 도시 수준의 영향
# - 학교 근접성(dist_nearest_schools): 좁은 bandwidth → 초지역 효과
# - CCTV 밀도(cctv_count): 중간 bandwidth → 동네 수준
# - 인구 밀도: 넓은 bandwidth → 도시 계획 수준

# 출력: bandwidth 맵, 계수 표면, 공간 레짐 → 심사위원 감탄 포인트
```

#### Step 3.3: 가중치 결정 방법

```
방법 1: AHP (Analytic Hierarchy Process)
- 문헌 기반 쌍대비교 행렬 구성
- 일관성 비율(CR) < 0.1 검증
- 고유값법으로 우선순위 벡터 산출

방법 2: 데이터 기반 가중치 (Random Forest Feature Importance)
- 실제 사고 발생 여부를 종속변수로
- 각 하위 지수를 독립변수로
- Feature Importance로 가중치 도출

방법 3: 하이브리드 (추천)
- w_hybrid = alpha * w_AHP + (1-alpha) * w_data
- AHP: 일관성 검증(CR < 0.10) → 이론적 정당성
- RF: 데이터 기반 → 실증적 정당성
- 이론 + 실증의 결합 = 심사위원 호감

가중치 민감도 분석:
- ±10% 교란 100회 → 상위 20% 순위 Jaccard 안정성 측정
- 안정성 > 0.85 → 강건한 가중치 체계
```

#### Step 3.4: TSVI 검증
```python
# 기존 4개 신도시에서 검증:
1. TSVI 상위 20% 격자 vs 실제 사고 다발 격자 일치율 (Precision/Recall)
2. ROC-AUC: TSVI의 사고 예측 성능
3. 보정 분석: 10분위 예측 vs 관측, Spearman 상관
4. Lift 차트: 상위 20% 사고 포착률, lift 비율
5. 공간 자기상관 검증: Moran's I
6. 도시별 교차 검증: Leave-One-City-Out AUC
```

### Phase 4: 모델링 및 최적화 (Week 4)

#### Step 4.1: 사고 위험도 예측 -- 계층적 모델 구조

```python
# 모델 계층: 단순 → 복잡 순서
# Level 1: NB-GLM (기준 모델) -- 과산포 카운트 모형
#   Y_i ~ NB(mu_i, k), log(mu_i) = X*beta + offset(log(T_i))
#   T_i: 노출 변수 (교통량 또는 도로연장)
#
# Level 2: Spatial NB -- 공간 자기상관 반영
#   spreg.ML_Lag 또는 ML_Error 모형
#
# Level 3: MGWR -- 변수별 다중스케일 공간 이질성
#   mgwr.gwr.MGWR (변수별 독립 bandwidth)
#
# Level 4: XGBoost / LightGBM -- 비선형 + 교호작용
#   SHAP values로 해석력 확보

# 독립변수: TSVI 하위 지표 + 토지이용 + 인구 + 도로 + 시설
# 종속변수: 격자별 사고 건수 (NB-GLM) 또는 이진 분류 (ML)

# 평가:
- 5-fold Spatial Block CV (KMeans 기반 공간 분할)
- Leave-One-City-Out AUC
- SHAP values로 변수 중요도 해석
```

#### Step 4.2: 시설 최적 배치 -- 수리최적화 (핵심 차별화)

```python
# 접근법 1: Max Coverage (PuLP ILP)
# 결정변수: x[i,f] = 격자 i에 시설 f 설치 여부 (이진)
# 목적함수: max sum x[i,f] * TSVI[i] * effect[f]
# 제약: 예산, 격자당 1개, 최소 이격거리

# 접근법 2: p-Median
# min sum_i sum_j TSVI_i * d_ij * x_ij
# s.t. sum_j y_j = p
# p개 시설의 수요 가중 거리 최소화

# 접근법 3: Set Covering
# min sum_j y_j (최소 시설 수)
# s.t. 모든 고위험 격자(상위 20%)가 반경 500m 내 1개 이상 시설에 커버

# 출력: 최적 배치 맵, Pareto front(예산 vs 효과), 시설별 분포
```

#### Step 4.3: 하남교산 적용 -- Bootstrap CI + 전이 분석

```python
# 5단계 전이 분석법:

# [Step 1] 하남교산 토지이용계획 디지털화
- zoneCode/zoneName → LandUseClassifier 자동 분류
- 100m x 100m 격자에 용도 비율 매핑

# [Step 2] 유사 격자 매칭 (Analogous Grid Matching)
- StandardScaler 정규화 + 코사인/유클리드 유사도
- 상위 K개 유사 격자의 가중 평균으로 특성 추정

# [Step 3] Bootstrap CI 산출 (핵심 신규)
- B=1000회 리샘플링 → 95% CI
- CI_95% = [TSVI*(0.025), TSVI*(0.975)]
- CV(변동계수) → 전이 불확실성 지표
- "지적 정직성" = 심사위원 호감

# [Step 4] 인구/교통 시뮬레이션
- 토지이용 비례 인구 배분 (10만 가구 × 2.5인/가구)
- 매칭 도시의 인구구조(연령비율) 적용

# [Step 5] TSVI 예측 및 시설 배치 제안
- 추정된 특성값으로 TSVI 산출
- 불확실성 맵(CI 폭) 동시 제시
```

### Phase 5: 정량적 효과 추정 (Week 4-5)

#### Step 5.0: EB 기반 반사실 효과추정 (신규)

```python
# CMF (Crash Modification Factor) 기반 효과 추정
# CMF = E[Y_with_facility] / E[Y_without_facility]
# 사고감소율 = (1 - CMF) * 100%

# EB 기반 반사실(counterfactual):
# 1. SPF로 시설 미설치 시 기대 사고건수(E[Y_without]) 추정
# 2. EB 보정으로 노이즈 제거
# 3. 시설 설치 후 관측값(Y_with)과 비교
# 4. CMF = Y_with / EB_without

# 시설별 CMF (문헌 기반 + 데이터 보정):
- CCTV: CMF = 0.85 (15% 감소)
- 과속방지턱: CMF = 0.80 (20% 감소)
- 스마트횡단보도: CMF = 0.78 (22% 감소)
- 과속카메라: CMF = 0.70 (30% 감소)
```

#### Step 5.1: 시설별 사고 감소 효과 산출

```python
# 방법론 1: 회귀 기반 추정 (메인)
accident_rate = f(TSVI, facility_density, road_characteristics, population)
delta_accident = predict(with_facility) - predict(without_facility)

# 방법론 2: Nilsson Power Model (속도-사고 관계)
# 사망사고 비율 변화 = (V_after / V_before)^4
# 중상사고 비율 변화 = (V_after / V_before)^3
# 경상사고 비율 변화 = (V_after / V_before)^2

# 출력:
- 시설 유형별 사고 감소 기대건수
- 교통량 변화 추정 (속도 감소에 따른 안전 속도 달성률)
- 비용 편익 비율 (B/C Ratio)
```

#### Step 5.2: 시나리오 분석

```python
# 시나리오 설계:
Scenario 1 (기본): 현행 기준 최소 시설 설치
Scenario 2 (TSVI 최적): TSVI 기반 PuLP 최적 배치
Scenario 3 (이상적): 전체 고위험 격자 시설 설치

# 각 시나리오별 산출:
- 예상 사고 감소 건수/비율
- 커버리지 (고위험 격자 대비)
- 소요 예산 추정
- B/C Ratio 비교 (할인율 4.5%, 10년 NPV)
```

### Phase 6: 시각화 및 보고서 작성 (Week 5-6)

---

## 3. 핵심 분석 방법론 상세

### 3.1 카운트 모형 (Count Models)

| 방법 | 수식 | 용도 | 라이브러리 |
|------|------|------|-----------|
| Poisson GLM | `Y ~ Pois(mu), log(mu) = Xb` | 기준 모형 | statsmodels |
| NB-GLM | `Y ~ NB(mu, k), log(mu) = Xb` | 과산포 카운트 (메인) | statsmodels |
| Zero-Inflated NB | `P(Y=0) = pi + (1-pi)*NB(0)` | 영과잉 데이터 | statsmodels |
| EB 보정 | `EB = w*y + (1-w)*mu_hat` | 소표본 수축 | custom |

**핵심 수식 1: NB-GLM SPF**
```
Y_i ~ NB(mu_i, k)
log(mu_i) = beta_0 + beta_1*log(AADT_i) + beta_2*road_length_i + ... + offset(log(T_i))
Var(Y_i) = mu_i + mu_i^2 / k  (k: 과산포 파라미터)
```

**핵심 수식 2: EB 보정**
```
E[Y_i | y_i] = w_i * y_i + (1 - w_i) * mu_hat_i
w_i = 1 / (1 + mu_hat_i / k_hat)
- w_i → 1: 관측값 신뢰 (사고 많은 격자)
- w_i → 0: 모형 의존 (사고 적은 격자)
```

### 3.2 공간통계 방법

| 방법 | 수식 | 용도 | 라이브러리 |
|------|------|------|-----------|
| Kernel Density (KDE) | `f(x) = (1/nh) sum K((x-x_i)/h)` | 사고 밀집도 히트맵 | scipy, sklearn |
| Getis-Ord Gi* | `Gi* = sum w_ij*x_j / sum x_j` | 통계적 핫스팟 탐지 | PySAL (esda) |
| Moran's I | `I = (n/S0) * sum w_ij*z_i*z_j / sum z_i^2` | 공간 자기상관 | PySAL (esda) |
| Spatial Lag Model | `y = rho*Wy + Xb + e` | 인접 격자 영향 | PySAL (spreg) |
| GWR | `y_i = sum_k b_k(u_i,v_i)*x_ik + e_i` | 지역별 차별화 계수 | mgwr |
| **MGWR** | `y_i = sum_k b_k(u_i,v_i)*x_ik, bw_k 독립` | **다중스케일 분석** | mgwr |

**핵심 수식 3: MGWR**
```
y_i = beta_0(u_i,v_i) + sum_k beta_k(u_i,v_i) * x_ik + epsilon_i
각 beta_k는 독립적 bandwidth b_k를 가짐
- 교통량: b_traffic >> (도시 수준)
- 학교 근접성: b_school << (초지역 수준)
```

### 3.3 머신러닝 방법

| 방법 | 용도 | 라이브러리 |
|------|------|-----------|
| XGBoost / LightGBM | 사고 위험도 예측 (메인) | xgboost, lightgbm |
| Random Forest | 변수 중요도 + 앙상블 | sklearn |
| SHAP | 모델 해석 (블랙박스 → 설명가능) [A.7] | shap |
| K-Means | 격자 유형 분류, 공간 블록 CV | sklearn |
| Bayesian Optimization | 하이퍼파라미터 튜닝 | optuna |

### 3.4 수리최적화 방법

| 방법 | 수식 | 용도 | 라이브러리 |
|------|------|------|-----------|
| Max Coverage (ILP) | `max sum x_if * TSVI_i * eff_f` | 예산 내 효과 극대화 | PuLP |
| p-Median | `min sum TSVI_i * d_ij * x_ij` | 거리 최소화 배치 | PuLP |
| Set Covering | `min sum y_j, s.t. 전수 커버` | 최소 시설 수 | PuLP |
| Simulated Annealing | 메타휴리스틱 | 복잡한 제약 조건 | custom |

**핵심 수식 4: p-Median**
```
min sum_i sum_j TSVI_i * d_ij * x_ij
s.t. sum_j y_j = p
     sum_j x_ij = 1  (모든 수요점은 1개 시설에 할당)
     x_ij <= y_j      (시설 설치된 곳에만 할당)
```

### 3.5 인과추론 및 효과추정

| 방법 | 수식 | 용도 | 라이브러리 |
|------|------|------|-----------|
| CMF | `CMF = E[Y_with] / E[Y_without]` | 시설 효과 계수 | custom |
| Nilsson Power Model | `사망변화 = (V2/V1)^4` | 속도-사고 관계 | custom |
| DID | `delta = (Y1_treat - Y0_treat) - (Y1_ctrl - Y0_ctrl)` | 시설 설치 효과 | custom |
| PSM | Propensity Score Matching | 비교 가능 격자 매칭 | sklearn |

**핵심 수식 5: CMF (Crash Modification Factor)**
```
CMF = E[Y_with] / E[Y_without]
사고감소율 = (1 - CMF) * 100%
- CCTV: CMF = 0.85 → 15% 감소
- 과속카메라: CMF = 0.70 → 30% 감소
```

**핵심 수식 6: Bootstrap CI (전이 불확실성)**
```
B = 1000회 리샘플링
각 시행: top_k 유사 격자에서 복원추출 → 가중평균 TSVI
CI_95% = [TSVI*(0.025), TSVI*(0.975)]
CV = std(TSVI*) / mean(TSVI*)  → 불확실성 지표
```

---

## 4. 데이터 융합(Cross-Data Fusion) 전략

### 4.1 핵심 융합 테이블

모든 분석의 기본 단위: **100m x 100m 격자(Grid)**

```
[마스터 격자 테이블] -- build_grid_master() 산출물
grid_id | region | geometry | centroid_x | centroid_y |
--- 인구 ---
total_resident_pop | elderly_ratio | childbearing_ratio | child_pop_estimated |
floating_pop_density | floating_pop_count |
worker_hourly_cv | worker_peak_ratio | visitor_hourly_cv |
--- 도로/교통 ---
road_length_total | max_speed_max | lanes_max | intersection_count |
avg_speed | traffic_volume_all | traffic_volume_passenger | traffic_volume_bus | traffic_volume_truck |
congestion_freq | congestion_time_intensity |
--- 교육/보호시설 ---
school_count | kindergarten_count | daycare_count | child_protection_zone_count |
dist_nearest_schools | dist_nearest_crosswalks | dist_nearest_bus_stops | dist_nearest_cctv |
--- 안전시설 ---
cctv_count | speed_bump_count | crosswalk_count | bus_stop_count |
--- 토지이용 ---
residential_ratio | commercial_ratio | industrial_ratio | green_ratio | education_ratio | road_ratio |
--- 사고 이력 ---
accident_total | accident_severe |
--- EB 보정 (신규) ---
eb_estimate | eb_excess | spf_expected |
--- MGWR (신규) ---
mgwr_coeff_traffic | mgwr_coeff_school | mgwr_coeff_cctv | mgwr_bw_traffic | mgwr_bw_school |
--- 최적화 (신규) ---
optimal_facility_type | optimal_priority_rank |
--- 산출 지수 ---
PVI | RRI | FGI | TRI | TSVI | TSVI_score | TSVI_grade
```

### 4.2 창의적 융합 분석 아이디어 (차별화 포인트)

#### 융합 1: "어린이 통학 위험 경로 분석"
```
학교/유치원/어린이집 위치 + 상주 어린이 인구 격자 + 도로망 + 교통량 + 사고이력
→ 최단 통학 경로별 누적 위험도 산출
→ "이 학교에 다니는 어린이는 통학길에서 평균 N개의 고위험 격자를 통과한다"
```

#### 융합 2: "시간대별 동적 위험 지도"
```
시간대별 유동인구 + 시간대별 교통량 + 시간대별 사고 이력
→ 24시간 동적 TSVI 변화 애니메이션
→ "오전 8시 등교 시간, 이 격자의 위험도는 평소의 3.2배"
```

#### 융합 3: "우회전 교차로 위험도 특별 분석" (대회 요구사항 직접 반영)
```
교차로 위치 + 우회전 구조 + 횡단보도 위치 + 보행자 사고 + 교통량
→ 우회전 교차로 유형별 위험도 분류
→ 무신호 우회전 vs 신호 우회전 사고율 비교
```

#### 융합 4: "토지이용 패턴 기반 도시 유사성 매칭"
```
기존 4개 신도시 토지이용 + 하남교산 토지이용계획
→ 격자별 토지이용 벡터 유사도 (코사인 유사도)
→ 하남교산의 각 격자에 가장 유사한 기존 신도시 격자 매칭
→ "하남교산 A-123 격자는 동탄2 B-456 격자와 토지이용 유사도 0.94"
```

#### 융합 5: "EB 기반 구조적 결함 지도" (신규)
```
SPF 기대 사고건수 + EB 보정 사고건수
→ excess = EB - expected
→ excess > 0인 격자 = 통계적으로 기대 이상 위험한 "구조적 결함" 지점
→ 시설 배치 우선순위의 학술적 근거
```

#### 융합 6: "교통사고 비용 추정"
```
사고 건수 + 중상/경상/사망 비율 + 교통사고 사회적 비용 계수 (국토부 기준)
→ 격자별 연간 교통사고 사회적 비용 추정
→ 시설 설치 비용 대비 사고 감소 편익 = B/C Ratio (할인율 4.5%, 10년 NPV)
```

---

## 5. 시각화 전략

### 5.1 Python 시각화 (분석 노트북 내)

| 시각화 유형 | 목적 | 라이브러리 | 예상 개수 |
|------------|------|-----------|----------|
| 인터랙티브 히트맵 | TSVI 분포, 사고 밀도 | folium, plotly | 5-8장 |
| 시계열 차트 | 월별/시간대별 사고 추이 | matplotlib, seaborn | 4-6장 |
| 상관관계 매트릭스 | 변수 간 관계 | seaborn heatmap | 2-3장 |
| SHAP 플롯 | 모델 해석 | shap | 3-4장 |
| 산점도/버블차트 | 격자별 특성 비교 | plotly | 3-4장 |
| 막대/도넛 차트 | 비교 통계 | matplotlib | 5-7장 |
| 시나리오 비교 차트 | 효과 추정 결과 | plotly | 2-3장 |
| **EB 수축 플롯** (신규) | 원시 vs EB 보정 산점도 | matplotlib | 1장 |
| **MGWR bandwidth 맵** (신규) | 변수별 공간 스케일 차이 | folium/matplotlib | 4장 |
| **보정 플롯** (신규) | 예측 vs 관측 45도선 | matplotlib | 2장 |
| **Pareto front** (신규) | 예산 vs 효과 트레이드오프 | plotly | 1장 |
| **Bootstrap CI 맵** (신규) | 하남교산 불확실성 | folium | 2장 |

### 5.2 QGIS 시각화 (핵심 -- 대회 필수 요구)

| 맵 유형 | 내용 | 기법 |
|---------|------|------|
| TSVI 종합 위험도 맵 | 격자별 색상 그라데이션 | Graduated Symbol |
| 사고 핫스팟 맵 | Gi* 통계량 기반 | Hot/Cold spot |
| 시설 공백 맵 | 안전시설 사각지대 | Buffer + Inverse |
| 어린이 통학 위험 경로 | 통학 경로 위험 구간 | Line graduated |
| 하남교산 예측 맵 | TSVI 예측값 + CI 폭 | Graduated (predicted) |
| 최적 배치 제안 맵 | PuLP 최적화 결과 | Point + Label |
| Before/After 비교 맵 | 시설 설치 전후 TSVI 변화 | Side-by-side / Swipe |
| 시간대별 동적 맵 | 24시간 위험도 변화 | QGIS Temporal Controller |
| **EB 초과사고 맵** (신규) | 구조적 결함 격자 | Diverging color |
| **MGWR 계수 맵** (신규) | 변수별 영향력 공간 분포 | Multi-panel |

### 5.3 시각화 스타일 가이드
```
- 색상 팔레트: 안전(녹색) → 주의(황색) → 위험(적색) 직관적 배색
- 폰트: 나눔스퀘어 또는 Pretendard (가독성 최우선)
- 지도 축척 표시 필수
- 범례(Legend) 반드시 포함
- 배경지도: OpenStreetMap 또는 Vworld
- 해상도: 최소 300 DPI (보고서 인쇄 품질)
```

---

## 6. 하남교산 적용 전략 (핵심 차별화)

### 6.1 문제: 아직 건설되지 않은 도시

하남교산은 3기 신도시로 아직 **미건설** 상태이므로:
- 실제 인구 데이터 없음
- 실제 교통량 데이터 없음
- 실제 사고 데이터 없음
- 토지이용계획 + 지구계획 도서만 존재

### 6.2 해결: 5단계 전이 분석법 (Bootstrap CI 추가)

```
[Step 1] 하남교산 토지이용계획 디지털화
- 지구계획 도서에서 토지이용 용도 추출 (주거, 상업, 교육, 녹지 등)
- zoneCode/zoneName → LandUseClassifier 자동 분류
- 100m x 100m 격자에 매핑

[Step 2] 유사 격자 매칭 (Analogous Grid Matching)
- 기존 4개 신도시의 각 격자를 토지이용 유형으로 벡터화
- StandardScaler 정규화 + 코사인 유사도 계산
- 상위 K개 유사 격자의 특성 가중 평균으로 하남교산 특성 추정

[Step 3] Bootstrap 불확실성 정량화 (신규)
- B=1000회 리샘플링 → 95% CI
- 격자별 CV(변동계수) 산출 → 전이 신뢰도 지표
- CI 폭이 넓은 격자 = 유사 격자 간 변동 큼 = 주의 필요

[Step 4] 인구/교통 시뮬레이션
- 유사 격자의 인구밀도, 연령구조, 교통량을 하남교산 예상 규모에 맞게 스케일링
- 하남교산 계획 인구(약 10만 가구) 기준으로 보정
- 매칭 도시의 인구구조(연령비율) 적용

[Step 5] TSVI 예측 및 시설 배치 제안
- 추정된 특성값으로 TSVI 산출
- 고위험 격자에 시설 최적 배치 (PuLP)
- 시나리오별 효과 추정 + 불확실성 맵 동시 제시
```

### 6.3 하남교산 맞춤 제안 4가지 (대회 요구사항 직접 대응)

```
1. 스쿨존 안전 설계
   - 학교 예정 부지 반경 300m 격자 TSVI 산출
   - 통학 예상 경로 위험 구간 식별
   - 스마트 횡단보도 + 속도제한 시설 배치 제안

2. 통학/통근 경로 안전
   - 주거지 → 학교/버스정류장 최적 안전 경로 설계
   - 위험 구간 우회 경로 제안
   - 보행자 전용 구간 확대 제안

3. 우회전 교차로 안전
   - 하남교산 도로 계획에서 우회전 교차로 추출
   - 기존 신도시 우회전 사고 패턴 적용
   - 무신호 → 신호화, 보행 섬, 속도 저감 시설 제안

4. 취약 시간대 대응
   - 등하교/출퇴근 시간대 동적 안전시설 가동 계획
   - 야간 보행자 감지 시스템 배치
   - 계절/날씨별 대응 시나리오
```

---

## 7. 정량적 효과 추정 방법론

### 7.0 EB 기반 반사실 효과추정 (신규)

```python
# CMF (Crash Modification Factor) 기반
# 1. SPF로 시설 미설치 시 기대 사고건수 E[Y_without] 추정
# 2. EB 보정으로 노이즈 제거
# 3. 시설 설치 후 관측값 Y_with과 비교
# 4. CMF = Y_with / EB_without

# 장점: 회귀분석 기반 인과적 효과 추정
# 단점: SPF 적합 품질에 의존
```

### 7.1 사고 감소 효과 추정

```python
# 방법론 1: 회귀 기반 추정 (메인)
accident_rate = f(TSVI, facility_density, road_characteristics, population)
delta_accident = predict(with_facility) - predict(without_facility)

# 방법론 2: CMF 적용 (보조)
prevented = sum(EB_estimate_i * (1 - CMF_facility))  for target grids

# 방법론 3: 문헌 기반 계수 적용 (검증)
# 국내외 연구 결과의 사고 감소율을 적용하여 교차 검증
```

### 7.2 교통량 변화 추정

```python
# 속도 감소 → 안전 속도 도달률 변화 (Nilsson Power Model)
# 시설 유형별 속도 감소 효과:
speed_reduction = {
    'speed_camera': -12 km/h,
    'speed_bump': -20 km/h,
    'smart_crosswalk': -7 km/h
}

# 속도 감소 → 사고 심각도 감소
# 사망사고 비율 변화 = (V_after / V_before)^4
# 중상사고 비율 변화 = (V_after / V_before)^3
# 경상사고 비율 변화 = (V_after / V_before)^2
```

### 7.3 비용편익분석 (B/C Ratio)

```python
# 비용 (Cost):
facility_cost = {
    'smart_crosswalk': 5000만원/개소,
    'speed_camera': 3000만원/개소,
    'cctv': 1500만원/대,
    'speed_bump': 200만원/개소,
    'safety_sign': 50만원/개소
}

# 편익 (Benefit):
# 교통사고 사회적 비용 (국토교통부 2024 기준)
accident_social_cost = {
    'death': 약 8.5억원/건,
    'serious_injury': 약 1.2억원/건,
    'minor_injury': 약 1500만원/건,
    'property_damage': 약 500만원/건
}

# NPV 계산 (할인율 4.5%, 10년)
# NPV = -Investment + sum(Annual_Benefit / (1+r)^t)  for t=1..10
# B/C Ratio = PV(Benefit) / Investment
# B/C > 1 이면 경제적 타당성 있음
```

---

## 8. 검증 프레임워크 (신설)

### 8.1 내부 검증 (Internal Validation)

**Level 1: 공간 블록 CV (5-fold)**
```
- KMeans 기반 공간 블록 분할
- 인접 격자가 같은 fold → spatial leakage 방지
- fold별 AUC, RMSE 보고
```

**Level 2: Leave-One-City-Out 외부 검증**
```
- 3개 도시 학습 → 1개 도시 테스트 (4-fold)
- 도시별 AUC + 평균 AUC
- 도시 간 전이 가능성의 정량적 근거
```

**Level 3: 보정 분석 (Calibration)**
```
- TSVI 10분위별 예측 사고율 vs 관측 사고율
- 45도 기준선에 가까울수록 보정 양호
- Spearman 상관계수 보고
```

**Level 4: Lift 차트**
```
- 상위 5%, 10%, 15%, 20%, 25%, 30%, 50% 사고 포착률
- lift@20% > 2.0 → TSVI가 랜덤보다 2배 이상 효과적
- 누적 포착률 곡선 시각화
```

### 8.2 전이 검증 (Transfer Validation)

**유사도 구간별 보정**
```
- 유사도 0.9 이상: 높은 신뢰도 전이
- 유사도 0.7-0.9: 중간 신뢰도, CI 폭 주시
- 유사도 0.7 미만: 낮은 신뢰도, 대안 필요
```

**Bootstrap 95% CI 커버리지**
```
- B=1000회, CI 커버 범위
- 격자별 CV: 평균 < 0.3 목표
- CI 맵으로 불확실성 공간 분포 시각화
```

### 8.3 민감도 분석 (Sensitivity Analysis)

| 분석 | 변동 범위 | 측정 지표 |
|------|----------|----------|
| 가중치 교란 | ±10% | 상위 20% Jaccard 안정성 |
| EB vs 원시 사고건수 | on/off | TSVI-사고 상관 변화 |
| MGWR vs GWR vs Global | 모형 전환 | AICc, 잔차 Moran's I |
| 예산 변동 | ±20% | 시설 배치 결과, B/C Ratio |
| k 매칭 수 | k=3,5,7,10 | 전이 TSVI 안정성 |

---

## 9. 보고서 구조 (발표 심사 60% 최적화)

### 9.1 PPT 보고서 구조 (약 38슬라이드)

```
[도입부] (5슬라이드)
1. 표지 - 프로젝트명, 팀명, SAFE-Grid Framework 로고
2. 문제 인식 - 왜 3기 신도시 교통안전이 중요한가? (통계/뉴스)
3. 분석 목표 - 명확한 3가지 분석 질문 제시
4. 분석 프레임워크 - SAFE-Grid 소개 (한 눈에 보이는 다이어그램)
5. 데이터 개요 - 25개 데이터셋 활용 전략 요약

[S: Spatial Risk Mapping] (7슬라이드)
6. 기존 신도시 교통사고 현황 분석
7. 공간적 핫스팟 분석 결과 (QGIS 맵)
8. 시간대별/취약계층별 사고 패턴
9. 4개 신도시 비교 분석
10. 우회전 교차로 특별 분석
11. 주요 발견사항 요약
12. 핵심 인사이트: "사고는 랜덤이 아니라 구조적이다"

[A: Age-group Vulnerability] (5슬라이드)
13. 어린이 취약성 분석 (통학 경로, 보호구역)
14. 고령자 취약성 분석 (보행 패턴, 의료접근성)
15. 임산부/가임여성 분석 (추정 방법론 포함)
16. 취약계층 복합 노출 지역 식별
17. TSVI 산출 방법론 및 검증 결과

[F: Facility Gap Detection + 모델링] (7슬라이드)
18. 안전시설 현황 및 공백 분석
19. EB 수축 플롯: 원시 vs 보정 사고건수 (학술적 엄밀성)
20. MGWR bandwidth 맵: 변수별 공간 스케일 차이 (학술적 참신함)
21. 시설 효과성 분석 (CMF 기반)
22. 하남교산 적용: 토지이용 유사 매칭 + Bootstrap CI
23. 하남교산 TSVI 예측 맵 + 불확실성 맵
24. 검증 결과: AUC, Lift, 보정 플롯

[E: Effect Estimation & Optimization] (7슬라이드)
25. 수리최적화 배치 결과 (p-Median, MaxCoverage)
26. 최적 배치 맵: PuLP 결과 + 할당 네트워크
27. 하남교산 맞춤 대안 4가지
28. 정량적 효과 추정 (사고 감소, Nilsson)
29. 비용편익분석 결과 (NPV, B/C Ratio)
30. 시나리오별 비교
31. Pareto front: 예산 vs 효과 트레이드오프

[결론] (3슬라이드)
32. 핵심 제언 요약 (1장에 압축)
33. 분석의 의의 및 확장 가능성
34. Q&A / 감사 슬라이드

[부록] (4슬라이드 -- 질의 대비용)
35. 방법론 상세 (NB-GLM, EB, MGWR 수식)
36. 민감도 분석 결과
37. 데이터 품질 리포트
38. 참고문헌
```

### 9.2 발표 스토리텔링 전략

```
[오프닝 후크] (30초)
"2024년 한 해, 어린이보호구역 내 교통사고 사상자는 XXX명이었습니다.
 3기 신도시 하남교산에 10만 가구가 입주하면, 이 아이들은 안전할까요?"

[문제 정의] (2분)
- 신도시 교통안전의 구조적 문제 제기
- 기존 접근법의 한계: "사고가 난 후 대응" → "사고 전 예방"으로 전환

[분석 과정] (7분)
- SAFE-Grid 프레임워크에 따라 순차 설명
- 각 단계에서 "발견 → 인사이트 → 시사점" 흐름 유지
- EB 수축 플롯, MGWR bandwidth 맵 → "이것이 우리의 차별점"
- QGIS 맵과 차트를 교차 활용하여 시각적 설득력 극대화

[결론 및 제언] (3분)
- 하남교산 맞춤 4가지 대안 (구체적 위치 + 예상 효과 + 불확실성)
- 비용편익 수치로 현실성 강조
- "아이들의 안전한 통학길을 위한 데이터 기반 솔루션"으로 마무리

[클로징] (30초)
- 감성적 마무리: "데이터가 증명한, 더 안전한 신도시의 설계도"
```

---

## 10. 타임라인 (6주 -- 2026.2.6 ~ 3.20)

### Week 1 (2/6 ~ 2/12): 데이터 수집 및 환경 구축
| 일자 | 작업 | 산출물 |
|------|------|--------|
| 2/6-7 | COMPAS 노트북 환경 확인, 데이터 다운로드/로드 테스트 | 환경 설정 완료 |
| 2/8-9 | 공개 데이터 15개 파일 전처리 시작 | 전처리 스크립트 |
| 2/10-12 | 격자 마스터 테이블 구축, 공간 조인 | master_grid 테이블 |

### Week 2 (2/13 ~ 2/19): 데이터 전처리 완료 및 EDA 시작
| 일자 | 작업 | 산출물 |
|------|------|--------|
| 2/13-14 | 비공개 데이터 10개 통합 (COMPAS 노트북) | 통합 데이터셋 |
| 2/15-16 | 데이터 품질 검증, 결측치/이상치 처리 | QA 보고서 |
| 2/17-19 | EDA 시작: 기술통계, 분포 확인, 초기 시각화 | EDA 노트북 v1 |

### Week 3 (2/20 ~ 2/26): EDA 완료 + TSVI 개발 + EB/NB-GLM
| 일자 | 작업 | 산출물 |
|------|------|--------|
| 2/20-21 | 심층 EDA: 공간 패턴, 시간 패턴, 취약계층 분석 | EDA 노트북 v2 |
| 2/22-23 | NB-GLM SPF 적합 + EB 보정 | EB 보정 결과 |
| 2/24-26 | TSVI 하위 지수 산출 + AHP 가중치(CR<0.10) | TSVI v1 |

### Week 4 (2/27 ~ 3/5): 모델링 + MGWR + Bootstrap + 하남교산
| 일자 | 작업 | 산출물 |
|------|------|--------|
| 2/27-28 | MGWR 적합 + bandwidth 분석 | MGWR 결과 |
| 3/1-2 | XGBoost + SHAP + 공간CV 검증 | 모델 성능 리포트 |
| 3/3-5 | 하남교산 토지이용 매칭 + Bootstrap CI | 하남교산 예측+CI 맵 |

### Week 5 (3/6 ~ 3/12): 최적 배치 및 효과 추정
| 일자 | 작업 | 산출물 |
|------|------|--------|
| 3/6-7 | PuLP 최적 배치 (MaxCoverage, p-Median) | 최적화 노트북 |
| 3/8-9 | CMF 효과 추정 + 비용편익분석 (NPV, B/C) | 효과 추정 결과 |
| 3/10-12 | QGIS 시각화 본격 제작 | QGIS 프로젝트 파일 |

### Week 6 (3/13 ~ 3/20): 보고서 작성 및 최종 제출
| 일자 | 작업 | 산출물 |
|------|------|--------|
| 3/13-14 | PPT 보고서 초안 작성 (38슬라이드) | PPT v1 |
| 3/15-16 | 코드 정리, 주석 추가, 재현성 확인 | 최종 .ipynb |
| 3/17-18 | PPT 수정/보완, 발표 스크립트 작성 | PPT v2 + 스크립트 |
| 3/19 | 최종 리뷰, 발표 리허설 | 최종 산출물 |
| 3/20 | 제출 (18:00 마감) | 제출 완료 |

### 주의: 병렬 작업 필수
```
COMPAS 노트북(비공개 데이터)과 로컬(공개 데이터)을 병렬로 작업
- COMPAS: 비공개 데이터 전처리 + 사고 분석
- 로컬: 공개 데이터 전처리 + QGIS 작업 + 보고서
```

---

## 11. 리스크 관리 및 대응 전략

### 11.1 기술적 리스크

| 리스크 | 확률 | 영향 | 대응 전략 |
|--------|------|------|----------|
| COMPAS 노트북 성능 부족 | 중 | 높음 | 데이터 샘플링, 청크 처리, 필수 분석 우선 |
| 격자 매핑 오류 | 중 | 높음 | CRS 통일 철저, 소규모 테스트 먼저 |
| 비공개 데이터 형식 불일치 | 중 | 중 | 코드북 꼼꼼히 확인, 빠른 EDA로 구조 파악 |
| 모델 성능 미달 | 낮 | 중 | 앙상블/간단한 모델로 대체, 해석력으로 보완 |
| QGIS 시각화 시간 초과 | 중 | 중 | 템플릿 미리 준비, Python으로 대체 가능 |
| **EB 수렴 실패** (신규) | 중 | 중 | alpha 초기값 변경, Poisson GLM 대체 |
| **MGWR 시간 초과** (신규) | 중 | 중 | GWR(단일 bandwidth) 대체, 격자 수 축소 |
| **PuLP 변수 과다** (신규) | 낮 | 중 | 후보 격자 수 제한(상위 50%), Greedy 대체 |

### 11.2 전략적 리스크

| 리스크 | 확률 | 영향 | 대응 전략 |
|--------|------|------|----------|
| 다른 팀과 접근법 유사 | 높음 | 높음 | TSVI + EB + MGWR + 수리최적화가 차별화 핵심 |
| 하남교산 적용 논리 약함 | 중 | 높음 | Bootstrap CI로 불확실성 정량화 → 지적 정직성 |
| 발표 시간 초과/부족 | 중 | 중 | 리허설 3회 이상, 시간 배분표 작성 |
| 코드 재현성 실패 | 낮 | 높음 | 환경(requirements.txt), 시드값 고정, 주석 철저 |

### 11.3 Fallback 전략

```
만약 NB-GLM이 수렴하지 않으면:
→ Poisson GLM + 과산포 보정(quasi-Poisson)으로 대체
→ EB 보정 대신 이동평균 평활화(spatial smoothing)

만약 MGWR이 시간 내 완료되지 않으면:
→ GWR(단일 bandwidth)로 대체 → 여전히 공간 이질성 포착
→ 격자 수를 500개 이하로 축소

만약 PuLP가 풀리지 않으면:
→ Greedy 알고리즘 + 시뮬레이티드 어닐링으로 대체
→ 근사 최적해도 충분히 경쟁력 있음

만약 시간이 부족하면:
→ QGIS 맵 5장 + PPT 25슬라이드 최소 완성
→ EB + MGWR 중 하나만 선택 (EB 우선)
→ 발표 심사 60%이므로 보고서 퀄리티 우선
```

---

## 12. 기술 스택 상세

### 12.1 Python 라이브러리

```python
# 데이터 처리
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon

# 통계 모델링 (신규)
import statsmodels.api as sm
from statsmodels.genmod.families import NegativeBinomial

# 공간 분석
import pysal
from pysal.explore import esda          # Moran's I, Getis-Ord Gi*
from pysal.model import spreg           # Spatial Regression
from mgwr.gwr import GWR, MGWR         # GWR, MGWR
from mgwr.sel_bw import Sel_BW

# 머신러닝
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.cluster import KMeans
from xgboost import XGBClassifier
import lightgbm as lgb
import shap
import optuna

# 최적화
from pulp import LpProblem, LpMinimize, LpMaximize, LpVariable, lpSum
from scipy.optimize import minimize

# 시각화
import matplotlib.pyplot as plt
import seaborn as sns
import folium
from folium.plugins import HeatMap, TimestampedGeoJson
import plotly.express as px
import plotly.graph_objects as go

# 기타
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler, StandardScaler
```

### 12.2 QGIS 작업 목록

```
1. 프로젝트 CRS 설정: EPSG:4326
2. 배경지도: Vworld 타일맵 또는 OpenStreetMap
3. 격자 레이어 스타일링: Graduated Symbol (TSVI 값)
4. 포인트 레이어: 학교, CCTV, 사고 위치 등
5. 버퍼 분석: 어린이보호구역 300m 버퍼
6. 네트워크 분석: 통학 경로 (QGIS Network Analysis)
7. Print Layout: A3 크기 지도 제작 (보고서용)
8. Temporal Controller: 시간대별 동적 맵 (옵션)
```

---

## 13. 심사위원 관점에서의 체크리스트

### 13.1 논리의 명확성
- [ ] 분석 목표가 명확하게 3가지 이내로 정의되어 있는가?
- [ ] 각 분석 단계가 논리적으로 연결되는가?
- [ ] 결론이 분석 결과에서 자연스럽게 도출되는가?
- [ ] 가정과 한계를 명시적으로 밝혔는가?
- [ ] EB/MGWR 등 방법론 선택의 근거가 명확한가?

### 13.2 가독성
- [ ] 보고서 구조가 일관성 있게 정리되어 있는가?
- [ ] 전문 용어에 대한 설명이 충분한가?
- [ ] 시각 자료가 직관적이고 이해하기 쉬운가?
- [ ] 핵심 메시지가 한 눈에 파악되는가?

### 13.3 데이터 활용성
- [ ] 25개 데이터셋 중 최소 20개 이상 활용했는가?
- [ ] 공개/비공개 데이터를 모두 활용했는가?
- [ ] 창의적인 데이터 융합 분석이 3건 이상인가?
- [ ] 외부 데이터를 보완적으로 활용했는가? (도로교통공단 통계 등)

### 13.4 분석의 창의성
- [ ] 기존에 없던 새로운 지표(TSVI)를 개발했는가?
- [ ] EB + MGWR + 수리최적화 = 학술적 차별화가 있는가?
- [ ] 하남교산 적용 방법이 독창적인가? (Bootstrap CI)
- [ ] "와, 이런 분석도 가능하구나" 순간이 있는가?

### 13.5 보고서 충실성
- [ ] PPT 디자인이 전문적이고 일관성 있는가?
- [ ] 모든 차트/맵에 제목, 범례, 출처가 있는가?
- [ ] 발표 스크립트가 자연스럽고 시간에 맞는가?
- [ ] 참고문헌/출처를 충실히 기재했는가?

### 13.6 코드 완성도
- [ ] 코드가 에러 없이 처음부터 끝까지 실행되는가?
- [ ] 주석과 마크다운 설명이 충분한가?
- [ ] 변수명이 직관적이고 코드 구조가 깔끔한가?
- [ ] requirements.txt 또는 환경 설정이 포함되었는가?
- [ ] 시드값이 고정되어 재현 가능한가?

### 13.7 심사위원이 볼 5가지 차별화 요소

1. **EB 수축 플롯**: 원시 사고건수 vs EB 보정값 산점도 → 통계적 정교함
2. **MGWR bandwidth 맵**: 4개 변수의 서로 다른 공간 스케일 → 학술적 참신함
3. **수리최적화 맵**: p-Median CCTV 배치 + 할당 네트워크 → 전문적 엄밀성
4. **보정 플롯**: 예측 vs 관측 45도 기준선 → 모델 검증 능력
5. **전이 불확실성 맵**: 하남교산 Bootstrap CI 폭 → 지적 정직성

---

## 14. 최종 제출물 체크리스트

```
[필수 제출물]
1. 데이터 파일 (ZIP)
   - 전처리된 데이터셋
   - 마스터 격자 테이블 (EB/MGWR/최적화 컬럼 포함)
   - TSVI 산출 결과

2. 소스 코드 (.ipynb)
   - 01_data_preprocessing.ipynb
   - 02_eda_exploration.ipynb
   - 03_tsvi_development.ipynb (EB + AHP)
   - 04_ml_modeling.ipynb (MGWR + XGBoost)
   - 05_hanam_gyosan_application.ipynb (Bootstrap CI)
   - 06_effect_estimation.ipynb (CMF + Nilsson)
   - 07_optimization.ipynb (PuLP)
   - 08_validation.ipynb (검증 프레임워크)

3. 분석 보고서 (PPT/PDF)
   - 38슬라이드 PPT
   - PDF 변환본

4. 발표 스크립트
   - 13분 내외 발표 원고

5. QGIS 시각화
   - .qgz 프로젝트 파일
   - 내보낸 맵 이미지 (PNG/PDF)
```

---

## 부록 A: 참고 문헌

### A.1 Empirical Bayes & Safety Performance Functions
1. Hauer, E. (1997). *Observational Before-After Studies in Road Safety*. Pergamon.
2. Persaud, B. & Lyon, C. (1999). Empirical Bayes before-after safety studies: Lessons learned from two decades of experience. *Accident Analysis & Prevention*, 39(3).
3. Hauer, E., Harwood, D., Council, F. & Griffith, M. (2002). Estimating Safety by the Empirical Bayes Method. *Transportation Research Record*, 1784.

### A.2 공간통계 & MGWR
4. Fotheringham, A. S., Yang, W. & Kang, W. (2017). Multiscale Geographically Weighted Regression (MGWR). *Annals of the American Association of Geographers*, 107(6).
5. Oshan, T. M., Li, Z., Kang, W., Wolf, L. J. & Fotheringham, A. S. (2019). mgwr: A Python implementation of multiscale geographically weighted regression. *Journal of Open Source Software*, 4(42).
6. Getis, A. & Ord, J. K. (1992). The Analysis of Spatial Association by Use of Distance Statistics. *Geographical Analysis*, 24(3).
7. Anselin, L. (1995). Local Indicators of Spatial Association—LISA. *Geographical Analysis*, 27(2).
8. Fotheringham, A. S., Brunsdon, C. & Charlton, M. (2002). *Geographically Weighted Regression*. Wiley.

### A.3 카운트 모형 & 사고 분석
9. Lord, D. & Mannering, F. (2010). The statistical analysis of crash-frequency data: A review and assessment of methodological alternatives. *Transportation Research Part A*, 44(5).
10. Lord, D., Washington, S. & Ivan, J. (2005). Poisson, Poisson-gamma and zero-inflated regression models of motor vehicle crashes. *Accident Analysis & Prevention*, 37(1).

### A.4 공간최적화
11. Daskin, M. S. (2013). *Network and Discrete Location: Models, Algorithms, and Applications*. Wiley.
12. Church, R. & ReVelle, C. (1974). The maximal covering location problem. *Papers of the Regional Science Association*, 32(1).

### A.5 전이 & 불확실성
13. Gelman, A., Carlin, J. B., Stern, H. S. et al. (2013). *Bayesian Data Analysis* (3rd ed.). CRC Press.
14. Efron, B. & Tibshirani, R. J. (1994). *An Introduction to the Bootstrap*. CRC Press.

### A.6 국내 문헌 및 데이터
15. 도로교통공단 (2024). *2024 교통사고 통계분석*.
16. 국토교통부 (2024). *교통사고 사회적 비용 산출 가이드라인*.
17. 경찰청 (2024). *어린이보호구역 교통사고 현황*.
18. 한국교통안전공단(KOTSA) (2024). *스마트 교통안전시설 효과분석*.
19. LH (2023). *3기 신도시 교통계획 보고서*.
20. TAAS 교통사고 분석시스템. https://taas.koroad.or.kr
21. 도로교통법 시행규칙, 민식이법 (어린이 보호구역 관련 특별법).

### A.7 방법론
22. Nilsson, G. (2004). Traffic safety dimensions and the Power Model to describe the effect of speed on safety. *Bulletin 221*, Lund Institute of Technology.
23. Lundberg, S. M. & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS 2017*.

---

*이 전략서는 2026.2.6 기준으로 작성되었으며, 분석 진행에 따라 지속 업데이트됩니다.*
*EB/MGWR/수리최적화/Bootstrap CI 방법론 추가 (2026.2.6 v2)*
