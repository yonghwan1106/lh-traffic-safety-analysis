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

#### Step 1.3: 인구 데이터 통합
```python
# 상주인구 (공개데이터)
- 격자별 연령대별/성별 상주인구

# 유동인구 (비공개데이터 - COMPAS 노트북)
- 월별/시간대별 유동인구
- 직장인구, 방문인구
- 평일/주말 서비스인구
```

#### Step 1.4: 도로 네트워크 데이터 통합
```python
# 비공개 데이터 (COMPAS 노트북)
- 상세 도로망 (Level 6)
- 평균 속도 데이터
- 추정 교통량 (전체/승용/버스/화물)
- 혼잡빈도/시간강도
```

#### Step 1.5: 데이터 품질 검증
```python
# 필수 QA 체크리스트:
1. 결측값 비율 확인 및 처리 전략 수립
2. 이상치 탐지 (IQR 방법 + 도메인 지식)
3. 좌표 유효성 검증 (한국 영역 내)
4. 시간 범위 일관성 확인
5. 격자 매핑 커버리지 확인 (매핑 안 된 데이터 비율)
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

### Phase 3: 핵심 분석 -- TSVI 지수 개발 (Week 3-4)

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
# PVI (인구 취약성 지수)
PVI = (어린이인구비율 * w_child +
       고령인구비율 * w_elderly +
       가임여성비율 * w_pregnant) * 유동인구밀도

# RRI (도로 위험성 지수)
RRI = f(교통량, 평균속도, 도로유형, 교차로밀도, 우회전교차로수)

# FGI (시설 공백 지수) -- 역수 개념
FGI = 1 - normalized(CCTV밀도 + 과속방지턱밀도 + 횡단보도접근성 +
                      어린이보호구역커버리지)

# TRI (시간대별 위험 지수)
TRI = 시간대별_사고확률 * 시간대별_취약계층_유동인구비율
```

#### Step 3.3: 가중치 결정 방법

```
방법 1: AHP (Analytic Hierarchy Process)
- 문헌 기반 쌍대비교 행렬 구성
- 일관성 비율(CR) < 0.1 검증

방법 2: 데이터 기반 가중치 (Random Forest Feature Importance)
- 실제 사고 발생 여부를 종속변수로
- 각 하위 지수를 독립변수로
- Feature Importance로 가중치 도출

방법 3: 하이브리드 (추천)
- AHP로 초기 가중치 설정 → 데이터로 보정
- 이론 + 실증의 결합 = 심사위원 호감
```

#### Step 3.4: TSVI 검증
```python
# 기존 4개 신도시에서 검증:
1. TSVI 상위 20% 격자 vs 실제 사고 다발 격자 일치율 (Precision/Recall)
2. ROC-AUC: TSVI의 사고 예측 성능
3. 공간 자기상관 검증: Moran's I
4. 도시별 교차 검증: 3개 도시로 학습 → 1개 도시로 테스트 (Leave-One-City-Out)
```

### Phase 4: 머신러닝 모델링 (Week 4)

#### Step 4.1: 사고 위험도 예측 모델

```python
# 모델 구조:
- 종속변수: 격자별 사고 발생 건수 (또는 이진: 사고 유무)
- 독립변수: TSVI 하위 지표 + 토지이용 + 인구 + 도로 + 시설 변수

# 모델 후보:
1. XGBoost / LightGBM (메인 모델 - 성능 + 해석력)
2. Random Forest (앙상블 비교)
3. Logistic Regression (기준 모델)
4. 공간 가중 회귀(GWR) (공간 이질성 고려)

# 평가:
- 5-fold Spatial Cross-Validation (일반 CV 아닌 공간 CV 필수)
- AUC, F1-Score, RMSE
- SHAP values로 변수 중요도 해석
```

#### Step 4.2: 시설 최적 배치 모델 (핵심 차별화)

```python
# 접근법: 제약 조건 최적화 (Constrained Optimization)

# 목적함수:
maximize Sigma(TSVI_reduction[i]) for all grid i

# 제약조건:
1. 총 시설 설치 예산 <= Budget (현실적 예산 가정)
2. 시설 간 최소 이격 거리
3. 어린이보호구역 우선 배치
4. 기존 시설과의 중복 방지

# 방법:
- 탐욕 알고리즘 (Greedy) + 시뮬레이티드 어닐링
- 또는 선형계획법 (PuLP 라이브러리)
- 시설 유형별: 스마트 횡단보도, 과속카메라, 안전CCTV, 과속방지턱
```

#### Step 4.3: 하남교산 적용 (Transfer Learning 컨셉)

```python
# 핵심 로직:
1. 기존 4개 신도시 격자 데이터로 모델 학습
2. 하남교산 토지이용계획 + 격자 데이터에 적용
3. 하남교산은 아직 미건설 → 인구/교통 데이터 없음

# 해결 전략:
- 하남교산 토지이용계획에서 주거/상업/교육 용도 추출
- 유사한 토지이용 패턴의 기존 신도시 격자를 매칭
- 매칭된 격자의 인구/교통 특성을 하남교산에 전이
- "만약 하남교산이 동탄2처럼 발전한다면" 시나리오 분석
```

### Phase 5: 정량적 효과 추정 (Week 4-5)

#### Step 5.1: 시설별 사고 감소 효과 산출

```python
# 방법론: DID (Difference-in-Differences) + Propensity Score Matching

# 로직:
1. 시설이 설치된 격자 (Treatment) vs 유사하지만 미설치 격자 (Control)
2. PSM으로 비교 가능한 격자 쌍 매칭
3. 시설 설치 전후 사고 변화의 차이 추정

# 보완: 문헌 기반 효과 계수
- 스마트 횡단보도: 보행자 사고 15-25% 감소 (도로교통공단 연구)
- 과속카메라: 속도 초과 30-40% 감소 (경찰청 통계)
- CCTV: 범죄 + 사고 10-20% 감소 (안전행정부 자료)
- 과속방지턱: 통과속도 20-30km/h 감소 (도로교통공단)

# 출력:
- 시설 유형별 사고 감소 기대건수
- 교통량 변화 추정 (속도 감소에 따른 안전 속도 달성률)
- 비용 편익 비율 (B/C Ratio)
```

#### Step 5.2: 시나리오 분석

```python
# 시나리오 설계:
Scenario 1 (기본): 현행 기준 최소 시설 설치
Scenario 2 (권장): TSVI 기반 최적 배치
Scenario 3 (이상적): 전체 고위험 격자 시설 설치

# 각 시나리오별 산출:
- 예상 사고 감소 건수/비율
- 취약계층별 안전도 개선율
- 소요 예산 추정
- B/C Ratio 비교
```

### Phase 6: 시각화 및 보고서 작성 (Week 5-6)

---

## 3. 핵심 분석 방법론 상세

### 3.1 공간통계 방법
| 방법 | 용도 | 라이브러리 |
|------|------|-----------|
| Kernel Density Estimation (KDE) | 사고 밀집도 히트맵 | scipy, sklearn |
| Getis-Ord Gi* | 통계적 유의한 핫스팟 탐지 | PySAL (esda) |
| Moran's I | 공간 자기상관 검정 | PySAL (esda) |
| Spatial Lag Model | 인접 격자 영향 반영 | PySAL (spreg) |
| GWR (지리가중회귀) | 지역별 차별화된 영향 계수 | mgwr |
| DBSCAN | 사고 군집화 | sklearn |

### 3.2 머신러닝 방법
| 방법 | 용도 | 라이브러리 |
|------|------|-----------|
| XGBoost / LightGBM | 사고 위험도 예측 (메인) | xgboost, lightgbm |
| Random Forest | 변수 중요도 + 앙상블 | sklearn |
| SHAP | 모델 해석 (블랙박스 → 설명가능) | shap |
| K-Means / DBSCAN | 격자 유형 분류 | sklearn |
| Bayesian Optimization | 하이퍼파라미터 튜닝 | optuna |
| PSM + DID | 시설 효과 인과추론 | causalml, dowhy |

### 3.3 최적화 방법
| 방법 | 용도 | 라이브러리 |
|------|------|-----------|
| 정수계획법 (ILP) | 시설 최적 배치 | PuLP, OR-Tools |
| Facility Location Problem | 최적 위치 선정 | scipy.optimize |
| Simulated Annealing | 복잡한 제약 조건 최적화 | custom |

---

## 4. 데이터 융합(Cross-Data Fusion) 전략

### 4.1 핵심 융합 테이블

모든 분석의 기본 단위: **100m x 100m 격자(Grid)**

```
[마스터 격자 테이블]
grid_id | region | geometry |
--- 인구 ---
resident_pop_child | resident_pop_elderly | resident_pop_pregnant_est |
floating_pop_hourly | worker_pop | visitor_pop | service_pop_weekday | service_pop_weekend |
--- 도로/교통 ---
road_type | road_length | intersection_count | right_turn_count |
avg_speed | traffic_volume_all | traffic_volume_car | traffic_volume_bus | traffic_volume_truck |
congestion_freq | congestion_intensity |
--- 교육/보호시설 ---
school_count | kindergarten_count | daycare_count | child_protection_zone |
dist_to_nearest_school | dist_to_nearest_crosswalk |
--- 안전시설 ---
cctv_count | speedbump_count | crosswalk_count | busstop_count |
--- 토지이용 ---
land_use_type | building_type | residential_ratio | commercial_ratio |
--- 사고 이력 ---
accident_total | accident_pedestrian | accident_child | accident_elderly |
accident_by_hour | accident_by_weather | accident_severity |
--- 산출 지수 ---
PVI | RRI | FGI | TRI | TSVI
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

#### 융합 5: "교통사고 비용 추정"
```
사고 건수 + 중상/경상/사망 비율 + 교통사고 사회적 비용 계수 (국토부 기준)
→ 격자별 연간 교통사고 사회적 비용 추정
→ 시설 설치 비용 대비 사고 감소 편익 = B/C Ratio
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

### 5.2 QGIS 시각화 (핵심 -- 대회 필수 요구)

| 맵 유형 | 내용 | 기법 |
|---------|------|------|
| TSVI 종합 위험도 맵 | 격자별 색상 그라데이션 | Graduated Symbol |
| 사고 핫스팟 맵 | Gi* 통계량 기반 | Hot/Cold spot |
| 시설 공백 맵 | 안전시설 사각지대 | Buffer + Inverse |
| 어린이 통학 위험 경로 | 통학 경로 위험 구간 | Line graduated |
| 하남교산 예측 맵 | TSVI 예측값 | Graduated (predicted) |
| 최적 배치 제안 맵 | 시설 설치 제안 위치 | Point + Label |
| Before/After 비교 맵 | 시설 설치 전후 TSVI 변화 | Side-by-side / Swipe |
| 시간대별 동적 맵 | 24시간 위험도 변화 | QGIS Temporal Controller |

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

### 6.2 해결: 4단계 전이 분석법

```
[Step 1] 하남교산 토지이용계획 디지털화
- 지구계획 도서에서 토지이용 용도 추출 (주거, 상업, 교육, 녹지 등)
- 100m x 100m 격자에 매핑
- 학교 예정 부지, 도로 예정 노선 추출

[Step 2] 유사 격자 매칭 (Analogous Grid Matching)
- 기존 4개 신도시의 각 격자를 토지이용 유형으로 벡터화
- 하남교산의 각 격자와 코사인 유사도 계산
- 상위 K개 유사 격자의 특성 가중 평균으로 하남교산 특성 추정

[Step 3] 인구/교통 시뮬레이션
- 유사 격자의 인구밀도, 연령구조, 교통량을 하남교산 예상 규모에 맞게 스케일링
- 하남교산 계획 인구(약 10만 가구) 기준으로 보정
- 학교 예정 부지 주변 어린이 인구 집중도 추정

[Step 4] TSVI 예측 및 시설 배치 제안
- 추정된 특성값으로 TSVI 산출
- 고위험 격자에 시설 최적 배치
- 시나리오별 효과 추정
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

### 7.1 사고 감소 효과 추정

```python
# 방법론 1: 회귀 기반 추정 (메인)
# 기존 신도시에서 시설 밀도와 사고율의 관계를 회귀 모델로 추정
# 하남교산에 시설 설치 시 예상 사고율 변화 계산

accident_rate = f(TSVI, facility_density, road_characteristics, population)
delta_accident = predict(with_facility) - predict(without_facility)

# 방법론 2: PSM + 준실험 (보조)
# 시설 설치 격자 vs 미설치 격자 매칭 후 사고율 차이 추정

# 방법론 3: 문헌 기반 계수 적용 (검증)
# 국내외 연구 결과의 사고 감소율을 적용하여 교차 검증
```

### 7.2 교통량 변화 추정

```python
# 속도 감소 → 안전 속도 도달률 변화
# 과속카메라/방지턱 설치 → 평균속도 감소 → 제한속도 준수율 증가

# 시설 유형별 속도 감소 효과 (문헌 기반):
speed_reduction = {
    'speed_camera': -10~-15 km/h,
    'speed_bump': -15~-25 km/h,
    'smart_crosswalk': -5~-10 km/h (경고등 작동 시)
}

# 속도 감소 → 사고 심각도 감소 (Nilsson's Power Model)
# 사망사고 비율 변화 = (V_after / V_before)^4
# 중상사고 비율 변화 = (V_after / V_before)^3
# 경상사고 비율 변화 = (V_after / V_before)^2
```

### 7.3 비용편익분석 (B/C Ratio)

```python
# 비용 (Cost):
facility_cost = {
    'smart_crosswalk': 5000만원/개소,  # 스마트 횡단보도
    'speed_camera': 3000만원/개소,     # 과속단속카메라
    'cctv': 1500만원/대,               # 안전 CCTV
    'speed_bump': 200만원/개소,         # 과속방지턱
    'safety_sign': 50만원/개소          # 안전표지판
}

# 편익 (Benefit):
# 교통사고 사회적 비용 (국토교통부 2024 기준)
accident_social_cost = {
    'death': 약 8.5억원/건,
    'serious_injury': 약 1.2억원/건,
    'minor_injury': 약 1500만원/건,
    'property_damage': 약 500만원/건
}

# B/C Ratio = 연간 사고 감소 편익 / 시설 투자 비용 (감가상각 10년)
# B/C > 1 이면 경제적 타당성 있음
```

---

## 8. 보고서 구조 (발표 심사 60% 최적화)

### 8.1 PPT 보고서 구조 (약 30-35슬라이드)

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

[F: Facility Gap Detection] (5슬라이드)
18. 안전시설 현황 및 공백 분석
19. 시설 효과성 분석 (DID/PSM 결과)
20. 시설 공백과 사고의 상관관계
21. 하남교산 적용: 토지이용 유사 매칭 방법론
22. 하남교산 TSVI 예측 맵

[E: Effect Estimation & Evidence-based] (5슬라이드)
23. 시설 최적 배치 모델 결과
24. 하남교산 맞춤 대안 4가지
25. 정량적 효과 추정 (사고 감소, 교통량 변화)
26. 비용편익분석 결과
27. 시나리오별 비교

[결론] (3슬라이드)
28. 핵심 제언 요약 (1장에 압축)
29. 분석의 의의 및 확장 가능성
30. Q&A / 감사 슬라이드
```

### 8.2 발표 스토리텔링 전략

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
- QGIS 맵과 차트를 교차 활용하여 시각적 설득력 극대화

[결론 및 제언] (3분)
- 하남교산 맞춤 4가지 대안 (구체적 위치 + 예상 효과)
- 비용편익 수치로 현실성 강조
- "아이들의 안전한 통학길을 위한 데이터 기반 솔루션"으로 마무리

[클로징] (30초)
- 감성적 마무리: "데이터가 증명한, 더 안전한 신도시의 설계도"
```

---

## 9. 타임라인 (6주 -- 2026.2.6 ~ 3.20)

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

### Week 3 (2/20 ~ 2/26): EDA 완료 및 TSVI 개발 시작
| 일자 | 작업 | 산출물 |
|------|------|--------|
| 2/20-21 | 심층 EDA: 공간 패턴, 시간 패턴, 취약계층 분석 | EDA 노트북 v2 |
| 2/22-23 | TSVI 하위 지수 설계 및 산출 | TSVI 설계 문서 |
| 2/24-26 | TSVI 가중치 결정 (AHP + RF), 검증 | TSVI 검증 결과 |

### Week 4 (2/27 ~ 3/5): 모델링 및 하남교산 적용
| 일자 | 작업 | 산출물 |
|------|------|--------|
| 2/27-28 | XGBoost 사고 예측 모델 개발 | 모델링 노트북 |
| 3/1-2 | SHAP 해석, 공간CV 검증 | 모델 성능 리포트 |
| 3/3-5 | 하남교산 토지이용 매칭 + TSVI 예측 | 하남교산 예측 맵 |

### Week 5 (3/6 ~ 3/12): 최적 배치 및 효과 추정
| 일자 | 작업 | 산출물 |
|------|------|--------|
| 3/6-7 | 시설 최적 배치 모델 개발 | 최적화 노트북 |
| 3/8-9 | 정량적 효과 추정 + 비용편익분석 | 효과 추정 결과 |
| 3/10-12 | QGIS 시각화 본격 제작 | QGIS 프로젝트 파일 |

### Week 6 (3/13 ~ 3/20): 보고서 작성 및 최종 제출
| 일자 | 작업 | 산출물 |
|------|------|--------|
| 3/13-14 | PPT 보고서 초안 작성 | PPT v1 |
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

## 10. 리스크 관리 및 대응 전략

### 10.1 기술적 리스크

| 리스크 | 확률 | 영향 | 대응 전략 |
|--------|------|------|----------|
| COMPAS 노트북 성능 부족 | 중 | 높음 | 데이터 샘플링, 청크 처리, 필수 분석 우선 |
| 격자 매핑 오류 | 중 | 높음 | CRS 통일 철저, 소규모 테스트 먼저 |
| 비공개 데이터 형식 불일치 | 중 | 중 | 코드북 꼼꼼히 확인, 빠른 EDA로 구조 파악 |
| 모델 성능 미달 | 낮 | 중 | 앙상블/간단한 모델로 대체, 해석력으로 보완 |
| QGIS 시각화 시간 초과 | 중 | 중 | 템플릿 미리 준비, Python으로 대체 가능 |

### 10.2 전략적 리스크

| 리스크 | 확률 | 영향 | 대응 전략 |
|--------|------|------|----------|
| 다른 팀과 접근법 유사 | 높음 | 높음 | TSVI 지수 + 최적화가 차별화 핵심 |
| 하남교산 적용 논리 약함 | 중 | 높음 | 유사 격자 매칭 방법론을 학술적으로 정당화 |
| 발표 시간 초과/부족 | 중 | 중 | 리허설 3회 이상, 시간 배분표 작성 |
| 코드 재현성 실패 | 낮 | 높음 | 환경(requirements.txt), 시드값 고정, 주석 철저 |

### 10.3 Fallback 전략

```
만약 TSVI 모델이 잘 안 되면:
→ 단순 가중합 점수 + 도메인 전문가 기반 가중치로 대체
→ 통계적 검증보다 직관적 설명력에 집중

만약 머신러닝이 성능이 안 나오면:
→ 회귀 분석 + 공간통계만으로도 충분히 경쟁력 있음
→ 모델 복잡도보다 인사이트의 깊이로 승부

만약 시간이 부족하면:
→ QGIS 맵 5장 + PPT 20슬라이드 최소 완성
→ 코드 완성도보다 분석 스토리 완성에 집중
→ 발표 심사 60%이므로 보고서 퀄리티 우선
```

---

## 11. 기술 스택 상세

### 11.1 Python 라이브러리

```python
# 데이터 처리
import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon

# 공간 분석
import pysal
from pysal.explore import esda  # Moran's I, Getis-Ord Gi*
from pysal.model import spreg   # Spatial Regression
from mgwr.gwr import GWR        # Geographically Weighted Regression

# 머신러닝
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import lightgbm as lgb
import shap
import optuna

# 인과추론
from causalml.inference.meta import BaseSRegressor
from dowhy import CausalModel

# 최적화
from pulp import LpProblem, LpMinimize, LpVariable
from scipy.optimize import minimize

# 시각화
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import folium
from folium.plugins import HeatMap, TimestampedGeoJson
import plotly.express as px
import plotly.graph_objects as go

# 기타
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
import warnings
warnings.filterwarnings('ignore')
```

### 11.2 QGIS 작업 목록

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

## 12. 심사위원 관점에서의 체크리스트

### 12.1 논리의 명확성
- [ ] 분석 목표가 명확하게 3가지 이내로 정의되어 있는가?
- [ ] 각 분석 단계가 논리적으로 연결되는가?
- [ ] 결론이 분석 결과에서 자연스럽게 도출되는가?
- [ ] 가정과 한계를 명시적으로 밝혔는가?

### 12.2 가독성
- [ ] 보고서 구조가 일관성 있게 정리되어 있는가?
- [ ] 전문 용어에 대한 설명이 충분한가?
- [ ] 시각 자료가 직관적이고 이해하기 쉬운가?
- [ ] 핵심 메시지가 한 눈에 파악되는가?

### 12.3 데이터 활용성
- [ ] 25개 데이터셋 중 최소 20개 이상 활용했는가?
- [ ] 공개/비공개 데이터를 모두 활용했는가?
- [ ] 창의적인 데이터 융합 분석이 3건 이상인가?
- [ ] 외부 데이터를 보완적으로 활용했는가? (도로교통공단 통계 등)

### 12.4 분석의 창의성
- [ ] 기존에 없던 새로운 지표(TSVI)를 개발했는가?
- [ ] 다른 팀과 차별화되는 분석 방법을 사용했는가?
- [ ] 하남교산 적용 방법이 독창적인가?
- [ ] "와, 이런 분석도 가능하구나" 순간이 있는가?

### 12.5 보고서 충실성
- [ ] PPT 디자인이 전문적이고 일관성 있는가?
- [ ] 모든 차트/맵에 제목, 범례, 출처가 있는가?
- [ ] 발표 스크립트가 자연스럽고 시간에 맞는가?
- [ ] 참고문헌/출처를 충실히 기재했는가?

### 12.6 코드 완성도
- [ ] 코드가 에러 없이 처음부터 끝까지 실행되는가?
- [ ] 주석과 마크다운 설명이 충분한가?
- [ ] 변수명이 직관적이고 코드 구조가 깔끔한가?
- [ ] requirements.txt 또는 환경 설정이 포함되었는가?
- [ ] 시드값이 고정되어 재현 가능한가?

---

## 13. 최종 제출물 체크리스트

```
[필수 제출물]
1. 데이터 파일 (ZIP)
   - 전처리된 데이터셋
   - 마스터 격자 테이블
   - TSVI 산출 결과

2. 소스 코드 (.ipynb)
   - 01_data_preprocessing.ipynb
   - 02_eda_exploration.ipynb
   - 03_tsvi_development.ipynb
   - 04_ml_modeling.ipynb
   - 05_hanam_gyosan_application.ipynb
   - 06_effect_estimation.ipynb
   - 07_optimization.ipynb

3. 분석 보고서 (PPT/PDF)
   - 30-35슬라이드 PPT
   - PDF 변환본

4. 발표 스크립트
   - 13분 내외 발표 원고

5. QGIS 시각화
   - .qgz 프로젝트 파일
   - 내보낸 맵 이미지 (PNG/PDF)
```

---

## 부록: 참고 문헌 및 데이터 출처 (미리 확보)

### 국내 연구/보고서
1. 도로교통공단, "2024 교통사고 통계분석", 2025
2. 국토교통부, "교통사고 사회적 비용 산출 가이드라인", 2024
3. 경찰청, "어린이보호구역 교통사고 현황", 2024
4. 한국교통안전공단, "스마트 교통안전시설 효과분석", 2024
5. LH, "3기 신도시 교통계획 보고서", 2023

### 분석 방법론 참고
6. Nilsson, G. (2004). Traffic safety dimensions and the Power Model
7. Getis, A., & Ord, J. K. (1992). The Analysis of Spatial Association
8. Lundberg, S. M., & Lee, S. I. (2017). SHAP Values
9. Fotheringham, A. S. (2002). Geographically Weighted Regression

---

*이 전략서는 2026.2.6 기준으로 작성되었으며, 분석 진행에 따라 지속 업데이트됩니다.*
