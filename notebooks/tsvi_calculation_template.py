# -*- coding: utf-8 -*-
"""
LH 3기신도시 교통안전 데이터 분석
TSVI (Traffic Safety Vulnerability Index) 산출 모듈

TSVI = w1*PVI + w2*RRI + w3*FGI + w4*TRI

PVI: Population Vulnerability Index (인구 취약성 지수)
RRI: Road Risk Index (도로 위험성 지수)
FGI: Facility Gap Index (시설 공백 지수)
TRI: Temporal Risk Index (시간대별 위험 변동 지수)

모듈 구성:
- TSVICalculator: TSVI 산출 (실제 컬럼 매핑)
- EmpiricalBayesAdjuster: EB 보정 + NB-GLM SPF
- AHPWeightValidator: AHP 가중치 일관성 검증
- MGWRAnalyzer: 다중스케일 지리가중회귀
- TSVIValidator: 종합 검증 스위트
- HanamGyosanTransfer: 하남교산 전이 분석 (Bootstrap CI)
- ScenarioAnalyzer: 시나리오 분석
- CostBenefitAnalyzer: 비용편익 분석
- FacilityOptimizer: PuLP 수리최적화
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
from typing import Dict, List, Optional, Tuple
import logging
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger('lh_traffic_safety')


# ============================================================
# TSVICalculator: TSVI 산출 (실제 컬럼 매핑)
# ============================================================
class TSVICalculator:
    """교통안전 취약성 지수(TSVI) 산출 클래스 - 실제 데이터 컬럼 매핑"""

    def __init__(self, weights=None):
        self.weights = weights or {
            'PVI': 0.30, 'RRI': 0.25, 'FGI': 0.25, 'TRI': 0.20
        }
        self.scaler = MinMaxScaler()
        self.is_fitted = False

    def calculate_pvi(self, grid_df):
        """
        PVI (인구 취약성 지수) 산출.
        실제 컬럼: elderly_ratio(File03), childbearing_ratio(File03),
        child_pop_estimated(File16/17 프록시), floating_pop_density(File04)
        """
        # 취약계층 가중 인구 비율
        elderly = self._safe_col(grid_df, 'elderly_ratio', 0)
        childbearing = self._safe_col(grid_df, 'childbearing_ratio', 0)
        child_pop = self._safe_col(grid_df, 'child_pop_estimated', 0)

        vulnerable_ratio = (
            self._normalize(child_pop) * 0.40 +
            elderly * 0.35 +
            childbearing * 0.25
        )

        # 유동인구 밀도와 결합
        pop_density = self._safe_col(grid_df, 'floating_pop_density', 0)
        pop_density_norm = self._normalize(pop_density)
        pvi = vulnerable_ratio * (1 + pop_density_norm)

        return self._normalize(pvi)

    def calculate_rri(self, grid_df):
        """
        RRI (도로 위험성 지수) 산출.
        실제 컬럼: traffic_volume_all(ALL_AADT, File10), avg_speed(velocity_AVRG, File09),
        max_speed_max(File08), road_length_total(File08), intersection_count(File08 노드),
        congestion_freq(FRIN_CG, File11), congestion_time_intensity(TI_CG, File12)
        """
        # 교통량 위험도
        traffic = self._safe_col(grid_df, 'traffic_volume_all', 0)
        traffic_risk = self._normalize(traffic)

        # 속도 위험도 (30km/h 초과분)
        speed = self._safe_col(grid_df, 'avg_speed', 0)
        speed_risk = self._normalize(np.maximum(speed - 30, 0))

        # 도로 연장 (도로가 길수록 노출 증가)
        road_len = self._safe_col(grid_df, 'road_length_total', 0)
        road_risk = self._normalize(road_len)

        # 교차로 밀도
        intersection = self._safe_col(grid_df, 'intersection_count', 0)
        intersection_risk = self._normalize(intersection)

        # 혼잡도 복합 (빈도 x 시간강도)
        cong_freq = self._safe_col(grid_df, 'congestion_freq', 0)
        cong_time = self._safe_col(grid_df, 'congestion_time_intensity', 0)
        congestion_risk = self._normalize(cong_freq * cong_time)

        rri = (
            traffic_risk * 0.25 +
            speed_risk * 0.25 +
            road_risk * 0.10 +
            intersection_risk * 0.20 +
            congestion_risk * 0.20
        )

        return self._normalize(rri)

    def calculate_fgi(self, grid_df):
        """
        FGI (시설 공백 지수) 산출. 값이 높을수록 안전시설 부족.
        거리 가중 점수: 역거리 함수 1 - 1/(1+dist/ref_dist)
        """
        # 시설 개수 기반 충족도
        cctv = self._safe_col(grid_df, 'cctv_count', 0)
        bump = self._safe_col(grid_df, 'speed_bump_count', 0)
        crosswalk = self._safe_col(grid_df, 'crosswalk_count', 0)
        child_zone = self._safe_col(grid_df, 'child_protection_zone_count', 0)

        count_coverage = (
            self._normalize(cctv) * 0.25 +
            self._normalize(bump) * 0.15 +
            self._normalize(crosswalk) * 0.20 +
            self._normalize(np.minimum(child_zone, 1)) * 0.10
        )

        # 거리 가중 점수: 가까울수록 높은 접근성 (역거리 함수)
        ref_dist = 300  # 참조거리 300m
        dist_score = 0
        n_dist = 0
        for dist_col in ['dist_nearest_crosswalks', 'dist_nearest_schools',
                         'dist_nearest_bus_stops', 'dist_nearest_cctv']:
            if dist_col in grid_df.columns:
                d = grid_df[dist_col].fillna(grid_df[dist_col].max())
                accessibility = 1.0 / (1.0 + d / ref_dist)
                dist_score = dist_score + accessibility
                n_dist += 1

        if n_dist > 0:
            dist_coverage = self._normalize(dist_score / n_dist)
            facility_coverage = count_coverage * 0.60 + dist_coverage * 0.40
        else:
            facility_coverage = count_coverage

        # 공백도 = 1 - 충족도
        fgi = 1 - self._normalize(facility_coverage)
        return self._normalize(fgi)

    def calculate_tri(self, grid_df):
        """
        TRI (시간대별 위험 변동 지수) 산출.
        시간대 변동성: pop_hourly_cv(TMST_00~23 변동계수), peak_nonpeak_ratio,
        accident_peak_ratio, congestion 정보
        """
        # 유동인구 시간대 변동성
        hourly_cv = self._safe_col(grid_df, 'worker_hourly_cv', 0)
        pop_var_risk = self._normalize(hourly_cv)

        # 피크/비피크 비율
        peak_ratio = self._safe_col(grid_df, 'worker_peak_ratio', 1)
        peak_risk = self._normalize(peak_ratio)

        # 혼잡 빈도 (시간대 위험의 proxy)
        cong_freq = self._safe_col(grid_df, 'congestion_freq', 0)
        cong_risk = self._normalize(cong_freq)

        tri = (
            pop_var_risk * 0.35 +
            peak_risk * 0.35 +
            cong_risk * 0.30
        )

        return self._normalize(tri)

    def calculate_tsvi(self, grid_df):
        """TSVI 종합 산출"""
        result = grid_df.copy()

        result['PVI'] = self.calculate_pvi(grid_df)
        result['RRI'] = self.calculate_rri(grid_df)
        result['FGI'] = self.calculate_fgi(grid_df)
        result['TRI'] = self.calculate_tri(grid_df)

        result['TSVI'] = (
            result['PVI'] * self.weights['PVI'] +
            result['RRI'] * self.weights['RRI'] +
            result['FGI'] * self.weights['FGI'] +
            result['TRI'] * self.weights['TRI']
        )

        result['TSVI_score'] = self._normalize(result['TSVI']) * 100

        result['TSVI_grade'] = pd.cut(
            result['TSVI_score'],
            bins=[0, 20, 40, 60, 80, 100],
            labels=['안전', '주의', '경계', '위험', '심각'],
            include_lowest=True
        )

        self.is_fitted = True
        logger.info(f"TSVI 산출 완료: {len(result)} 격자")
        print(f"[INFO] TSVI 산출 완료: {len(result)} 격자")
        print(f"  등급 분포:\n{result['TSVI_grade'].value_counts().to_string()}")

        return result

    def optimize_weights_with_data(self, grid_df, accident_col='accident_total'):
        """실제 사고 데이터 기반 가중치 최적화"""
        features = ['PVI', 'RRI', 'FGI', 'TRI']
        X = grid_df[features].fillna(0)
        y = (grid_df[accident_col] > 0).astype(int)

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)

        importances = rf.feature_importances_
        new_weights = dict(zip(features, importances / importances.sum()))

        print(f"\n[INFO] 데이터 기반 가중치 최적화 결과:")
        print(f"  기존 가중치: {self.weights}")
        print(f"  최적 가중치: {new_weights}")

        return new_weights, rf

    def optimize_weights_hybrid(self, grid_df, ahp_weights: Dict[str, float],
                                accident_col='accident_total',
                                alpha: float = 0.5) -> Dict[str, float]:
        """
        AHP(CR<0.10 검증됨) + RF Feature Importance 하이브리드.
        w_hybrid = alpha * w_ahp + (1-alpha) * w_data
        """
        data_weights, rf = self.optimize_weights_with_data(grid_df, accident_col)
        features = ['PVI', 'RRI', 'FGI', 'TRI']

        hybrid = {}
        for key in features:
            hybrid[key] = alpha * ahp_weights.get(key, 0.25) + (1 - alpha) * data_weights[key]

        total = sum(hybrid.values())
        hybrid = {k: round(v / total, 4) for k, v in hybrid.items()}

        print(f"  하이브리드 가중치 (alpha={alpha}): {hybrid}")
        self.weights = hybrid
        return hybrid

    @staticmethod
    def _normalize(series):
        """Min-Max 정규화 (0~1)"""
        if isinstance(series, (int, float)):
            return series
        min_val = series.min()
        max_val = series.max()
        if max_val - min_val < 1e-8:
            return pd.Series(0, index=series.index)
        return (series - min_val) / (max_val - min_val)

    @staticmethod
    def _safe_col(df, col, default=0):
        """컬럼이 없으면 default 반환"""
        if col in df.columns:
            return df[col].fillna(default)
        return pd.Series(default, index=df.index)


# ============================================================
# EmpiricalBayesAdjuster: EB 보정 + NB-GLM SPF
# ============================================================
class EmpiricalBayesAdjuster:
    """
    Empirical Bayes 보정기.
    Negative Binomial GLM을 Safety Performance Function(SPF)으로 적합하고,
    소표본 격자의 극단값을 평균 방향으로 수축(shrinkage)한다.

    참고: Hauer(1997), Persaud & Lyon(1999)
    """

    def __init__(self):
        self.model = None
        self.k_dispersion = None
        self.is_fitted = False

    def fit_spf(self, grid_df: pd.DataFrame,
                accident_col: str = 'accident_total',
                predictors: List[str] = None,
                alpha_init: float = 1.0) -> 'EmpiricalBayesAdjuster':
        """
        Negative Binomial GLM을 Safety Performance Function으로 적합.

        Y_i ~ NB(mu_i, k)
        log(mu_i) = X * beta

        Parameters
        ----------
        accident_col : str
            종속변수 (사고건수)
        predictors : list
            독립변수 리스트
        alpha_init : float
            NB 분산 파라미터 초기값
        """
        import statsmodels.api as sm

        if predictors is None:
            predictors = ['traffic_volume_all', 'road_length_total',
                          'floating_pop_density', 'intersection_count', 'avg_speed']

        avail = [p for p in predictors if p in grid_df.columns]
        if not avail:
            logger.warning("SPF 독립변수 없음 - 적합 불가")
            return self

        df_clean = grid_df[avail + [accident_col]].dropna()
        X = sm.add_constant(df_clean[avail].astype(float))
        y = df_clean[accident_col].astype(float)

        # NB GLM 적합
        try:
            nb_model = sm.GLM(y, X, family=sm.families.NegativeBinomial(alpha=alpha_init))
            self.model = nb_model.fit()
            self.k_dispersion = 1.0 / self.model.scale  # 과산포 파라미터
            self.is_fitted = True
            logger.info(f"NB-GLM SPF 적합 완료: k={self.k_dispersion:.4f}, "
                         f"AIC={self.model.aic:.1f}")
            print(f"[INFO] NB-GLM SPF 적합 완료")
            print(f"  과산포 파라미터 k = {self.k_dispersion:.4f}")
            print(f"  AIC = {self.model.aic:.1f}")
            print(f"  유의한 변수: {[p for p, pv in zip(avail, self.model.pvalues[1:]) if pv < 0.05]}")
        except Exception as e:
            logger.error(f"NB-GLM 적합 실패: {e}")
            # Fallback: 포아송 GLM
            try:
                pois_model = sm.GLM(y, X, family=sm.families.Poisson())
                self.model = pois_model.fit()
                self.k_dispersion = self.model.scale
                self.is_fitted = True
                logger.info("Poisson GLM으로 대체 적합")
            except Exception as e2:
                logger.error(f"Poisson GLM도 실패: {e2}")

        return self

    def predict_expected(self, grid_df: pd.DataFrame,
                         predictors: List[str] = None) -> pd.Series:
        """SPF 기반 기대 사고건수 예측"""
        import statsmodels.api as sm

        if not self.is_fitted:
            logger.warning("SPF 미적합 - 전체 평균 반환")
            return pd.Series(grid_df.get('accident_total', pd.Series(0)).mean(),
                             index=grid_df.index)

        if predictors is None:
            predictors = ['traffic_volume_all', 'road_length_total',
                          'floating_pop_density', 'intersection_count', 'avg_speed']

        avail = [p for p in predictors if p in grid_df.columns]
        X = sm.add_constant(grid_df[avail].fillna(0).astype(float))

        try:
            expected = self.model.predict(X)
        except Exception:
            expected = pd.Series(0, index=grid_df.index)

        return expected

    def adjust(self, observed: pd.Series, expected: pd.Series) -> pd.Series:
        """
        EB 보정 적용.

        EB_estimate = w * observed + (1 - w) * expected
        w = 1 / (1 + expected / k_dispersion)

        소표본 격자의 극단값을 전체 평균 방향으로 수축.
        """
        k = self.k_dispersion if self.k_dispersion else 1.0
        w = 1.0 / (1.0 + expected / k)
        eb_estimate = w * observed + (1 - w) * expected

        logger.info(f"EB 보정 완료: 평균 수축 비율 w = {w.mean():.4f}")
        print(f"[INFO] EB 보정: 평균 가중치 w={w.mean():.4f}, "
              f"원시 범위=[{observed.min():.1f}, {observed.max():.1f}], "
              f"EB 범위=[{eb_estimate.min():.1f}, {eb_estimate.max():.1f}]")

        return eb_estimate

    def compute_excess(self, observed: pd.Series, expected: pd.Series) -> pd.Series:
        """
        초과 사고건수 산출 (구조적 결함 지표).
        excess > 0: 구조적으로 위험한 격자
        excess < 0: 기대보다 안전한 격자
        """
        eb = self.adjust(observed, expected)
        return eb - expected


# ============================================================
# AHPWeightValidator: AHP 가중치 일관성 검증
# ============================================================
class AHPWeightValidator:
    """
    AHP(Analytic Hierarchy Process) 가중치 산출 및 일관성 검증.
    CR(일관성 비율) < 0.10 자동 검증.
    """

    # 무작위 지수표 (Saaty, 1980)
    RI_TABLE = {1: 0, 2: 0, 3: 0.58, 4: 0.90, 5: 1.12,
                6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

    def compute_weights(self, comparison_matrix: np.ndarray,
                        criteria_names: List[str]) -> Dict:
        """
        쌍대비교 행렬 → 고유값법 → 우선순위 벡터.

        Parameters
        ----------
        comparison_matrix : np.ndarray
            n x n 쌍대비교 행렬 (대각선 = 1)
        criteria_names : list
            기준 이름 리스트

        Returns
        -------
        dict with keys: 'weights', 'CR', 'CI', 'lambda_max', 'is_consistent'
        """
        n = len(comparison_matrix)
        # 고유값 분해
        eigenvalues, eigenvectors = np.linalg.eig(comparison_matrix)
        max_idx = np.argmax(eigenvalues.real)
        lambda_max = eigenvalues.real[max_idx]
        priority_vector = eigenvectors[:, max_idx].real
        priority_vector = priority_vector / priority_vector.sum()

        # 일관성 지수
        CI = (lambda_max - n) / (n - 1) if n > 1 else 0
        RI = self.RI_TABLE.get(n, 1.49)
        CR = CI / RI if RI > 0 else 0

        weights = dict(zip(criteria_names, np.round(priority_vector, 4)))
        is_consistent = CR < 0.10

        result = {
            'weights': weights,
            'CR': round(CR, 4),
            'CI': round(CI, 4),
            'lambda_max': round(lambda_max, 4),
            'is_consistent': is_consistent
        }

        status = "PASS" if is_consistent else "FAIL"
        print(f"[AHP] CR = {CR:.4f} ({status})")
        print(f"  가중치: {weights}")

        if not is_consistent:
            print(f"  [경고] CR >= 0.10 → 쌍대비교 행렬의 일관성 부족. 행렬 재검토 필요.")

        return result

    def sensitivity_analysis(self, comparison_matrix: np.ndarray,
                              criteria_names: List[str],
                              grid_df: pd.DataFrame,
                              perturbation: float = 0.10,
                              n_trials: int = 100) -> pd.DataFrame:
        """
        가중치 교란 → 순위 안정성 평가.
        각 시행에서 가중치를 ±perturbation 범위로 랜덤 교란하고
        TSVI 상위 20% 격자의 순위 변동을 측정.
        """
        np.random.seed(42)
        base_result = self.compute_weights(comparison_matrix, criteria_names)
        base_weights = base_result['weights']

        # 기준 TSVI 순위
        features = criteria_names
        base_tsvi = sum(grid_df[f].fillna(0) * base_weights[f] for f in features)
        base_top20 = set(base_tsvi.nlargest(int(len(base_tsvi) * 0.2)).index)

        stability_scores = []
        for _ in range(n_trials):
            # 가중치 교란
            perturbed = {}
            for f in features:
                noise = np.random.uniform(-perturbation, perturbation)
                perturbed[f] = max(0.01, base_weights[f] * (1 + noise))
            total = sum(perturbed.values())
            perturbed = {k: v / total for k, v in perturbed.items()}

            # 교란 TSVI
            pert_tsvi = sum(grid_df[f].fillna(0) * perturbed[f] for f in features)
            pert_top20 = set(pert_tsvi.nlargest(int(len(pert_tsvi) * 0.2)).index)

            # Jaccard 유사도
            jaccard = len(base_top20 & pert_top20) / len(base_top20 | pert_top20)
            stability_scores.append(jaccard)

        mean_stability = np.mean(stability_scores)
        print(f"[민감도 분석] 가중치 ±{perturbation * 100:.0f}% 교란 ({n_trials}회)")
        print(f"  상위 20% 순위 안정성 (Jaccard): {mean_stability:.4f}")

        return pd.DataFrame({
            'trial': range(n_trials),
            'jaccard_stability': stability_scores
        })


# ============================================================
# MGWRAnalyzer: 다중스케일 지리가중회귀
# ============================================================
class MGWRAnalyzer:
    """
    MGWR(Multiscale Geographically Weighted Regression) 분석기.
    변수별 독립적인 bandwidth로 공간적 영향 범위 차이를 포착.

    참고: Fotheringham et al.(2017), Oshan et al.(2019)
    """

    def __init__(self):
        self.model = None
        self.bandwidths = None
        self.selector = None

    def fit(self, coords: np.ndarray, y: np.ndarray, X: np.ndarray,
            kernel: str = 'bisquare', fixed: bool = False) -> 'MGWRAnalyzer':
        """
        MGWR 적합.

        Parameters
        ----------
        coords : np.ndarray (n, 2)
            격자 중심 좌표 (UTM)
        y : np.ndarray (n, 1)
            종속변수 (예: accident_total 또는 TSVI)
        X : np.ndarray (n, k)
            독립변수 행렬
        """
        try:
            from mgwr.sel_bw import Sel_BW
            from mgwr.gwr import MGWR

            y_reshaped = y.reshape(-1, 1) if y.ndim == 1 else y
            selector = Sel_BW(coords, y_reshaped, X, multi=True,
                              kernel=kernel, fixed=fixed)
            self.bandwidths = selector.search()
            self.selector = selector

            self.model = MGWR(coords, y_reshaped, X,
                              selector=selector,
                              kernel=kernel, fixed=fixed).fit()

            logger.info(f"MGWR 적합 완료: bandwidths = {self.bandwidths}")
            print(f"[INFO] MGWR 적합 완료")
            print(f"  변수별 bandwidth: {self.bandwidths}")
            print(f"  AICc = {self.model.aicc:.1f}")

        except ImportError:
            logger.error("mgwr 패키지 미설치. pip install mgwr")
        except Exception as e:
            logger.error(f"MGWR 적합 실패: {e}")
            # Fallback: 일반 GWR
            try:
                from mgwr.sel_bw import Sel_BW
                from mgwr.gwr import GWR

                y_reshaped = y.reshape(-1, 1) if y.ndim == 1 else y
                sel = Sel_BW(coords, y_reshaped, X, kernel=kernel, fixed=fixed)
                bw = sel.search()
                self.model = GWR(coords, y_reshaped, X, bw,
                                 kernel=kernel, fixed=fixed).fit()
                self.bandwidths = [bw] * X.shape[1]
                logger.info(f"GWR(단일 bandwidth) 대체 적합: bw={bw}")
            except Exception as e2:
                logger.error(f"GWR 대체도 실패: {e2}")

        return self

    def get_coefficient_surface(self, variable_idx: int) -> Optional[np.ndarray]:
        """특정 변수의 공간적 계수 변이 추출"""
        if self.model is None:
            return None
        return self.model.params[:, variable_idx]

    def identify_spatial_regimes(self, n_regimes: int = 4) -> Optional[np.ndarray]:
        """계수 프로파일 기반 격자 군집화"""
        if self.model is None:
            return None

        from sklearn.cluster import KMeans

        coefficients = self.model.params
        kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
        labels = kmeans.fit_predict(coefficients)

        print(f"[MGWR] 공간 레짐 {n_regimes}개 식별:")
        for i in range(n_regimes):
            count = (labels == i).sum()
            print(f"  레짐 {i}: {count}개 격자 ({count / len(labels) * 100:.1f}%)")

        return labels


# ============================================================
# TSVIValidator: 종합 검증 스위트
# ============================================================
class TSVIValidator:
    """
    TSVI 종합 검증 스위트.
    공간 CV, 보정 분석, Lift 차트, Moran's I 등.
    """

    def __init__(self, grid_df: pd.DataFrame, accident_col: str = 'accident_total'):
        self.grid_df = grid_df
        self.accident_col = accident_col

    def validate_full(self) -> Dict:
        """전체 검증 스위트 실행"""
        results = {}
        results['classification'] = self._basic_classification_metrics()
        results['calibration'] = self._calibration_analysis()
        results['lift'] = self._lift_chart_data()
        results['moran'] = self._spatial_autocorrelation()
        return results

    def _basic_classification_metrics(self) -> Dict:
        """AUC, Precision/Recall/F1 @ top20%, 사고 포착률"""
        df = self.grid_df.copy()
        if self.accident_col not in df.columns or 'TSVI_score' not in df.columns:
            return {'error': '필요 컬럼 없음'}

        y_true = (df[self.accident_col] > 0).astype(int)
        threshold = df['TSVI_score'].quantile(0.80)
        y_pred = (df['TSVI_score'] >= threshold).astype(int)

        metrics = {
            'AUC': float(roc_auc_score(y_true, df['TSVI_score'])),
            'Precision_top20': float(precision_score(y_true, y_pred, zero_division=0)),
            'Recall_top20': float(recall_score(y_true, y_pred, zero_division=0)),
            'F1_top20': float(f1_score(y_true, y_pred, zero_division=0)),
            'Accident_capture_rate': float(
                y_true[y_pred == 1].sum() / max(y_true.sum(), 1)
            )
        }

        print(f"[검증] 분류 메트릭:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")
        return metrics

    def _spatial_cross_validation(self, region_col: str = 'region') -> Dict:
        """Leave-One-City-Out AUC"""
        df = self.grid_df
        if region_col not in df.columns:
            return {'error': 'region 컬럼 없음'}

        regions = [r for r in df[region_col].unique() if 'gyosan' not in str(r).lower()]
        city_aucs = {}

        for test_city in regions:
            test_mask = df[region_col] == test_city
            test_df = df[test_mask]

            y_true = (test_df[self.accident_col] > 0).astype(int)
            if y_true.sum() == 0 or y_true.sum() == len(y_true):
                continue

            auc = roc_auc_score(y_true, test_df['TSVI_score'])
            city_aucs[test_city] = auc

        avg_auc = np.mean(list(city_aucs.values())) if city_aucs else 0

        print(f"[검증] Leave-One-City-Out AUC:")
        for city, auc in city_aucs.items():
            print(f"  {city}: {auc:.4f}")
        print(f"  평균: {avg_auc:.4f}")

        return {'city_aucs': city_aucs, 'mean_auc': avg_auc}

    def _calibration_analysis(self, n_bins: int = 10) -> Dict:
        """10분위 예측 vs 관측, Spearman 상관"""
        from scipy.stats import spearmanr

        df = self.grid_df.copy()
        if self.accident_col not in df.columns or 'TSVI_score' not in df.columns:
            return {'error': '필요 컬럼 없음'}

        df['tsvi_decile'] = pd.qcut(df['TSVI_score'], n_bins, labels=False,
                                     duplicates='drop')

        calibration = df.groupby('tsvi_decile').agg(
            predicted_mean=('TSVI_score', 'mean'),
            observed_rate=(self.accident_col, lambda x: (x > 0).mean()),
            observed_mean=(self.accident_col, 'mean'),
            n_grids=('TSVI_score', 'count')
        ).reset_index()

        rho, pval = spearmanr(calibration['predicted_mean'],
                               calibration['observed_mean'])

        result = {
            'calibration_table': calibration,
            'spearman_rho': float(rho),
            'spearman_pval': float(pval)
        }

        print(f"[검증] 보정 분석: Spearman rho = {rho:.4f} (p={pval:.4f})")
        return result

    def _lift_chart_data(self) -> Dict:
        """상위 k%의 누적 사고 포착률, lift@20%"""
        df = self.grid_df.copy()
        if self.accident_col not in df.columns or 'TSVI_score' not in df.columns:
            return {'error': '필요 컬럼 없음'}

        df_sorted = df.sort_values('TSVI_score', ascending=False)
        total_accidents = df[self.accident_col].sum()

        if total_accidents == 0:
            return {'error': '사고 데이터 없음'}

        percentages = [5, 10, 15, 20, 25, 30, 50]
        lift_data = {}

        for pct in percentages:
            n = int(len(df_sorted) * pct / 100)
            captured = df_sorted.head(n)[self.accident_col].sum()
            capture_rate = captured / total_accidents
            lift = capture_rate / (pct / 100)
            lift_data[f'top_{pct}pct'] = {
                'capture_rate': float(capture_rate),
                'lift': float(lift)
            }

        print(f"[검증] Lift 차트:")
        for k, v in lift_data.items():
            print(f"  {k}: 포착률={v['capture_rate']:.4f}, lift={v['lift']:.2f}")
        return lift_data

    def _spatial_autocorrelation(self) -> Dict:
        """Moran's I 검정 (Queen 인접)"""
        try:
            from esda.moran import Moran
            from libpysal.weights import Queen

            df = self.grid_df
            if 'TSVI_score' not in df.columns:
                return {'error': 'TSVI_score 없음'}
            if not hasattr(df, 'geometry'):
                return {'error': 'geometry 없음'}

            w = Queen.from_dataframe(df)
            w.transform = 'r'

            moran = Moran(df['TSVI_score'].values, w)

            result = {
                'I': float(moran.I),
                'p_value': float(moran.p_sim),
                'z_score': float(moran.z_sim)
            }
            print(f"[검증] Moran's I = {moran.I:.4f} (p={moran.p_sim:.4f})")
            return result

        except ImportError:
            return {'error': 'esda/libpysal 미설치'}
        except Exception as e:
            return {'error': str(e)}


# ============================================================
# HanamGyosanTransfer: 하남교산 전이 분석
# ============================================================
class HanamGyosanTransfer:
    """하남교산 전이 분석 클래스 (Bootstrap CI 포함)"""

    def __init__(self, existing_grids_df):
        self.existing = existing_grids_df
        self.land_use_features = [
            'residential_ratio', 'commercial_ratio', 'industrial_ratio',
            'green_ratio', 'education_ratio', 'road_ratio'
        ]

    def prepare_land_use_vectors(self, land_use_df: pd.DataFrame,
                                  zone_name_col: str = 'zoneName') -> pd.DataFrame:
        """실제 zoneCode/zoneName → 토지이용 분류"""
        if zone_name_col in land_use_df.columns:
            land_use_df['lu_category'] = land_use_df[zone_name_col].apply(
                lambda x: self._classify_simple(x))
        return land_use_df

    @staticmethod
    def _classify_simple(zone_name):
        """간단한 용도 분류"""
        if pd.isna(zone_name):
            return 'unknown'
        name = str(zone_name)
        if '주거' in name:
            return 'residential'
        elif '상업' in name:
            return 'commercial'
        elif '공업' in name:
            return 'industrial'
        elif '녹지' in name or '공원' in name:
            return 'green'
        elif '학교' in name or '교육' in name:
            return 'education'
        elif '도로' in name:
            return 'road'
        return 'other'

    def match_similar_grids(self, gyosan_grid_df, top_k=5,
                            metric='cosine') -> pd.DataFrame:
        """
        하남교산 격자와 유사한 기존 신도시 격자 매칭.
        StandardScaler 정규화, 다중 유사도 메트릭 지원.
        """
        from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

        avail_features = [f for f in self.land_use_features
                          if f in self.existing.columns and f in gyosan_grid_df.columns]

        if not avail_features:
            logger.warning("토지이용 피처 없음 - 매칭 불가")
            return gyosan_grid_df

        # StandardScaler 정규화
        scaler = StandardScaler()
        existing_vectors = scaler.fit_transform(
            self.existing[avail_features].fillna(0).values)
        gyosan_vectors = scaler.transform(
            gyosan_grid_df[avail_features].fillna(0).values)

        # 유사도 계산
        if metric == 'cosine':
            similarities = cosine_similarity(gyosan_vectors, existing_vectors)
        elif metric == 'euclidean':
            dists = euclidean_distances(gyosan_vectors, existing_vectors)
            similarities = 1.0 / (1.0 + dists)
        else:
            similarities = cosine_similarity(gyosan_vectors, existing_vectors)

        # 전이 대상 컬럼
        exclude = set(self.land_use_features + ['geometry', 'grid_id', 'region',
                                                 'gid', 'gbn', 'centroid_x', 'centroid_y'])
        transfer_cols = [c for c in self.existing.columns if c not in exclude]

        estimated_values = {col: [] for col in transfer_cols}

        for i in range(len(gyosan_grid_df)):
            top_k_idx = np.argsort(similarities[i])[-top_k:]
            top_k_sims = similarities[i][top_k_idx]
            weights = top_k_sims / (top_k_sims.sum() + 1e-8)

            for col in transfer_cols:
                values = self.existing.iloc[top_k_idx][col]
                if pd.api.types.is_numeric_dtype(values):
                    estimated_values[col].append(np.average(values.fillna(0), weights=weights))
                else:
                    estimated_values[col].append(values.mode().iloc[0] if len(values.mode()) > 0 else None)

        result = gyosan_grid_df.copy()
        for col, vals in estimated_values.items():
            result[f'{col}_estimated'] = vals

        max_sims = similarities.max(axis=1)
        result['max_similarity'] = max_sims
        result['avg_similarity_top_k'] = np.sort(similarities, axis=1)[:, -top_k:].mean(axis=1)

        print(f"[INFO] 하남교산 유사 격자 매칭 완료 (metric={metric}):")
        print(f"  평균 최고 유사도: {max_sims.mean():.4f}")
        print(f"  유사도 0.8 이상: {(max_sims >= 0.8).sum()} / {len(gyosan_grid_df)}")

        return result

    def transfer_with_bootstrap_ci(self, gyosan_grid_df: pd.DataFrame,
                                    target_col: str = 'TSVI_score',
                                    top_k: int = 5,
                                    n_bootstrap: int = 1000,
                                    ci_level: float = 0.95) -> pd.DataFrame:
        """
        Bootstrap 리샘플링 → 95% CI, CV 산출.

        B=1000회, CI_95% = [TSVI*(0.025), TSVI*(0.975)]
        """
        from sklearn.metrics.pairwise import cosine_similarity

        avail_features = [f for f in self.land_use_features
                          if f in self.existing.columns and f in gyosan_grid_df.columns]

        if not avail_features or target_col not in self.existing.columns:
            logger.warning("Bootstrap 전이 불가")
            return gyosan_grid_df

        existing_vectors = self.existing[avail_features].fillna(0).values
        gyosan_vectors = gyosan_grid_df[avail_features].fillna(0).values
        similarities = cosine_similarity(gyosan_vectors, existing_vectors)

        alpha = 1 - ci_level
        results = {
            f'{target_col}_mean': [],
            f'{target_col}_ci_lower': [],
            f'{target_col}_ci_upper': [],
            f'{target_col}_cv': []
        }

        np.random.seed(42)

        for i in range(len(gyosan_grid_df)):
            top_k_idx = np.argsort(similarities[i])[-top_k:]
            top_k_values = self.existing.iloc[top_k_idx][target_col].values
            top_k_sims = similarities[i][top_k_idx]
            weights = top_k_sims / (top_k_sims.sum() + 1e-8)

            bootstrap_means = []
            for _ in range(n_bootstrap):
                boot_idx = np.random.choice(len(top_k_values), size=len(top_k_values), replace=True)
                boot_values = top_k_values[boot_idx]
                boot_weights = weights[boot_idx]
                boot_weights = boot_weights / (boot_weights.sum() + 1e-8)
                bootstrap_means.append(np.average(boot_values, weights=boot_weights))

            bootstrap_means = np.array(bootstrap_means)
            results[f'{target_col}_mean'].append(float(np.mean(bootstrap_means)))
            results[f'{target_col}_ci_lower'].append(float(np.percentile(bootstrap_means, alpha / 2 * 100)))
            results[f'{target_col}_ci_upper'].append(float(np.percentile(bootstrap_means, (1 - alpha / 2) * 100)))
            cv = np.std(bootstrap_means) / (np.mean(bootstrap_means) + 1e-8)
            results[f'{target_col}_cv'].append(float(cv))

        result = gyosan_grid_df.copy()
        for col, vals in results.items():
            result[col] = vals

        mean_cv = np.mean(results[f'{target_col}_cv'])
        mean_width = np.mean(
            np.array(results[f'{target_col}_ci_upper']) -
            np.array(results[f'{target_col}_ci_lower'])
        )

        print(f"[INFO] Bootstrap 전이 ({n_bootstrap}회):")
        print(f"  평균 CV = {mean_cv:.4f}")
        print(f"  평균 CI 폭 = {mean_width:.4f}")
        print(f"  CI 커버리지 = {ci_level * 100:.0f}%")

        return result

    def estimate_population_enhanced(self, gyosan_grid_df,
                                      planned_households=100000,
                                      match_city: str = None):
        """
        토지이용 비례 인구 배분 + 매칭 도시 인구구조 활용.
        """
        avg_persons_per_household = 2.5
        planned_pop = planned_households * avg_persons_per_household

        # 주거지 비율 기반 배분
        res_ratio = gyosan_grid_df.get('residential_ratio',
                                        pd.Series(0, index=gyosan_grid_df.index))
        total_res = res_ratio.sum()

        result = gyosan_grid_df.copy()
        if total_res > 0:
            result['estimated_pop'] = res_ratio / total_res * planned_pop
        else:
            result['estimated_pop'] = planned_pop / len(result)

        # 매칭 도시의 인구구조(연령비율) 적용
        if match_city and 'region' in self.existing.columns:
            city_data = self.existing[self.existing['region'] == match_city]
            if len(city_data) > 0:
                for ratio_col in ['elderly_ratio', 'childbearing_ratio']:
                    if ratio_col in city_data.columns:
                        city_avg = city_data[ratio_col].mean()
                        result[f'{ratio_col}_estimated'] = city_avg
                        print(f"  {match_city} {ratio_col} 평균: {city_avg:.4f}")

        print(f"[INFO] 하남교산 인구 추정: 총 {planned_pop:,.0f}명")
        return result


# ============================================================
# ScenarioAnalyzer: 시나리오 분석
# ============================================================
class ScenarioAnalyzer:
    """시나리오별 사고 감소 추정"""

    # 문헌 기반 사고 감소율 (CMF = 1 - reduction_rate)
    FACILITY_EFFECTS = {
        'cctv': 0.15,                # CCTV 15% 감소
        'speed_bump': 0.20,          # 과속방지턱 20% 감소
        'smart_crosswalk': 0.22,     # 스마트횡단보도 22% 감소
        'speed_camera': 0.30,        # 과속카메라 30% 감소
        'safety_sign': 0.05          # 안전표지판 5% 감소
    }

    # Nilsson Power Model: 속도 감소 → 사고 심각도 변화
    SPEED_REDUCTION = {
        'speed_camera': 12,     # km/h 감소
        'speed_bump': 20,       # km/h 감소
        'smart_crosswalk': 7    # km/h 감소
    }

    def __init__(self, config=None):
        self.config = config
        self.scenarios = {}

    def define_scenario(self, name: str, description: str,
                        facility_plan: Dict[str, int],
                        target_grids: str = 'top_20pct') -> Dict:
        """
        시설 배치 시나리오 정의.

        Parameters
        ----------
        facility_plan : dict
            {시설유형: 설치개수} 예: {'cctv': 50, 'speed_camera': 20}
        target_grids : str
            'top_20pct', 'top_10pct', 'all_risk', 'custom'
        """
        scenario = {
            'name': name,
            'description': description,
            'facility_plan': facility_plan,
            'target_grids': target_grids
        }
        self.scenarios[name] = scenario
        return scenario

    def estimate_accident_reduction(self, grid_df: pd.DataFrame,
                                     scenario_name: str,
                                     accident_col: str = 'accident_total') -> Dict:
        """시나리오별 사고 감소 추정"""
        if scenario_name not in self.scenarios:
            return {'error': f'시나리오 {scenario_name} 미정의'}

        scenario = self.scenarios[scenario_name]
        plan = scenario['facility_plan']

        # 대상 격자 선정
        target = scenario['target_grids']
        if target == 'top_20pct' and 'TSVI_score' in grid_df.columns:
            threshold = grid_df['TSVI_score'].quantile(0.80)
            mask = grid_df['TSVI_score'] >= threshold
        elif target == 'top_10pct' and 'TSVI_score' in grid_df.columns:
            threshold = grid_df['TSVI_score'].quantile(0.90)
            mask = grid_df['TSVI_score'] >= threshold
        else:
            mask = pd.Series(True, index=grid_df.index)

        target_accidents = grid_df.loc[mask, accident_col].sum()
        total_accidents = grid_df[accident_col].sum()

        # 시설별 감소 효과 합산
        total_reduction = 0
        cost_total = 0
        facility_cost = {
            'smart_crosswalk': 5000, 'speed_camera': 3000,
            'cctv': 1500, 'speed_bump': 200, 'safety_sign': 50
        }

        for facility, count in plan.items():
            effect = self.FACILITY_EFFECTS.get(facility, 0)
            unit_cost = facility_cost.get(facility, 0)
            reduction = target_accidents * effect * min(count / max(mask.sum(), 1), 1.0)
            total_reduction += reduction
            cost_total += count * unit_cost

        result = {
            'scenario': scenario_name,
            'target_grids': int(mask.sum()),
            'target_accidents': float(target_accidents),
            'estimated_reduction': float(total_reduction),
            'reduction_rate': float(total_reduction / max(total_accidents, 1)),
            'total_cost_만원': float(cost_total),
        }

        print(f"[시나리오: {scenario_name}]")
        print(f"  대상 격자: {result['target_grids']}")
        print(f"  예상 사고 감소: {result['estimated_reduction']:.1f}건 "
              f"({result['reduction_rate'] * 100:.1f}%)")
        print(f"  총 비용: {result['total_cost_만원']:,.0f}만원")

        return result

    @staticmethod
    def nilsson_severity_change(v_before: float, v_after: float) -> Dict:
        """
        Nilsson Power Model: 속도 변화에 따른 사고 심각도 변화율.
        사망 = (V_after/V_before)^4
        중상 = (V_after/V_before)^3
        경상 = (V_after/V_before)^2
        """
        ratio = v_after / max(v_before, 1)
        return {
            'death_change': ratio ** 4,
            'serious_injury_change': ratio ** 3,
            'minor_injury_change': ratio ** 2
        }

    def compare_scenarios(self, grid_df: pd.DataFrame) -> pd.DataFrame:
        """모든 시나리오 비교표"""
        rows = []
        for name in self.scenarios:
            result = self.estimate_accident_reduction(grid_df, name)
            if 'error' not in result:
                rows.append(result)

        if not rows:
            return pd.DataFrame()

        comparison = pd.DataFrame(rows)
        comparison['cost_per_reduced_accident'] = (
            comparison['total_cost_만원'] /
            comparison['estimated_reduction'].replace(0, np.nan)
        )

        print(f"\n{'=' * 70}")
        print(f"  시나리오 비교표")
        print(f"{'=' * 70}")
        print(comparison.to_string(index=False))
        return comparison


# ============================================================
# CostBenefitAnalyzer: 비용편익분석
# ============================================================
class CostBenefitAnalyzer:
    """비용편익분석 (NPV, B/C Ratio, 투자회수기간)"""

    def __init__(self, discount_rate: float = 4.5, horizon: int = 10):
        self.discount_rate = discount_rate / 100.0
        self.horizon = horizon
        # 국토교통부 2024 기준 사회적 비용 (만원)
        self.social_cost = {
            'death': 85000,
            'serious_injury': 12000,
            'minor_injury': 1500,
            'property_damage': 500
        }
        # 평균 사고 심각도 분포 (도로교통공단 2024)
        self.severity_distribution = {
            'death': 0.012,
            'serious_injury': 0.048,
            'minor_injury': 0.340,
            'property_damage': 0.600
        }

    def compute_annual_benefit(self, prevented_accidents: float) -> float:
        """
        예방 사고 x 심각도 분포 x 사회적 비용 계수.
        단위: 만원/년
        """
        annual_benefit = 0
        for severity, proportion in self.severity_distribution.items():
            cost = self.social_cost[severity]
            annual_benefit += prevented_accidents * proportion * cost
        return annual_benefit

    def compute_npv(self, total_investment: float,
                    annual_benefit: float) -> Dict:
        """
        NPV, B/C Ratio, 투자회수기간 산출.

        Parameters
        ----------
        total_investment : float (만원)
        annual_benefit : float (만원/년)
        """
        # NPV 계산
        npv = -total_investment
        cumulative = -total_investment
        payback_year = None

        for t in range(1, self.horizon + 1):
            pv_benefit = annual_benefit / ((1 + self.discount_rate) ** t)
            npv += pv_benefit
            cumulative += pv_benefit
            if cumulative >= 0 and payback_year is None:
                payback_year = t

        # 총 편익 현재가치
        total_pv_benefit = sum(
            annual_benefit / ((1 + self.discount_rate) ** t)
            for t in range(1, self.horizon + 1)
        )

        bc_ratio = total_pv_benefit / max(total_investment, 1)

        result = {
            'total_investment_만원': total_investment,
            'annual_benefit_만원': annual_benefit,
            'npv_만원': round(npv, 0),
            'bc_ratio': round(bc_ratio, 2),
            'payback_year': payback_year,
            'total_pv_benefit_만원': round(total_pv_benefit, 0),
            'discount_rate': self.discount_rate * 100,
            'horizon': self.horizon
        }

        print(f"[비용편익분석]")
        print(f"  투자비: {total_investment:,.0f}만원")
        print(f"  연간 편익: {annual_benefit:,.0f}만원")
        print(f"  NPV: {npv:,.0f}만원")
        print(f"  B/C Ratio: {bc_ratio:.2f}")
        print(f"  투자회수기간: {payback_year}년" if payback_year else "  투자회수기간: >10년")

        return result


# ============================================================
# FacilityOptimizer: PuLP 수리최적화
# ============================================================
class FacilityOptimizer:
    """
    PuLP ILP 기반 시설 최적 배치.
    - max_coverage: 예산 제약 하 TSVI 감소 극대화
    - p_median: p개 시설의 수요 가중 거리 최소화
    - set_covering: 고위험 격자 전수 커버 최소 시설
    """

    def __init__(self, facility_cost: Dict[str, int] = None,
                 facility_effects: Dict[str, float] = None):
        self.facility_cost = facility_cost or {
            'smart_crosswalk': 5000, 'speed_camera': 3000,
            'cctv': 1500, 'speed_bump': 200, 'safety_sign': 50
        }
        self.facility_effects = facility_effects or ScenarioAnalyzer.FACILITY_EFFECTS

    def max_coverage_optimization(self, grid_df: pd.DataFrame,
                                   budget: float,
                                   facility_types: List[str] = None,
                                   max_per_grid: int = 1) -> Dict:
        """
        예산 제약 하 TSVI 감소 극대화 (ILP).

        결정변수: x[i,f] = 격자 i에 시설 f 설치 여부 (이진)
        목적함수: max sum x[i,f] * TSVI[i] * effect[f]
        제약: 예산, 격자당 최대 설치 수
        """
        from pulp import LpProblem, LpMaximize, LpVariable, LpStatus, lpSum, value

        if facility_types is None:
            facility_types = list(self.facility_cost.keys())

        # 고위험 격자만 대상 (상위 50%)
        if 'TSVI_score' in grid_df.columns:
            threshold = grid_df['TSVI_score'].quantile(0.50)
            candidate_idx = grid_df[grid_df['TSVI_score'] >= threshold].index.tolist()
        else:
            candidate_idx = grid_df.index.tolist()

        # 문제 정의
        prob = LpProblem("MaxCoverage_TSVI", LpMaximize)

        # 결정변수
        x = {}
        for i in candidate_idx:
            for f in facility_types:
                x[i, f] = LpVariable(f"x_{i}_{f}", cat='Binary')

        # 목적함수: sum x[i,f] * TSVI[i] * effect[f]
        tsvi_col = 'TSVI_score' if 'TSVI_score' in grid_df.columns else None
        objective_terms = []
        for i in candidate_idx:
            tsvi_val = grid_df.loc[i, tsvi_col] if tsvi_col else 50
            for f in facility_types:
                effect = self.facility_effects.get(f, 0.1)
                objective_terms.append(x[i, f] * tsvi_val * effect)

        prob += lpSum(objective_terms)

        # 예산 제약
        budget_terms = []
        for i in candidate_idx:
            for f in facility_types:
                budget_terms.append(x[i, f] * self.facility_cost.get(f, 0))
        prob += lpSum(budget_terms) <= budget

        # 격자당 최대 설치 수
        for i in candidate_idx:
            prob += lpSum([x[i, f] for f in facility_types]) <= max_per_grid

        # 풀이
        prob.solve()
        status = LpStatus[prob.status]

        # 결과 추출
        placements = []
        total_cost = 0
        for i in candidate_idx:
            for f in facility_types:
                if value(x[i, f]) == 1:
                    placements.append({'grid_idx': i, 'facility': f,
                                       'cost': self.facility_cost[f]})
                    total_cost += self.facility_cost[f]

        result = {
            'status': status,
            'objective_value': float(value(prob.objective)) if prob.objective else 0,
            'total_cost': total_cost,
            'budget': budget,
            'n_facilities': len(placements),
            'placements': placements
        }

        print(f"[최적화: MaxCoverage]")
        print(f"  상태: {status}")
        print(f"  설치 시설: {len(placements)}개")
        print(f"  총 비용: {total_cost:,.0f}만원 / 예산: {budget:,.0f}만원")
        print(f"  목적함수값: {result['objective_value']:.2f}")

        # 시설별 분포
        from collections import Counter
        facility_dist = Counter(p['facility'] for p in placements)
        for f, cnt in facility_dist.items():
            print(f"    {f}: {cnt}개")

        return result

    def p_median_optimization(self, grid_df: pd.DataFrame,
                               p: int,
                               demand_col: str = 'TSVI_score',
                               max_grids: int = 200) -> Dict:
        """
        p-Median: p개 시설의 수요 가중 거리 최소화.

        min sum_i sum_j TSVI_i * d_ij * x_ij
        s.t. sum_j y_j = p
        """
        from pulp import LpProblem, LpMinimize, LpVariable, LpStatus, lpSum, value

        # 후보 격자 (상위 N개만 - 계산량 제한)
        if demand_col in grid_df.columns:
            candidates = grid_df.nlargest(min(max_grids, len(grid_df)), demand_col)
        else:
            candidates = grid_df.head(min(max_grids, len(grid_df)))

        n = len(candidates)
        idx = candidates.index.tolist()

        # 거리 행렬 (중심점 간 유클리드)
        if 'centroid_x' in candidates.columns and 'centroid_y' in candidates.columns:
            coords = candidates[['centroid_x', 'centroid_y']].values
            from scipy.spatial.distance import cdist
            dist_matrix = cdist(coords, coords, 'euclidean')
        else:
            dist_matrix = np.ones((n, n)) * 1000
            np.fill_diagonal(dist_matrix, 0)

        # 수요
        demand = candidates[demand_col].values if demand_col in candidates.columns \
            else np.ones(n)

        # 문제 정의
        prob = LpProblem("P_Median", LpMinimize)

        y = {j: LpVariable(f"y_{j}", cat='Binary') for j in range(n)}
        x = {(i, j): LpVariable(f"x_{i}_{j}", cat='Binary')
             for i in range(n) for j in range(n)}

        # 목적함수
        prob += lpSum(demand[i] * dist_matrix[i][j] * x[i, j]
                       for i in range(n) for j in range(n))

        # 제약: 정확히 p개 시설
        prob += lpSum(y[j] for j in range(n)) == p

        # 각 수요점은 정확히 1개 시설에 할당
        for i in range(n):
            prob += lpSum(x[i, j] for j in range(n)) == 1

        # 시설이 설치된 곳에만 할당
        for i in range(n):
            for j in range(n):
                prob += x[i, j] <= y[j]

        prob.solve()
        status = LpStatus[prob.status]

        facility_locations = [idx[j] for j in range(n) if value(y[j]) == 1]

        result = {
            'status': status,
            'p': p,
            'facility_locations': facility_locations,
            'objective_value': float(value(prob.objective)) if prob.objective else 0,
        }

        print(f"[최적화: p-Median (p={p})]")
        print(f"  상태: {status}")
        print(f"  시설 위치: {len(facility_locations)}개")

        return result

    def set_covering_optimization(self, grid_df: pd.DataFrame,
                                   coverage_radius: float = 500,
                                   risk_threshold_pct: float = 0.80) -> Dict:
        """
        Set Covering: 고위험 격자 전수 커버 최소 시설 수.

        Parameters
        ----------
        coverage_radius : float (m)
            시설 커버 반경
        risk_threshold_pct : float
            TSVI 상위 비율 (0.80 = 상위 20%)
        """
        from pulp import LpProblem, LpMinimize, LpVariable, LpStatus, lpSum, value

        # 고위험 격자
        if 'TSVI_score' in grid_df.columns:
            threshold = grid_df['TSVI_score'].quantile(risk_threshold_pct)
            high_risk = grid_df[grid_df['TSVI_score'] >= threshold]
        else:
            high_risk = grid_df

        n = len(high_risk)
        if n == 0:
            return {'error': '고위험 격자 없음'}

        idx = high_risk.index.tolist()

        # 커버리지 매트릭스
        if 'centroid_x' in high_risk.columns:
            coords = high_risk[['centroid_x', 'centroid_y']].values
            from scipy.spatial.distance import cdist
            dist_matrix = cdist(coords, coords, 'euclidean')
            covers = dist_matrix <= coverage_radius
        else:
            covers = np.eye(n, dtype=bool)

        # 문제 정의
        prob = LpProblem("SetCovering", LpMinimize)
        y = {j: LpVariable(f"y_{j}", cat='Binary') for j in range(n)}

        # 목적함수: 최소 시설 수
        prob += lpSum(y[j] for j in range(n))

        # 모든 고위험 격자가 적어도 1개 시설에 커버
        for i in range(n):
            covering_sites = [j for j in range(n) if covers[i][j]]
            prob += lpSum(y[j] for j in covering_sites) >= 1

        prob.solve()
        status = LpStatus[prob.status]

        facility_count = sum(1 for j in range(n) if value(y[j]) == 1)
        facility_locations = [idx[j] for j in range(n) if value(y[j]) == 1]

        result = {
            'status': status,
            'min_facilities': facility_count,
            'facility_locations': facility_locations,
            'coverage_radius': coverage_radius,
            'covered_grids': n,
        }

        print(f"[최적화: Set Covering]")
        print(f"  상태: {status}")
        print(f"  필요 최소 시설: {facility_count}개 (반경 {coverage_radius}m)")
        print(f"  커버 대상: {n}개 고위험 격자")

        return result


# ============================================================
# 사용 예시 및 모듈 검증
# ============================================================
if __name__ == "__main__":
    print("=" * 60)
    print("  TSVI Calculator 모듈 로드 완료")
    print("=" * 60)

    # 클래스 인스턴스화 테스트
    calc = TSVICalculator()
    print(f"\n[TSVICalculator] weights = {calc.weights}")

    eb = EmpiricalBayesAdjuster()
    print(f"[EmpiricalBayesAdjuster] fitted = {eb.is_fitted}")

    ahp = AHPWeightValidator()
    # AHP 테스트: 4x4 쌍대비교 행렬
    test_matrix = np.array([
        [1, 2, 1, 3],
        [1/2, 1, 1/2, 2],
        [1, 2, 1, 2],
        [1/3, 1/2, 1/2, 1]
    ])
    ahp_result = ahp.compute_weights(test_matrix, ['PVI', 'RRI', 'FGI', 'TRI'])

    mgwr = MGWRAnalyzer()
    print(f"\n[MGWRAnalyzer] model = {mgwr.model}")

    scenario = ScenarioAnalyzer()
    print(f"[ScenarioAnalyzer] 시설 효과: {scenario.FACILITY_EFFECTS}")

    cba = CostBenefitAnalyzer()
    print(f"[CostBenefitAnalyzer] 할인율 = {cba.discount_rate * 100}%")

    optimizer = FacilityOptimizer()
    print(f"[FacilityOptimizer] 시설 비용: {optimizer.facility_cost}")

    print("\n모든 클래스 인스턴스화 성공")
    print("\n사용법:")
    print("  calculator = TSVICalculator()")
    print("  result = calculator.calculate_tsvi(grid_df)")
    print("  eb = EmpiricalBayesAdjuster()")
    print("  eb.fit_spf(grid_df)")
    print("  eb_adjusted = eb.adjust(observed, expected)")
    print("  validator = TSVIValidator(grid_df)")
    print("  metrics = validator.validate_full()")
