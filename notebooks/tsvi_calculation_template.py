# -*- coding: utf-8 -*-
"""
LH 3기신도시 교통안전 데이터 분석
TSVI (Traffic Safety Vulnerability Index) 산출 모듈

TSVI = w1*PVI + w2*RRI + w3*FGI + w4*TRI

PVI: Population Vulnerability Index (인구 취약성 지수)
RRI: Road Risk Index (도로 위험성 지수)
FGI: Facility Gap Index (시설 공백 지수)
TRI: Temporal Risk Index (시간대별 위험 변동 지수)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')


class TSVICalculator:
    """교통안전 취약성 지수(TSVI) 산출 클래스"""

    def __init__(self, weights=None):
        """
        Parameters
        ----------
        weights : dict, optional
            TSVI 하위 지수 가중치. 기본값은 AHP 기반 초기 가중치.
            {'PVI': 0.30, 'RRI': 0.25, 'FGI': 0.25, 'TRI': 0.20}
        """
        self.weights = weights or {
            'PVI': 0.30,
            'RRI': 0.25,
            'FGI': 0.25,
            'TRI': 0.20
        }
        self.scaler = MinMaxScaler()
        self.is_fitted = False

    def calculate_pvi(self, grid_df):
        """
        PVI (인구 취약성 지수) 산출

        필요 컬럼:
        - pop_child_ratio: 어린이 인구 비율
        - pop_elderly_ratio: 고령 인구 비율
        - pop_female_childbearing_ratio: 가임기 여성 비율
        - floating_pop_density: 유동인구 밀도
        """
        # 취약계층 가중 인구 비율
        vulnerable_ratio = (
            grid_df['pop_child_ratio'] * 0.40 +
            grid_df['pop_elderly_ratio'] * 0.35 +
            grid_df['pop_female_childbearing_ratio'] * 0.25
        )

        # 유동인구 밀도와 결합 (취약계층이 많고 + 유동인구도 많으면 더 위험)
        pop_density_norm = self._normalize(grid_df['floating_pop_density'])
        pvi = vulnerable_ratio * (1 + pop_density_norm)

        return self._normalize(pvi)

    def calculate_rri(self, grid_df):
        """
        RRI (도로 위험성 지수) 산출

        필요 컬럼:
        - traffic_volume: 교통량
        - avg_speed: 평균 속도
        - road_length: 격자 내 도로 연장
        - intersection_count: 교차로 수
        - right_turn_intersection: 우회전 교차로 수
        """
        # 교통량 위험도 (교통량이 많을수록 위험)
        traffic_risk = self._normalize(grid_df['traffic_volume'])

        # 속도 위험도 (속도가 높을수록 위험, 특히 30km/h 초과 시)
        speed_risk = self._normalize(
            np.maximum(grid_df['avg_speed'] - 30, 0)  # 30km/h 기준
        )

        # 교차로 밀도 위험도
        intersection_risk = self._normalize(grid_df['intersection_count'])

        # 우회전 교차로 가중 (우회전 교차로는 보행자에 더 위험)
        right_turn_risk = self._normalize(grid_df['right_turn_intersection']) * 1.5
        right_turn_risk = np.minimum(right_turn_risk, 1.0)

        rri = (
            traffic_risk * 0.30 +
            speed_risk * 0.30 +
            intersection_risk * 0.20 +
            right_turn_risk * 0.20
        )

        return self._normalize(rri)

    def calculate_fgi(self, grid_df):
        """
        FGI (시설 공백 지수) 산출
        값이 높을수록 안전시설이 부족하다는 의미

        필요 컬럼:
        - cctv_count: CCTV 수
        - speedbump_count: 과속방지턱 수
        - crosswalk_density: 횡단보도 밀도
        - child_protection_zone: 어린이보호구역 여부 (0/1)
        - dist_to_nearest_crosswalk: 가장 가까운 횡단보도까지 거리
        """
        # 시설 충족도 (시설이 많을수록 높음)
        facility_coverage = (
            self._normalize(grid_df['cctv_count']) * 0.25 +
            self._normalize(grid_df['speedbump_count']) * 0.20 +
            self._normalize(grid_df['crosswalk_density']) * 0.25 +
            grid_df['child_protection_zone'].astype(float) * 0.15 +
            (1 - self._normalize(grid_df['dist_to_nearest_crosswalk'])) * 0.15
        )

        # 공백도 = 1 - 충족도
        fgi = 1 - self._normalize(facility_coverage)

        return self._normalize(fgi)

    def calculate_tri(self, grid_df):
        """
        TRI (시간대별 위험 변동 지수) 산출

        필요 컬럼:
        - accident_peak_ratio: 출퇴근/등하교 시간대 사고 비율
        - floating_pop_variance: 시간대별 유동인구 변동성
        - congestion_freq: 혼잡 빈도
        - congestion_time_intensity: 혼잡 시간 강도
        """
        # 피크 시간대 사고 집중도
        peak_risk = self._normalize(grid_df['accident_peak_ratio'])

        # 유동인구 변동성 (변동이 클수록 예측 불가 → 위험)
        pop_variance_risk = self._normalize(grid_df['floating_pop_variance'])

        # 혼잡 관련 위험
        congestion_risk = self._normalize(
            grid_df['congestion_freq'] * grid_df['congestion_time_intensity']
        )

        tri = (
            peak_risk * 0.40 +
            pop_variance_risk * 0.30 +
            congestion_risk * 0.30
        )

        return self._normalize(tri)

    def calculate_tsvi(self, grid_df):
        """
        TSVI (교통안전 취약성 지수) 종합 산출

        Returns
        -------
        pd.DataFrame : 원본 + PVI, RRI, FGI, TRI, TSVI 컬럼 추가
        """
        result = grid_df.copy()

        # 하위 지수 산출
        result['PVI'] = self.calculate_pvi(grid_df)
        result['RRI'] = self.calculate_rri(grid_df)
        result['FGI'] = self.calculate_fgi(grid_df)
        result['TRI'] = self.calculate_tri(grid_df)

        # TSVI 종합 산출
        result['TSVI'] = (
            result['PVI'] * self.weights['PVI'] +
            result['RRI'] * self.weights['RRI'] +
            result['FGI'] * self.weights['FGI'] +
            result['TRI'] * self.weights['TRI']
        )

        # 최종 정규화 (0~100 스케일)
        result['TSVI_score'] = self._normalize(result['TSVI']) * 100

        # 위험 등급 분류
        result['TSVI_grade'] = pd.cut(
            result['TSVI_score'],
            bins=[0, 20, 40, 60, 80, 100],
            labels=['안전', '주의', '경계', '위험', '심각'],
            include_lowest=True
        )

        self.is_fitted = True
        print(f"[INFO] TSVI 산출 완료: {len(result)} 격자")
        print(f"  등급 분포:")
        print(result['TSVI_grade'].value_counts().to_string())

        return result

    def optimize_weights_with_data(self, grid_df, accident_col='accident_total'):
        """
        실제 사고 데이터 기반 가중치 최적화

        Parameters
        ----------
        grid_df : DataFrame with PVI, RRI, FGI, TRI columns calculated
        accident_col : str, 실제 사고 건수 컬럼명
        """
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

        # 하이브리드: AHP 가중치와 데이터 가중치의 평균
        hybrid_weights = {}
        for key in features:
            hybrid_weights[key] = round(
                (self.weights[key] + new_weights[key]) / 2, 3
            )
        # 정규화
        total = sum(hybrid_weights.values())
        hybrid_weights = {k: round(v/total, 3) for k, v in hybrid_weights.items()}

        print(f"  하이브리드 가중치: {hybrid_weights}")
        self.weights = hybrid_weights

        return hybrid_weights, rf

    def validate_tsvi(self, grid_df, accident_col='accident_total'):
        """
        TSVI 검증: 실제 사고 데이터와의 일치도 평가

        Returns
        -------
        dict : 검증 메트릭 (precision, recall, auc 등)
        """
        from sklearn.metrics import (
            roc_auc_score, precision_score, recall_score,
            f1_score, confusion_matrix
        )

        y_true = (grid_df[accident_col] > 0).astype(int)

        # TSVI 상위 20%를 고위험으로 분류
        threshold = grid_df['TSVI_score'].quantile(0.80)
        y_pred = (grid_df['TSVI_score'] >= threshold).astype(int)

        metrics = {
            'AUC': roc_auc_score(y_true, grid_df['TSVI_score']),
            'Precision_top20': precision_score(y_true, y_pred, zero_division=0),
            'Recall_top20': recall_score(y_true, y_pred, zero_division=0),
            'F1_top20': f1_score(y_true, y_pred, zero_division=0),
            'Accident_capture_rate': y_true[y_pred == 1].sum() / y_true.sum()
        }

        print(f"\n[INFO] TSVI 검증 결과:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

        return metrics

    @staticmethod
    def _normalize(series):
        """Min-Max 정규화 (0~1)"""
        min_val = series.min()
        max_val = series.max()
        if max_val - min_val < 1e-8:
            return pd.Series(0, index=series.index)
        return (series - min_val) / (max_val - min_val)


class HanamGyosanTransfer:
    """하남교산 전이 분석 클래스"""

    def __init__(self, existing_grids_df):
        """
        Parameters
        ----------
        existing_grids_df : DataFrame
            기존 4개 신도시의 마스터 격자 데이터 (모든 특성 포함)
        """
        self.existing = existing_grids_df
        self.land_use_features = [
            'residential_ratio', 'commercial_ratio', 'industrial_ratio',
            'green_ratio', 'education_ratio', 'road_ratio'
        ]

    def match_similar_grids(self, gyosan_grid_df, top_k=5):
        """
        하남교산 격자와 유사한 기존 신도시 격자 매칭

        Parameters
        ----------
        gyosan_grid_df : DataFrame
            하남교산 격자 (토지이용계획 기반)
        top_k : int
            매칭할 유사 격자 수

        Returns
        -------
        DataFrame : 하남교산 격자 + 매칭된 기존 격자의 특성 추정값
        """
        from sklearn.metrics.pairwise import cosine_similarity

        # 토지이용 벡터 구성
        existing_vectors = self.existing[self.land_use_features].fillna(0).values
        gyosan_vectors = gyosan_grid_df[self.land_use_features].fillna(0).values

        # 코사인 유사도 계산
        similarities = cosine_similarity(gyosan_vectors, existing_vectors)

        # 상위 K개 유사 격자의 가중 평균으로 특성 추정
        transfer_cols = [
            col for col in self.existing.columns
            if col not in self.land_use_features + ['geometry', 'grid_id', 'region']
        ]

        estimated_values = {}
        for i in range(len(gyosan_grid_df)):
            top_k_idx = np.argsort(similarities[i])[-top_k:]
            top_k_sims = similarities[i][top_k_idx]

            # 유사도 가중 평균
            weights = top_k_sims / top_k_sims.sum()
            for col in transfer_cols:
                if col not in estimated_values:
                    estimated_values[col] = []
                values = self.existing.iloc[top_k_idx][col].values
                estimated_values[col].append(np.average(values, weights=weights))

        result = gyosan_grid_df.copy()
        for col, values in estimated_values.items():
            result[f'{col}_estimated'] = values

        # 유사도 통계
        max_sims = similarities.max(axis=1)
        result['max_similarity'] = max_sims
        result['avg_similarity_top_k'] = np.sort(similarities, axis=1)[:, -top_k:].mean(axis=1)

        print(f"[INFO] 하남교산 유사 격자 매칭 완료:")
        print(f"  평균 최고 유사도: {max_sims.mean():.4f}")
        print(f"  유사도 0.8 이상 격자: {(max_sims >= 0.8).sum()} / {len(gyosan_grid_df)}")

        return result

    def estimate_population(self, gyosan_grid_df, planned_households=100000):
        """
        하남교산 계획 인구 기반 인구 추정

        Parameters
        ----------
        gyosan_grid_df : DataFrame with estimated values
        planned_households : int
            계획 세대수 (하남교산 약 10만 세대)
        """
        # 기존 신도시의 세대당 인구 비율 참조
        avg_persons_per_household = 2.5  # 한국 평균 가구원 수

        planned_pop = planned_households * avg_persons_per_household
        total_residential = gyosan_grid_df['residential_ratio'].sum()

        if total_residential > 0:
            gyosan_grid_df['estimated_pop'] = (
                gyosan_grid_df['residential_ratio'] / total_residential * planned_pop
            )
        else:
            gyosan_grid_df['estimated_pop'] = 0

        print(f"[INFO] 하남교산 인구 추정: 총 {planned_pop:,.0f}명 (계획)")

        return gyosan_grid_df


# ============================================================
# 사용 예시
# ============================================================
if __name__ == "__main__":
    print("TSVI Calculator 모듈 로드 완료")
    print("사용법:")
    print("  calculator = TSVICalculator()")
    print("  result = calculator.calculate_tsvi(grid_df)")
    print("  metrics = calculator.validate_tsvi(result)")
