# -*- coding: utf-8 -*-
"""
LH 3기신도시 교통안전 데이터 분석 프로젝트
00. 프로젝트 환경 설정 및 유틸리티

모듈 구성:
- ProjectConfig: 중앙 설정 데이터클래스
- DataColumns: 25개 경쟁 데이터 파일의 실제 컬럼 스키마
- PopulationExtractor: 인구 파생변수 산출
- LandUseClassifier: 토지이용 코드 분류
- SpatialCVSplitter: 공간 교차검증 분할기
- 유틸리티 함수들 (격자 로드, 공간 조인, 정규화 등)
- build_grid_master(): 25개 파일을 격자에 집계하는 오케스트레이터
"""

# ============================================================
# 필수 라이브러리
# ============================================================
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import warnings
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

warnings.filterwarnings('ignore')


# ============================================================
# 로깅 설정
# ============================================================
def setup_logging(level=logging.INFO, log_file=None):
    """프로젝트 전역 로거 설정"""
    fmt = '[%(asctime)s] %(levelname)s - %(message)s'
    datefmt = '%Y-%m-%d %H:%M:%S'
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding='utf-8'))
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, handlers=handlers)
    return logging.getLogger('lh_traffic_safety')


logger = setup_logging()


# ============================================================
# 한글 폰트 설정
# ============================================================
def setup_korean_font():
    """한글 폰트 설정 (Windows/Linux/COMPAS 환경 대응)"""
    import platform
    system = platform.system()

    if system == 'Windows':
        plt.rcParams['font.family'] = 'Malgun Gothic'
    elif system == 'Linux':
        try:
            plt.rcParams['font.family'] = 'NanumGothic'
        except Exception:
            import subprocess
            subprocess.run(['apt-get', 'install', '-y', 'fonts-nanum'],
                           capture_output=True)
            fm._load_fontmanager(try_read_cache=False)
            plt.rcParams['font.family'] = 'NanumGothic'

    plt.rcParams['axes.unicode_minus'] = False
    logger.info(f"한글 폰트 설정 완료 (OS: {system})")


# ============================================================
# ProjectConfig 데이터클래스
# ============================================================
@dataclass
class ProjectConfig:
    """프로젝트 중앙 설정"""

    # 좌표계
    CRS_WGS84: str = "EPSG:4326"
    CRS_UTM52N: str = "EPSG:32652"
    CRS_KOREA: str = "EPSG:5179"  # Korea 2000 / Unified CS

    # 분석 대상 지역
    REGIONS: Dict[str, Dict] = field(default_factory=lambda: {
        'dongtan1': {'name': '동탄1신도시', 'data_type': 'existing'},
        'dongtan2': {'name': '동탄2신도시', 'data_type': 'existing'},
        'wirye': {'name': '위례신도시', 'data_type': 'existing'},
        'hanam_misa': {'name': '하남미사', 'data_type': 'existing'},
        'pangyo': {'name': '판교신도시', 'data_type': 'existing'},
        'hanam_gyosan': {'name': '하남교산(3기)', 'data_type': 'target_3rd_gen'}
    })

    # TSVI 가중치 초기값 (AHP 기반)
    TSVI_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        'PVI': 0.30, 'RRI': 0.25, 'FGI': 0.25, 'TRI': 0.20
    })

    # 시설별 비용 (만원)
    FACILITY_COST: Dict[str, int] = field(default_factory=lambda: {
        'smart_crosswalk': 5000,
        'speed_camera': 3000,
        'cctv': 1500,
        'speed_bump': 200,
        'safety_sign': 50
    })

    # 교통사고 사회적 비용 (만원/건, 국토교통부 2024 기준)
    ACCIDENT_SOCIAL_COST: Dict[str, int] = field(default_factory=lambda: {
        'death': 85000,
        'serious_injury': 12000,
        'minor_injury': 1500,
        'property_damage': 500
    })

    # 사회적 할인율 (%)
    DISCOUNT_RATE: float = 4.5

    # 분석 기간 (년)
    ANALYSIS_HORIZON: int = 10

    # 데이터 소스 모드
    DATA_MODE: str = 'local'  # 'compas' or 'local'

    def get_existing_regions(self) -> Dict:
        """기존 신도시만 반환"""
        return {k: v for k, v in self.REGIONS.items() if v['data_type'] == 'existing'}

    def get_target_regions(self) -> Dict:
        """3기 신도시(하남교산)만 반환"""
        return {k: v for k, v in self.REGIONS.items() if v['data_type'] == 'target_3rd_gen'}


# ============================================================
# DataColumns: 25개 데이터 파일의 실제 컬럼 스키마
# ============================================================
class DataColumns:
    """경쟁 데이터 파일의 실제 컬럼명 매핑"""

    # File 01: 4개 도시 격자 (공개)
    GRID_4CITIES = {
        'grid_id': 'gid',
        'region_code': 'gbn',
        'year': 'std_yr'
    }

    # File 03: 상주인구 (공개)
    RESIDENT_POP = {
        'region_code': 'gbn',
        'grid_id': 'gid',
        'year': 'year',
        'male_20s': 'm_20g_pop',
        'male_30s': 'm_30g_pop',
        'male_40s': 'm_40g_pop',
        'male_50s': 'm_50g_pop',
        'male_60s': 'm_60g_pop',
        'male_70s': 'm_70g_pop',
        'male_80s': 'm_80g_pop',
        'male_90s': 'm_90g_pop',
        'male_100s': 'm_100g_pop',
        'female_20s': 'w_20g_pop',
        'female_30s': 'w_30g_pop',
        'female_40s': 'w_40g_pop',
        'female_50s': 'w_50g_pop',
        'female_60s': 'w_60g_pop',
        'female_70s': 'w_70g_pop',
        'female_80s': 'w_80g_pop',
        'female_90s': 'w_90g_pop',
        'female_100s': 'w_100g_pop',
    }

    # File 04: 유동인구 (비공개)
    FLOATING_POP = {
        'year_month': 'STD_YM',
        'male_10s': 'm_10g_pop',
        'male_20s': 'm_20g_pop',
        'male_30s': 'm_30g_pop',
        'male_40s': 'm_40g_pop',
        'male_50s': 'm_50g_pop',
        'male_60s': 'm_60g_pop',
        'female_10s': 'w_10g_pop',
        'female_20s': 'w_20g_pop',
        'female_30s': 'w_30g_pop',
        'female_40s': 'w_40g_pop',
        'female_50s': 'w_50g_pop',
        'female_60s': 'w_60g_pop',
        'longitude': 'lon',
        'latitude': 'lat'
    }

    # File 05: 직장인구 (비공개) - 24시간 시간대별
    WORKER_POP = {
        'hour_cols': [f'TMST_{h:02d}' for h in range(24)]
    }

    # File 06: 방문인구 (비공개) - 24시간 시간대별
    VISITOR_POP = {
        'hour_cols': [f'TMST_{h:02d}' for h in range(24)]
    }

    # File 07: 서비스인구 (비공개)
    SERVICE_POP = {
        'weekday_flag': 'hw',
        'worker_pop': 'w_pop',
        'visitor_pop': 'v_pop'
    }

    # File 08: 도로 네트워크 (비공개)
    ROAD_NETWORK = {
        'link_id': 'link_id',
        'max_speed': 'max_speed',
        'road_rank': 'road_rank',
        'lanes': 'lanes',
        'oneway': 'oneway',
        'length': 'length'
    }

    # File 09: 도로 속도 (비공개)
    ROAD_SPEED = {
        'link_id': 'v_link_id',
        'avg_speed': 'velocity_AVRG',
        'probe_count': 'probe'
    }

    # File 10: 추정교통량 (비공개)
    ROAD_TRAFFIC = {
        'total_aadt': 'ALL_AADT',
        'passenger_aadt': 'PSCR_AADT',
        'bus_aadt': 'BUS_AADT',
        'truck_aadt': 'FGCR_AADT'
    }

    # File 11: 혼잡빈도 (비공개)
    CONGESTION_FREQ = {
        'congestion_freq': 'FRIN_CG'
    }

    # File 12: 혼잡시간 (비공개)
    CONGESTION_TIME = {
        'congestion_time': 'TI_CG'
    }

    # File 13: 교통사고 (공개)
    ACCIDENTS = {
        'year': 'acc_yr',
        'police_code': 'polc_cd',
        'accident_type': 'acc_tp',
        'accident_type_sub': 'acc_tp_sub',
        'day_night': 'dg_night',
        'day_of_week': 'day_cd',
        'violation_type': 'vlt_tp',
        'road_type': 'road_tp',
        'road_type2': 'road_tp2',
        'road_surface': 'road_sf',
        'weather': 'wthr_cd',
        'death_count': 'dth_aplcnt_cnt',
        'serious_injury_count': 'si_aplcnt_cnt',
        'minor_injury_count': 'mi_aplcnt_cnt',
        'reported_injury_count': 'ri_aplcnt_cnt',
        'injury_count': 'inj_aplcnt_cnt',
        'longitude': 'lon',
        'latitude': 'lat'
    }

    # File 14: 어린이보호구역 (공개)
    CHILD_ZONE = {
        'zone_name': 'zone_nm',
        'address': 'addr'
    }

    # File 15: 초등학교 (공개)
    SCHOOLS = {
        'school_name': 'school_nm',
        'student_count': 'student_cnt'
    }

    # File 16: 유치원 (공개)
    KINDERGARTENS = {
        'name': 'kg_nm',
        'capacity': 'capacity'
    }

    # File 17: 어린이집 (공개)
    DAYCARE = {
        'name': 'dc_nm',
        'capacity': 'capacity'
    }

    # File 18: 횡단보도 (공개)
    CROSSWALKS = {
        'type': 'cw_type',
        'signal': 'signal_yn'
    }

    # File 19: 버스정류장 (공개)
    BUS_STOPS = {
        'stop_name': 'stop_nm',
        'stop_id': 'stop_id'
    }

    # File 20: CCTV (공개)
    CCTV = {
        'purpose': 'cctv_purpose',
        'count': 'cctv_cnt'
    }

    # File 21: 과속방지턱 (공개)
    SPEED_BUMPS = {
        'type': 'bump_type'
    }

    # Files 22/23: 토지이용계획 (공개)
    LAND_USE = {
        'zone_code': 'zoneCode',
        'zone_name': 'zoneName',
        'block_name': 'blockName',
        'block_type': 'blockType'
    }

    @classmethod
    def get_child_elderly_columns(cls) -> Dict[str, List[str]]:
        """어린이/고령자 인구 관련 컬럼 반환"""
        return {
            'elderly_male': ['m_70g_pop', 'm_80g_pop', 'm_90g_pop', 'm_100g_pop'],
            'elderly_female': ['w_70g_pop', 'w_80g_pop', 'w_90g_pop', 'w_100g_pop'],
            'childbearing_female': ['w_20g_pop', 'w_30g_pop'],
        }

    @classmethod
    def get_peak_hours(cls) -> Dict[str, List[int]]:
        """시간대 정의"""
        return {
            'school_morning': list(range(7, 9)),     # 등교 07-09
            'school_afternoon': list(range(13, 16)),  # 하교 13-16
            'commute_morning': list(range(7, 9)),     # 출근 07-09
            'commute_evening': list(range(17, 19)),   # 퇴근 17-19
            'night': list(range(22, 24)) + list(range(0, 2))  # 심야 22-02
        }

    @classmethod
    def get_all_resident_pop_columns(cls) -> List[str]:
        """상주인구 모든 연령별 컬럼 반환"""
        cols = []
        for prefix in ['m', 'w']:
            for age in ['20g', '30g', '40g', '50g', '60g', '70g', '80g', '90g', '100g']:
                cols.append(f'{prefix}_{age}_pop')
        return cols


# ============================================================
# PopulationExtractor: 인구 파생변수 산출
# ============================================================
class PopulationExtractor:
    """상주인구/유동인구에서 파생변수를 산출하는 클래스"""

    def __init__(self, config: ProjectConfig = None):
        self.config = config or ProjectConfig()
        self._col_info = DataColumns.get_child_elderly_columns()

    def extract_elderly_ratio(self, pop_df: pd.DataFrame) -> pd.Series:
        """File 03 상주인구에서 70세 이상 고령인구 비율 산출"""
        elderly_cols = self._col_info['elderly_male'] + self._col_info['elderly_female']
        all_pop_cols = DataColumns.get_all_resident_pop_columns()

        available_elderly = [c for c in elderly_cols if c in pop_df.columns]
        available_all = [c for c in all_pop_cols if c in pop_df.columns]

        if not available_elderly or not available_all:
            logger.warning("고령인구 컬럼 부재 - 0으로 반환")
            return pd.Series(0.0, index=pop_df.index)

        total_elderly = pop_df[available_elderly].sum(axis=1)
        total_pop = pop_df[available_all].sum(axis=1)
        return (total_elderly / total_pop.replace(0, np.nan)).fillna(0)

    def extract_childbearing_ratio(self, pop_df: pd.DataFrame) -> pd.Series:
        """File 03에서 가임기 여성(20-30대) 비율 산출"""
        cb_cols = self._col_info['childbearing_female']
        all_pop_cols = DataColumns.get_all_resident_pop_columns()

        available_cb = [c for c in cb_cols if c in pop_df.columns]
        available_all = [c for c in all_pop_cols if c in pop_df.columns]

        if not available_cb or not available_all:
            logger.warning("가임기 여성 컬럼 부재 - 0으로 반환")
            return pd.Series(0.0, index=pop_df.index)

        total_cb = pop_df[available_cb].sum(axis=1)
        total_pop = pop_df[available_all].sum(axis=1)
        return (total_cb / total_pop.replace(0, np.nan)).fillna(0)

    def estimate_child_population(self, facilities_df: pd.DataFrame,
                                  capacity_col: str = 'capacity') -> pd.Series:
        """
        File 16(유치원)/17(어린이집) 시설용량을 아동인구 프록시로 사용.
        File 03에 0-19세 데이터가 없으므로 시설용량으로 추정.
        """
        if capacity_col in facilities_df.columns:
            return facilities_df[capacity_col].fillna(0).astype(float)
        logger.warning(f"'{capacity_col}' 컬럼 부재 - 0으로 반환")
        return pd.Series(0.0, index=facilities_df.index)

    def compute_floating_pop_density(self, floating_df: pd.DataFrame,
                                     grid_area_m2: float = 10000) -> pd.Series:
        """
        File 04 유동인구 전체 합계 / 격자면적 → 밀도 산출.
        grid_area_m2: 100m x 100m = 10,000 m^2 기본값
        """
        pop_cols = [c for c in floating_df.columns
                    if c.endswith('_pop') and c not in ['lon', 'lat']]
        if not pop_cols:
            logger.warning("유동인구 컬럼 부재 - 0으로 반환")
            return pd.Series(0.0, index=floating_df.index)

        total_pop = floating_df[pop_cols].sum(axis=1)
        return total_pop / (grid_area_m2 / 1e6)  # 인/km^2


# ============================================================
# LandUseClassifier: 토지이용 코드 분류
# ============================================================
class LandUseClassifier:
    """한국 용도지역 코드를 카테고리로 분류"""

    ZONE_CATEGORY_MAP = {
        # 주거지역
        '제1종전용주거': 'residential',
        '제2종전용주거': 'residential',
        '제1종일반주거': 'residential',
        '제2종일반주거': 'residential',
        '제3종일반주거': 'residential',
        '준주거': 'residential_mixed',
        # 상업지역
        '중심상업': 'commercial',
        '일반상업': 'commercial',
        '근린상업': 'commercial',
        '유통상업': 'commercial',
        # 공업지역
        '전용공업': 'industrial',
        '일반공업': 'industrial',
        '준공업': 'industrial',
        # 녹지/자연
        '보전녹지': 'green',
        '생산녹지': 'green',
        '자연녹지': 'green',
        '자연환경보전': 'natural',
        # 관리지역
        '보전관리': 'managed',
        '생산관리': 'managed',
        '계획관리': 'managed',
        # 교육/문화
        '학교': 'education',
        '문화시설': 'culture',
        # 도로
        '도로': 'road',
        '교통시설': 'road',
    }

    @classmethod
    def classify(cls, zone_name: str) -> str:
        """용도지역 이름을 카테고리로 분류"""
        if pd.isna(zone_name):
            return 'unknown'
        for key, category in cls.ZONE_CATEGORY_MAP.items():
            if key in str(zone_name):
                return category
        return 'other'

    @classmethod
    def classify_grid_land_use(cls, grid_gdf: gpd.GeoDataFrame,
                               land_use_gdf: gpd.GeoDataFrame,
                               zone_name_col: str = 'zoneName') -> pd.DataFrame:
        """공간 오버레이로 격자별 토지이용 비율 산출"""
        land_use_gdf = land_use_gdf.to_crs(grid_gdf.crs)
        land_use_gdf['lu_category'] = land_use_gdf[zone_name_col].apply(cls.classify)

        overlay = gpd.overlay(grid_gdf[['geometry']].reset_index(),
                              land_use_gdf[['geometry', 'lu_category']],
                              how='intersection')
        overlay['area'] = overlay.geometry.area

        # 격자별 카테고리 면적 비율
        grid_area = grid_gdf.geometry.area
        categories = ['residential', 'commercial', 'industrial', 'green',
                       'education', 'road', 'other']
        result = pd.DataFrame(index=grid_gdf.index)

        for cat in categories:
            cat_areas = overlay[overlay['lu_category'] == cat].groupby('index')['area'].sum()
            result[f'{cat}_ratio'] = (cat_areas / grid_area).reindex(grid_gdf.index).fillna(0)

        logger.info(f"토지이용 분류 완료: {len(grid_gdf)} 격자, {len(categories)} 카테고리")
        return result


# ============================================================
# SpatialCVSplitter: 공간 교차검증 분할기
# ============================================================
class SpatialCVSplitter:
    """공간 교차검증용 데이터 분할기"""

    @staticmethod
    def leave_one_city_out(grid_df: pd.DataFrame,
                           region_col: str = 'region') -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        3개 도시 학습 → 1개 도시 테스트 (Leave-One-City-Out 4-fold).
        기존 4개 신도시만 대상.
        """
        regions = grid_df[region_col].unique()
        regions = [r for r in regions if 'gyosan' not in str(r).lower()]
        folds = []
        for test_region in regions:
            train_idx = grid_df[grid_df[region_col] != test_region].index.values
            test_idx = grid_df[grid_df[region_col] == test_region].index.values
            folds.append((train_idx, test_idx))
            logger.info(f"  Fold: test={test_region}, train={len(train_idx)}, test={len(test_idx)}")
        return folds

    @staticmethod
    def spatial_block_cv(grid_df: pd.DataFrame, n_splits: int = 5,
                         coord_cols: Tuple[str, str] = ('centroid_x', 'centroid_y'),
                         random_state: int = 42) -> List[Tuple[np.ndarray, np.ndarray]]:
        """
        KMeans 기반 공간 블록 분할 (spatial leakage 방지).
        인접 격자가 같은 fold에 배정되어 누출 방지.
        """
        from sklearn.cluster import KMeans

        coords = grid_df[list(coord_cols)].values
        kmeans = KMeans(n_clusters=n_splits, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(coords)

        folds = []
        for fold_id in range(n_splits):
            test_idx = grid_df.index[labels == fold_id].values
            train_idx = grid_df.index[labels != fold_id].values
            folds.append((train_idx, test_idx))
        logger.info(f"공간 블록 CV: {n_splits}-fold, 블록 크기 "
                     f"{[int((labels == i).sum()) for i in range(n_splits)]}")
        return folds


# ============================================================
# 유틸리티 함수
# ============================================================
def load_grid_data(filepath, region_name=None):
    """격자 데이터 로드 및 기본 전처리"""
    gdf = gpd.read_file(filepath, encoding='utf-8')
    gdf = gdf.to_crs(ProjectConfig.CRS_WGS84)
    if region_name:
        gdf['region'] = region_name
    logger.info(f"{region_name or filepath}: {len(gdf)} 격자 로드 완료")
    return gdf


def points_from_lonlat(df: pd.DataFrame, lon_col: str = 'lon',
                       lat_col: str = 'lat',
                       crs: str = "EPSG:4326") -> gpd.GeoDataFrame:
    """lon/lat 컬럼을 가진 DataFrame → GeoDataFrame 변환 (NaN 처리 포함)"""
    valid = df.dropna(subset=[lon_col, lat_col]).copy()
    if len(valid) < len(df):
        logger.warning(f"좌표 NaN {len(df) - len(valid)}건 제외")
    geometry = gpd.points_from_xy(valid[lon_col], valid[lat_col])
    return gpd.GeoDataFrame(valid, geometry=geometry, crs=crs)


def spatial_join_to_grid(grid_gdf, point_gdf, count_col_name, how='left'):
    """포인트 데이터를 격자에 공간 조인하여 개수 집계"""
    point_gdf = point_gdf.to_crs(grid_gdf.crs)
    joined = gpd.sjoin(grid_gdf, point_gdf, how=how, predicate='contains')
    counts = joined.groupby(grid_gdf.index.name or joined.index).size()
    grid_gdf[count_col_name] = counts.reindex(grid_gdf.index, fill_value=0).astype(int)
    logger.info(f"{count_col_name}: 매핑 완료 (비영 격자: {(grid_gdf[count_col_name] > 0).sum()})")
    return grid_gdf


def spatial_join_to_grid_enhanced(grid_gdf: gpd.GeoDataFrame,
                                  data_gdf: gpd.GeoDataFrame,
                                  agg_dict: Dict[str, Tuple[str, str]],
                                  how: str = 'left') -> gpd.GeoDataFrame:
    """
    유연한 집계 딕셔너리 기반 공간 조인.

    Parameters
    ----------
    agg_dict : dict
        {출력컬럼명: (입력컬럼명, 집계함수)} 형태.
        예: {'traffic_volume_sum': ('ALL_AADT', 'sum'),
             'avg_speed_mean': ('velocity_AVRG', 'mean')}
    """
    data_gdf = data_gdf.to_crs(grid_gdf.crs)
    joined = gpd.sjoin(grid_gdf[['geometry']], data_gdf, how=how, predicate='contains')

    idx_name = grid_gdf.index.name or 'index'
    for out_col, (in_col, func) in agg_dict.items():
        if in_col in joined.columns:
            grouped = joined.groupby(joined.index)[in_col].agg(func)
            grid_gdf[out_col] = grouped.reindex(grid_gdf.index, fill_value=0)
        else:
            logger.warning(f"컬럼 '{in_col}' 없음 → '{out_col}' = 0")
            grid_gdf[out_col] = 0

    return grid_gdf


def calculate_distance_to_nearest(grid_gdf, point_gdf, dist_col_name):
    """격자 중심에서 가장 가까운 포인트까지의 거리 계산 (기본 버전)"""
    grid_utm = grid_gdf.to_crs(ProjectConfig.CRS_UTM52N)
    point_utm = point_gdf.to_crs(ProjectConfig.CRS_UTM52N)

    grid_centroids = grid_utm.geometry.centroid

    distances = []
    for centroid in grid_centroids:
        if len(point_utm) > 0:
            min_dist = point_utm.geometry.distance(centroid).min()
        else:
            min_dist = np.nan
        distances.append(min_dist)

    grid_gdf[dist_col_name] = distances
    logger.info(f"{dist_col_name}: 계산 완료 (평균: {np.nanmean(distances):.0f}m)")
    return grid_gdf


def calculate_distance_to_nearest_vectorized(grid_gdf: gpd.GeoDataFrame,
                                              point_gdf: gpd.GeoDataFrame,
                                              dist_col_name: str) -> gpd.GeoDataFrame:
    """STRtree 기반 최근접 거리 계산 (10-100배 속도 향상)"""
    from shapely import STRtree

    grid_utm = grid_gdf.to_crs(ProjectConfig.CRS_UTM52N)
    point_utm = point_gdf.to_crs(ProjectConfig.CRS_UTM52N)

    if len(point_utm) == 0:
        grid_gdf[dist_col_name] = np.nan
        return grid_gdf

    centroids = grid_utm.geometry.centroid
    tree = STRtree(point_utm.geometry.values)

    distances = []
    for c in centroids:
        nearest_idx = tree.nearest(c)
        nearest_geom = point_utm.geometry.values[nearest_idx]
        distances.append(c.distance(nearest_geom))

    grid_gdf[dist_col_name] = distances
    logger.info(f"{dist_col_name}: STRtree 계산 완료 (평균: {np.nanmean(distances):.0f}m)")
    return grid_gdf


def normalize_column(series, method='minmax'):
    """컬럼 정규화 (기본)"""
    if method == 'minmax':
        return (series - series.min()) / (series.max() - series.min() + 1e-8)
    elif method == 'zscore':
        return (series - series.mean()) / (series.std() + 1e-8)
    elif method == 'robust':
        q25, q75 = series.quantile(0.25), series.quantile(0.75)
        return (series - q25) / (q75 - q25 + 1e-8)


def normalize_column_enhanced(series: pd.Series, method: str = 'minmax',
                               winsorize_pct: float = 0.0) -> pd.Series:
    """
    향상된 정규화: 이상치 윈저화 + 다양한 정규화 방법.

    Parameters
    ----------
    winsorize_pct : float
        윈저화 비율 (0.05 = 상하위 5% 클리핑)
    method : str
        'minmax', 'zscore', 'robust', 'rank'
    """
    s = series.copy()

    # 윈저화
    if winsorize_pct > 0:
        lower = s.quantile(winsorize_pct)
        upper = s.quantile(1 - winsorize_pct)
        s = s.clip(lower, upper)

    if method == 'minmax':
        return (s - s.min()) / (s.max() - s.min() + 1e-8)
    elif method == 'zscore':
        return (s - s.mean()) / (s.std() + 1e-8)
    elif method == 'robust':
        q25, q75 = s.quantile(0.25), s.quantile(0.75)
        return (s - q25) / (q75 - q25 + 1e-8)
    elif method == 'rank':
        return s.rank(pct=True)
    return s


def print_data_quality_report(df, name="DataFrame"):
    """데이터 품질 리포트 출력 (기본)"""
    print(f"\n{'=' * 60}")
    print(f" 데이터 품질 리포트: {name}")
    print(f"{'=' * 60}")
    print(f" 행 수: {len(df):,}")
    print(f" 열 수: {len(df.columns)}")
    print(f"\n 결측값 현황:")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    missing_df = pd.DataFrame({'결측수': missing, '비율(%)': missing_pct})
    missing_df = missing_df[missing_df['결측수'] > 0].sort_values('비율(%)', ascending=False)
    if len(missing_df) > 0:
        print(missing_df.to_string())
    else:
        print("  결측값 없음")
    print(f"{'=' * 60}\n")


def print_data_quality_report_enhanced(df: pd.DataFrame, name: str = "DataFrame"):
    """향상된 데이터 품질 리포트: 구조화된 출력 + NULL 마스킹 패턴 인식"""
    n_rows, n_cols = df.shape
    print(f"\n{'=' * 70}")
    print(f"  데이터 품질 리포트: {name}")
    print(f"{'=' * 70}")
    print(f"  행: {n_rows:,}  |  열: {n_cols}  |  메모리: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    # 결측값
    missing = df.isnull().sum()
    missing_pct = (missing / n_rows * 100).round(2)
    has_missing = missing[missing > 0].sort_values(ascending=False)

    if len(has_missing) > 0:
        print(f"\n  결측값 ({len(has_missing)}개 컬럼):")
        for col in has_missing.index[:10]:
            bar = '#' * int(missing_pct[col] / 5)
            print(f"    {col:30s} {has_missing[col]:>6,} ({missing_pct[col]:5.1f}%) |{bar}")
        if len(has_missing) > 10:
            print(f"    ... 외 {len(has_missing) - 10}개 컬럼")
    else:
        print("\n  결측값 없음")

    # NULL 마스킹 패턴: -999, 9999, 0이 과도하게 많은 컬럼 탐지
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    suspicious = []
    for col in numeric_cols:
        for sentinel in [-999, -99, 9999, 99999]:
            cnt = (df[col] == sentinel).sum()
            if cnt > n_rows * 0.01:
                suspicious.append((col, sentinel, cnt))
    if suspicious:
        print(f"\n  의심 마스킹 값:")
        for col, val, cnt in suspicious[:5]:
            print(f"    {col}: {val} = {cnt:,}건 ({cnt / n_rows * 100:.1f}%)")

    # 수치형 요약
    if len(numeric_cols) > 0:
        print(f"\n  수치형 컬럼 ({len(numeric_cols)}개): "
              f"전체 0 비율 = {(df[numeric_cols] == 0).sum().sum() / (n_rows * len(numeric_cols)) * 100:.1f}%")

    print(f"{'=' * 70}\n")


def load_compas_data(file_key: str, config: ProjectConfig = None,
                     local_path: str = None, **kwargs) -> pd.DataFrame:
    """
    COMPAS API 래퍼: COMPAS 환경이면 geoband API, 로컬이면 파일 로드.
    """
    config = config or ProjectConfig()

    if config.DATA_MODE == 'compas':
        try:
            import geoband
            dataset = geoband.API.load_dataset(file_key)
            df = dataset.to_dataframe(**kwargs)
            logger.info(f"COMPAS 로드: {file_key} ({len(df)} rows)")
            return df
        except ImportError:
            logger.warning("geoband 미설치 → 로컬 모드로 전환")
        except Exception as e:
            logger.warning(f"COMPAS 로드 실패: {e} → 로컬 모드로 전환")

    if local_path:
        path = Path(local_path)
        if path.suffix == '.csv':
            df = pd.read_csv(path, encoding='utf-8', **kwargs)
        elif path.suffix in ['.shp', '.geojson', '.gpkg']:
            df = gpd.read_file(path, encoding='utf-8')
        elif path.suffix in ['.xlsx', '.xls']:
            df = pd.read_excel(path, **kwargs)
        else:
            df = pd.read_csv(path, encoding='utf-8', **kwargs)
        logger.info(f"로컬 로드: {path.name} ({len(df)} rows)")
        return df

    logger.error(f"데이터 로드 실패: {file_key}")
    return pd.DataFrame()


# ============================================================
# 격자 마스터 테이블 빌더
# ============================================================
def _aggregate_resident_pop_to_grid(grid_gdf: gpd.GeoDataFrame,
                                     pop_df: pd.DataFrame,
                                     grid_id_col: str = 'gid') -> gpd.GeoDataFrame:
    """상주인구(File 03)를 격자에 집계"""
    extractor = PopulationExtractor()

    if grid_id_col in pop_df.columns and grid_id_col in grid_gdf.columns:
        merged = grid_gdf.merge(pop_df, on=grid_id_col, how='left', suffixes=('', '_pop'))

        grid_gdf['elderly_ratio'] = extractor.extract_elderly_ratio(merged)
        grid_gdf['childbearing_ratio'] = extractor.extract_childbearing_ratio(merged)

        all_cols = DataColumns.get_all_resident_pop_columns()
        avail = [c for c in all_cols if c in merged.columns]
        if avail:
            grid_gdf['total_resident_pop'] = merged[avail].sum(axis=1)
    else:
        logger.warning("상주인구 격자ID 매칭 불가 - 공간조인 필요")

    logger.info(f"상주인구 집계 완료: elderly_ratio, childbearing_ratio")
    return grid_gdf


def _aggregate_floating_pop_to_grid(grid_gdf: gpd.GeoDataFrame,
                                     floating_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """유동인구(File 04)를 격자에 집계"""
    if 'lon' in floating_df.columns and 'lat' in floating_df.columns:
        float_gdf = points_from_lonlat(floating_df)
        extractor = PopulationExtractor()
        # 격자별 유동인구 합계
        grid_gdf = spatial_join_to_grid(grid_gdf, float_gdf, 'floating_pop_count')
        grid_gdf['floating_pop_density'] = extractor.compute_floating_pop_density(
            grid_gdf[['floating_pop_count']].rename(
                columns={'floating_pop_count': 'total_pop'}))
    logger.info("유동인구 집계 완료")
    return grid_gdf


def _aggregate_hourly_pop_to_grid(grid_gdf: gpd.GeoDataFrame,
                                   worker_df: pd.DataFrame,
                                   visitor_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """직장인구(File 05)/방문인구(File 06)의 24시간 시간대별 변동 집계"""
    hour_cols = DataColumns.WORKER_POP['hour_cols']

    for label, df in [('worker', worker_df), ('visitor', visitor_df)]:
        avail = [c for c in hour_cols if c in df.columns]
        if avail:
            # 시간대별 변동계수 (CV)
            hourly_vals = df[avail]
            cv = hourly_vals.std(axis=1) / (hourly_vals.mean(axis=1) + 1e-8)
            df[f'{label}_hourly_cv'] = cv

            # 피크/비피크 비율
            peak_hours = DataColumns.get_peak_hours()
            peak_cols = [f'TMST_{h:02d}' for h in peak_hours['commute_morning'] +
                         peak_hours['commute_evening']]
            peak_avail = [c for c in peak_cols if c in avail]
            if peak_avail:
                peak_mean = hourly_vals[peak_avail].mean(axis=1)
                total_mean = hourly_vals.mean(axis=1)
                df[f'{label}_peak_ratio'] = (peak_mean / (total_mean + 1e-8))

    logger.info("시간대별 인구 집계 완료")
    return grid_gdf


def _aggregate_road_network_to_grid(grid_gdf: gpd.GeoDataFrame,
                                     road_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """도로 네트워크(File 08)를 격자에 집계"""
    if not isinstance(road_gdf, gpd.GeoDataFrame):
        logger.warning("도로 네트워크가 GeoDataFrame이 아님")
        return grid_gdf

    agg_dict = {
        'road_length_total': ('length', 'sum'),
        'max_speed_max': ('max_speed', 'max'),
        'lanes_max': ('lanes', 'max'),
    }
    grid_gdf = spatial_join_to_grid_enhanced(grid_gdf, road_gdf, agg_dict)

    # 교차로 수: 도로 링크 노드에서 추정 (3개 이상 링크가 만나는 점)
    # 실제 구현 시 road_gdf의 시작/끝 노드를 추출하여 빈도 기반으로 산출
    logger.info("도로 네트워크 집계 완료")
    return grid_gdf


def _aggregate_road_speed_traffic_to_grid(grid_gdf: gpd.GeoDataFrame,
                                           speed_gdf: gpd.GeoDataFrame,
                                           traffic_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """도로 속도(File 09) + 추정교통량(File 10) 집계"""
    if isinstance(speed_gdf, gpd.GeoDataFrame):
        agg_speed = {'avg_speed': ('velocity_AVRG', 'mean')}
        grid_gdf = spatial_join_to_grid_enhanced(grid_gdf, speed_gdf, agg_speed)

    if isinstance(traffic_gdf, gpd.GeoDataFrame):
        agg_traffic = {
            'traffic_volume_all': ('ALL_AADT', 'sum'),
            'traffic_volume_passenger': ('PSCR_AADT', 'sum'),
            'traffic_volume_bus': ('BUS_AADT', 'sum'),
            'traffic_volume_truck': ('FGCR_AADT', 'sum'),
        }
        grid_gdf = spatial_join_to_grid_enhanced(grid_gdf, traffic_gdf, agg_traffic)

    logger.info("도로 속도/교통량 집계 완료")
    return grid_gdf


def _aggregate_accidents_to_grid(grid_gdf: gpd.GeoDataFrame,
                                  accident_df: pd.DataFrame) -> gpd.GeoDataFrame:
    """교통사고(File 13) 집계"""
    cols = DataColumns.ACCIDENTS
    if cols['longitude'] in accident_df.columns:
        acc_gdf = points_from_lonlat(accident_df,
                                     lon_col=cols['longitude'],
                                     lat_col=cols['latitude'])

        # 전체 사고 건수
        grid_gdf = spatial_join_to_grid(grid_gdf, acc_gdf, 'accident_total')

        # 사망+중상 사고
        if cols['death_count'] in acc_gdf.columns:
            acc_gdf['severe'] = (acc_gdf[cols['death_count']].fillna(0) +
                                 acc_gdf[cols['serious_injury_count']].fillna(0))
            severe_gdf = acc_gdf[acc_gdf['severe'] > 0]
            if len(severe_gdf) > 0:
                grid_gdf = spatial_join_to_grid(grid_gdf, severe_gdf, 'accident_severe')

    logger.info("교통사고 집계 완료")
    return grid_gdf


def _aggregate_facilities_to_grid(grid_gdf: gpd.GeoDataFrame,
                                   facilities: Dict[str, gpd.GeoDataFrame]) -> gpd.GeoDataFrame:
    """
    안전시설 (Files 14-21) 집계.
    facilities: {'child_zone': gdf, 'schools': gdf, 'crosswalks': gdf, ...}
    """
    facility_mapping = {
        'child_zone': 'child_protection_zone_count',
        'schools': 'school_count',
        'kindergartens': 'kindergarten_count',
        'daycare': 'daycare_count',
        'crosswalks': 'crosswalk_count',
        'bus_stops': 'bus_stop_count',
        'cctv': 'cctv_count',
        'speed_bumps': 'speed_bump_count'
    }

    for key, col_name in facility_mapping.items():
        if key in facilities and len(facilities[key]) > 0:
            grid_gdf = spatial_join_to_grid(grid_gdf, facilities[key], col_name)

    # 최근접 거리 계산 (핵심 시설)
    distance_targets = ['crosswalks', 'schools', 'bus_stops', 'cctv']
    for target in distance_targets:
        if target in facilities and len(facilities[target]) > 0:
            dist_col = f'dist_nearest_{target}'
            grid_gdf = calculate_distance_to_nearest_vectorized(
                grid_gdf, facilities[target], dist_col)

    logger.info("안전시설 집계 완료")
    return grid_gdf


def _aggregate_land_use_to_grid(grid_gdf: gpd.GeoDataFrame,
                                 land_use_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """토지이용(Files 22/23) 집계"""
    lu_ratios = LandUseClassifier.classify_grid_land_use(grid_gdf, land_use_gdf)
    for col in lu_ratios.columns:
        grid_gdf[col] = lu_ratios[col]
    logger.info("토지이용 집계 완료")
    return grid_gdf


def build_grid_master(grid_gdf: gpd.GeoDataFrame,
                      data_dict: Dict[str, Any]) -> gpd.GeoDataFrame:
    """
    25개 파일을 순차적으로 격자에 집계하는 오케스트레이터.

    Parameters
    ----------
    grid_gdf : GeoDataFrame
        기본 격자 (File 01)
    data_dict : dict
        키별 데이터프레임:
        - 'resident_pop': File 03
        - 'floating_pop': File 04
        - 'worker_pop': File 05
        - 'visitor_pop': File 06
        - 'road_network': File 08
        - 'road_speed': File 09
        - 'road_traffic': File 10
        - 'congestion_freq': File 11
        - 'congestion_time': File 12
        - 'accidents': File 13
        - 'facilities': dict of Files 14-21
        - 'land_use': Files 22/23

    Returns
    -------
    GeoDataFrame : 모든 변수가 집계된 마스터 격자
    """
    logger.info(f"{'=' * 50}")
    logger.info(f"격자 마스터 테이블 빌드 시작: {len(grid_gdf)} 격자")
    logger.info(f"{'=' * 50}")

    result = grid_gdf.copy()

    # 중심점 좌표 추가
    centroids = result.to_crs(ProjectConfig.CRS_UTM52N).geometry.centroid
    result['centroid_x'] = centroids.x
    result['centroid_y'] = centroids.y

    # 1. 상주인구
    if 'resident_pop' in data_dict:
        result = _aggregate_resident_pop_to_grid(result, data_dict['resident_pop'])

    # 2. 유동인구
    if 'floating_pop' in data_dict:
        result = _aggregate_floating_pop_to_grid(result, data_dict['floating_pop'])

    # 3. 시간대별 인구
    worker = data_dict.get('worker_pop', pd.DataFrame())
    visitor = data_dict.get('visitor_pop', pd.DataFrame())
    if not worker.empty or not visitor.empty:
        result = _aggregate_hourly_pop_to_grid(result, worker, visitor)

    # 4. 도로 네트워크
    if 'road_network' in data_dict:
        result = _aggregate_road_network_to_grid(result, data_dict['road_network'])

    # 5. 도로 속도 + 교통량
    speed = data_dict.get('road_speed', gpd.GeoDataFrame())
    traffic = data_dict.get('road_traffic', gpd.GeoDataFrame())
    if not speed.empty or not traffic.empty:
        result = _aggregate_road_speed_traffic_to_grid(result, speed, traffic)

    # 6. 혼잡도
    for key, col in [('congestion_freq', 'congestion_freq'),
                     ('congestion_time', 'congestion_time_intensity')]:
        if key in data_dict and isinstance(data_dict[key], gpd.GeoDataFrame):
            src_col = list(getattr(DataColumns, key.upper(), {}).values())[0] \
                if hasattr(DataColumns, key.upper()) else None
            if src_col:
                agg = {col: (src_col, 'mean')}
                result = spatial_join_to_grid_enhanced(result, data_dict[key], agg)

    # 7. 교통사고
    if 'accidents' in data_dict:
        result = _aggregate_accidents_to_grid(result, data_dict['accidents'])

    # 8. 안전시설
    if 'facilities' in data_dict:
        result = _aggregate_facilities_to_grid(result, data_dict['facilities'])

    # 9. 토지이용
    if 'land_use' in data_dict:
        result = _aggregate_land_use_to_grid(result, data_dict['land_use'])

    logger.info(f"{'=' * 50}")
    logger.info(f"격자 마스터 테이블 빌드 완료: {result.shape[1]} 컬럼")
    logger.info(f"{'=' * 50}")

    return result


# ============================================================
# 실행
# ============================================================
if __name__ == "__main__":
    setup_korean_font()

    config = ProjectConfig()
    print("\n[프로젝트 설정 완료]")
    print(f"분석 대상 지역: {', '.join(v['name'] for v in config.REGIONS.values())}")
    print(f"TSVI 가중치: {config.TSVI_WEIGHTS}")
    print(f"좌표계: {config.CRS_WGS84}")
    print(f"데이터 모드: {config.DATA_MODE}")

    # DataColumns 검증
    ce_cols = DataColumns.get_child_elderly_columns()
    print(f"\n[DataColumns 검증]")
    print(f"고령 남성 컬럼: {ce_cols['elderly_male']}")
    print(f"고령 여성 컬럼: {ce_cols['elderly_female']}")
    print(f"가임기 여성 컬럼: {ce_cols['childbearing_female']}")

    peak_hours = DataColumns.get_peak_hours()
    print(f"\n[시간대 정의]")
    for name, hours in peak_hours.items():
        print(f"  {name}: {hours}")

    print(f"\n[모든 상주인구 컬럼]: {len(DataColumns.get_all_resident_pop_columns())}개")
    print(f"[사고 컬럼 수]: {len(DataColumns.ACCIDENTS)}개")

    print("\n모든 모듈 로드 성공")
