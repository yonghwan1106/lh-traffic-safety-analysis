# -*- coding: utf-8 -*-
"""
LH 3기신도시 교통안전 데이터 분석 프로젝트
00. 프로젝트 환경 설정 및 유틸리티
"""

# ============================================================
# 필수 라이브러리 설치 (COMPAS 노트북에서 실행)
# ============================================================
# !pip install geopandas pysal mgwr xgboost lightgbm shap optuna pulp folium plotly

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

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
        # COMPAS 노트북 환경
        try:
            plt.rcParams['font.family'] = 'NanumGothic'
        except:
            import subprocess
            subprocess.run(['apt-get', 'install', '-y', 'fonts-nanum'],
                         capture_output=True)
            fm._load_fontmanager(try_read_cache=False)
            plt.rcParams['font.family'] = 'NanumGothic'

    plt.rcParams['axes.unicode_minus'] = False
    print(f"[INFO] 한글 폰트 설정 완료 (OS: {system})")

# ============================================================
# 프로젝트 상수 정의
# ============================================================
# 좌표계
CRS_WGS84 = "EPSG:4326"
CRS_UTM52N = "EPSG:32652"  # 미터 단위 계산용

# 분석 대상 지역
REGIONS = {
    'dongtan1': '동탄1신도시',
    'dongtan2': '동탄2신도시',
    'wirye': '위례신도시',
    'hanam_misa': '하남미사',
    'pangyo': '판교신도시',
    'hanam_gyosan': '하남교산(3기)'
}

# 취약계층 연령 구분
AGE_GROUPS = {
    'child': (0, 12),       # 어린이
    'youth': (13, 17),      # 청소년
    'adult': (18, 64),      # 성인
    'elderly': (65, 100),   # 고령자
    'pregnant_est': (20, 39)  # 임산부 추정 (가임기 여성)
}

# TSVI 가중치 초기값 (AHP 기반, 추후 데이터 보정)
TSVI_WEIGHTS = {
    'PVI': 0.30,  # 인구 취약성
    'RRI': 0.25,  # 도로 위험성
    'FGI': 0.25,  # 시설 공백도
    'TRI': 0.20   # 시간대별 위험 변동
}

# 시설별 비용 (만원)
FACILITY_COST = {
    'smart_crosswalk': 5000,
    'speed_camera': 3000,
    'cctv': 1500,
    'speed_bump': 200,
    'safety_sign': 50
}

# 교통사고 사회적 비용 (만원/건)
ACCIDENT_SOCIAL_COST = {
    'death': 85000,
    'serious_injury': 12000,
    'minor_injury': 1500,
    'property_damage': 500
}

# ============================================================
# 유틸리티 함수
# ============================================================
def load_grid_data(filepath, region_name=None):
    """격자 데이터 로드 및 기본 전처리"""
    gdf = gpd.read_file(filepath, encoding='utf-8')
    gdf = gdf.to_crs(CRS_WGS84)
    if region_name:
        gdf['region'] = region_name
    print(f"[INFO] {region_name or filepath}: {len(gdf)} 격자 로드 완료")
    return gdf

def spatial_join_to_grid(grid_gdf, point_gdf, count_col_name, how='left'):
    """포인트 데이터를 격자에 공간 조인하여 개수 집계"""
    point_gdf = point_gdf.to_crs(grid_gdf.crs)
    joined = gpd.sjoin(grid_gdf, point_gdf, how=how, predicate='contains')
    counts = joined.groupby(grid_gdf.index.name or joined.index).size()
    grid_gdf[count_col_name] = counts.reindex(grid_gdf.index, fill_value=0).astype(int)
    print(f"[INFO] {count_col_name}: 매핑 완료 (비영 격자: {(grid_gdf[count_col_name]>0).sum()})")
    return grid_gdf

def calculate_distance_to_nearest(grid_gdf, point_gdf, dist_col_name):
    """격자 중심에서 가장 가까운 포인트까지의 거리 계산"""
    # UTM 좌표계로 변환하여 미터 단위 거리 계산
    grid_utm = grid_gdf.to_crs(CRS_UTM52N)
    point_utm = point_gdf.to_crs(CRS_UTM52N)

    grid_centroids = grid_utm.geometry.centroid

    distances = []
    for centroid in grid_centroids:
        if len(point_utm) > 0:
            min_dist = point_utm.geometry.distance(centroid).min()
        else:
            min_dist = np.nan
        distances.append(min_dist)

    grid_gdf[dist_col_name] = distances
    print(f"[INFO] {dist_col_name}: 계산 완료 (평균: {np.nanmean(distances):.0f}m)")
    return grid_gdf

def normalize_column(series, method='minmax'):
    """컬럼 정규화"""
    if method == 'minmax':
        return (series - series.min()) / (series.max() - series.min() + 1e-8)
    elif method == 'zscore':
        return (series - series.mean()) / (series.std() + 1e-8)
    elif method == 'robust':
        q25, q75 = series.quantile(0.25), series.quantile(0.75)
        return (series - q25) / (q75 - q25 + 1e-8)

def print_data_quality_report(df, name="DataFrame"):
    """데이터 품질 리포트 출력"""
    print(f"\n{'='*60}")
    print(f" 데이터 품질 리포트: {name}")
    print(f"{'='*60}")
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
    print(f"{'='*60}\n")

# ============================================================
# 실행
# ============================================================
if __name__ == "__main__":
    setup_korean_font()
    print("\n[프로젝트 설정 완료]")
    print(f"분석 대상 지역: {', '.join(REGIONS.values())}")
    print(f"TSVI 가중치: {TSVI_WEIGHTS}")
    print(f"좌표계: {CRS_WGS84}")
