import pandas as pd
import numpy as np
import os
import glob

def clean_amount(val):
    """금액 데이터의 쉼표(,)를 제거하고 숫자로 변환합니다."""
    if pd.isna(val):
        return np.nan
    try:
        return float(str(val).replace(',', '').strip())
    except ValueError:
        return np.nan

# =========================================================================
# [자동 검색 및 디버깅 기능 강화] 파일명에 핵심 단어만 있으면 자동으로 찾습니다.
# =========================================================================

def load_accident_data():
    """파일명에 '보증사고'가 포함된 CSV 파일을 자동으로 찾아 로드합니다."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    # '보증사고' 글자가 들어간 파일 찾기
    pattern = os.path.join(data_dir, "*전세가율*.csv")
    files = glob.glob(pattern)
    
    if not files:
        # 에러 발생 시 현재 폴더에 실제로 무슨 파일이 있는지 보여주는 똑똑한 기능
        actual_files = os.listdir(data_dir) if os.path.exists(data_dir) else []
        file_list_str = "\n".join([f"- {f}" for f in actual_files]) if actual_files else "(폴더가 비어있거나 없습니다)"
        
        raise FileNotFoundError(
            f"❌ [에러] data/ 폴더에서 '보증사고'라는 단어가 포함된 파일을 찾을 수 없습니다.\n"
            f"📂 현재 data/ 폴더 안의 실제 파일 목록:\n{file_list_str}\n"
            f"💡 해결책: HUG 파일 이름에 '보증사고'라는 글자가 들어가도록 이름을 수정해 주세요!"
        )
        
    file_path = max(files, key=os.path.getctime)
    return pd.read_csv(file_path, encoding='cp949')

def load_jeonse_rate_data():
    """파일명에 '전세가율'이 포함된 CSV 파일을 자동으로 찾아 로드합니다."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    # '전세가율' 글자가 들어간 파일 찾기
    pattern = os.path.join(data_dir, "*전세가율*.csv")
    files = glob.glob(pattern)
    
    if not files:
        actual_files = os.listdir(data_dir) if os.path.exists(data_dir) else []
        file_list_str = "\n".join([f"- {f}" for f in actual_files]) if actual_files else "(폴더가 비어있거나 없습니다)"
        
        raise FileNotFoundError(
            f"❌ [에러] data/ 폴더에서 '전세가율'이라는 단어가 포함된 파일을 찾을 수 없습니다.\n"
            f"📂 현재 data/ 폴더 안의 실제 파일 목록:\n{file_list_str}\n"
            f"💡 해결책: 부동산원 파일 이름에 '전세가율'이라는 글자가 들어가도록 이름을 수정해 주세요!"
        )
        
    file_path = max(files, key=os.path.getctime)
    return pd.read_csv(file_path, encoding='cp949')

# 국토부 실거래가 기반 대용량 패널 데이터 및 특성 엔지니어링 알고리즘
def load_and_process_real_estate():
    """국토부 실거래가 대용량 데이터를 로드하여 '동+월'별 전세가율 패널 데이터를 생성합니다."""
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    mae_pattern = os.path.join(data_dir, "연립다세대(매매)_실거래가_*.csv")
    rent_pattern = os.path.join(data_dir, "연립다세대(전월세)_실거래가_*.csv")
    
    mae_files = glob.glob(mae_pattern)
    rent_files = glob.glob(rent_pattern)
    
    if not mae_files or not rent_files:
        raise FileNotFoundError("data/ 폴더에 '연립다세대(매매)_실거래가' 및 '연립다세대(전월세)_실거래가' CSV 파일이 있는지 확인해주세요.")
    
    mae_file = max(mae_files, key=os.path.getctime)
    rent_file = max(rent_files, key=os.path.getctime)

    df_mae = pd.read_csv(mae_file, skiprows=15, encoding="cp949")
    df_mae['거래금액(만원)'] = df_mae['거래금액(만원)'].apply(clean_amount)
    
    df_rent = pd.read_csv(rent_file, skiprows=15, encoding="cp949")
    df_rent = df_rent[df_rent['전월세구분'] == '전세'].copy()
    df_rent['보증금(만원)'] = df_rent['보증금(만원)'].apply(clean_amount)
    
    df_mae['시군구'] = df_mae['시군구'].astype(str)
    df_rent['시군구'] = df_rent['시군구'].astype(str)
    
    df_mae['구'] = df_mae['시군구'].apply(lambda x: x.split()[1] if len(x.split()) > 1 else '')
    df_mae['동'] = df_mae['시군구'].apply(lambda x: x.split()[2] if len(x.split()) > 2 else '')
    
    df_rent['구'] = df_rent['시군구'].apply(lambda x: x.split()[1] if len(x.split()) > 1 else '')
    df_rent['동'] = df_rent['시군구'].apply(lambda x: x.split()[2] if len(x.split()) > 2 else '')

    mae_grouped = df_mae.groupby(['구', '동', '계약년월'], as_index=False).agg(
        평균매매가=('거래금액(만원)', 'mean'), 매매건수=('거래금액(만원)', 'count')
    )
    rent_grouped = df_rent.groupby(['구', '동', '계약년월'], as_index=False).agg(
        평균전세가=('보증금(만원)', 'mean'), 전세건수=('보증금(만원)', 'count')
    )

    df_merged = pd.merge(mae_grouped, rent_grouped, on=['구', '동', '계약년월'], how='inner')
    df_merged['전세가율'] = (df_merged['평균전세가'] / df_merged['평균매매가']) * 100
    df_merged = df_merged[(df_merged['전세가율'] > 10) & (df_merged['전세가율'] < 150)]
    
    cond_danger = df_merged['전세가율'] >= 80
    limited_rent_count = np.minimum(df_merged['전세건수'], 50)
    
    df_merged['위험점수'] = np.where(
        cond_danger,
        (df_merged['전세가율'] * 0.7) + (limited_rent_count * 0.6),
        (df_merged['전세가율'] * 0.5)
    )
    
    max_score = df_merged['위험점수'].max() if len(df_merged) > 0 else 1
    df_merged['위험점수'] = (df_merged['위험점수'] / max_score * 100).round(1)

    return df_merged.sort_values(by='위험점수', ascending=False).reset_index(drop=True)