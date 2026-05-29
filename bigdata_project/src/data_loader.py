import streamlit as st
import pandas as pd

@st.cache_data
def load_ratio_data():
    """
    한국부동산원 전세가율 데이터를 로드하고 서울특별시 데이터만 필터링합니다.
    """
    df = pd.read_csv("data/한국부동산원_전세가율.csv", encoding="cp949")
    
    # 가이드 최적화를 위해 서울특별시 데이터만 남김
    # (컬럼명은 데이터 파일에 따라 '시도명' 또는 '지역' 등 실제 확인 후 수정 필요)
    if '시도명' in df.columns:
        df = df[df['시도명'] == '서울특별시']
    return df

@st.cache_data
def load_accident_data():
    """
    주택도시보증공사(HUG) 보증사고 현황 데이터를 로드하고 서울특별시 데이터만 필터링합니다.
    """
    df = pd.read_csv("data/주택도시보증공사_보증사고.xlsx", encoding="cp949")
    
    if '시도명' in df.columns:
        df = df[df['시도명'] == '서울특별시']
    return df