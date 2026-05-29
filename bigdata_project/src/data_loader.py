import pandas as pd
import streamlit as st

@st.cache_data
def load_accident_data():
    """
    주택도시보증공사(HUG) 보증사고 엑셀 파일을 안전하게 로드합니다.
    """
    column_names = ['지역코드', '광역지자체', '기초지자체', '사고건수', '사고금액', '사고율']
    df = pd.read_excel("data/주택도시보증공사_보증사고.xlsx", skiprows=5, header=None, names=column_names)
    
    # 텍스트 공백 제거
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].astype(str).str.strip()
            
    # '서울'이 포함된 행만 필터링 (소계 행 제외)
    df = df[df['광역지자체'].str.contains('서울', na=False) & (df['기초지자체'] != '소계')]
    
    # 수치형 변환
    df['사고건수'] = pd.to_numeric(df['사고건수'], errors='coerce').fillna(0).astype(int)
    df['사고금액'] = pd.to_numeric(df['사고금액'], errors='coerce').fillna(0).astype(float)
    df['사고율'] = pd.to_numeric(df['사고율'], errors='coerce').fillna(0).astype(float)
    
    return df.reset_index(drop=True)


@st.cache_data
def load_jeonse_rate_data():
    """
    한국부동산원 전세가율 CSV 파일을 인코딩 및 명칭 비일치 예외를 처리하여 로드합니다.
    """
    df = pd.read_csv("data/한국부동산원_전세가율.csv", header=0, encoding="cp949")
    df = df.iloc[2:].reset_index(drop=True)
    
    # 깨진 다중 지역 컬럼명 재정의
    df = df.rename(columns={
        '지역': '광역지자체',
        '지역.1': '대권역',
        '지역.2': '중권역',
        '지역.3': '기초지자체'
    })
    
    # 텍스트 공백 제거
    for col in ['광역지자체', '대권역', '중권역', '기초지자체']:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()
            
    # ⭐ [해결 핵심 1] '서울'과 '서울특별시'를 모두 안전하게 붙잡아오도록 변경
    df = df[df['광역지자체'].str.contains('서울', na=False)]
    
    # ⭐ [해결 핵심 2] 여러 지역 열 중 '종로구', '강서구' 등 '~구'로 끝나는 진짜 자치구 이름을 찾아 대입
    def find_real_gu(row):
        for c in ['기초지자체', '중권역', '대권역']:
            if c in row and str(row[c]).endswith('구'):
                return str(row[c])
        return str(row['기초지자체'])
        
    df['기초지자체'] = df.apply(find_real_gu, axis=1)
    
    # 구 분석에 방해되는 불필요한 행(소계, 서울 자체 행) 제거
    df = df[~df['기초지자체'].str.contains('소계|전체|서울', na=False)]
    
    # 중복 자치구 데이터 정돈
    df = df.drop_duplicates(subset=['기초지자체'])
    
    return df.reset_index(drop=True)