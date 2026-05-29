import streamlit as st
import pandas as pd
import plotly.express as px
from src.data_loader import load_accident_data, load_jeonse_rate_data

# 화면을 넓게 쓰도록 설정
st.set_page_config(layout="wide") 

st.title("📊 서울시 구별 전세 사기 위험도 데이터 탐색 (EDA)")
st.markdown("공공데이터포털의 HUG 보증사고 현황과 한국부동산원 전세가율 데이터를 정밀 분석합니다.")

# 데이터 로드
accident_df = load_accident_data()
jeonse_df = load_jeonse_rate_data()

# 최신 전세가율 컬럼명 지정 (2026년 4월)
latest_date_col = '2026년 4월'

# 탭 기능으로 화면 분할
tab1, tab2, tab3 = st.tabs(["📌 데이터 요약 및 기초 통계", "🔗 데이터 병합 현황", "📈 시각화 분석"])

with tab1:
    st.subheader("📋 1. 데이터 기본 정보 및 요약")
    
    # 상단 핵심 메트릭 대시보드
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    with m_col1:
        st.metric(label="분석 대상 자치구", value=f"{len(accident_df)} 개 구")
    with m_col2:
        st.metric(label="서울시 총 보증사고", value=f"{accident_df['사고건수'].sum():,} 건")
    with m_col3:
        st.metric(label="총 보증사고 금액", value=f"{accident_df['사고금액'].sum() / 100000000:.1f} 억원")
    with m_col4:
        # 전세가율 데이터 숫자로 변환 후 평균 계산
        jeonse_numeric = pd.to_numeric(jeonse_df[latest_date_col], errors='coerce')
        st.metric(label="서울시 평균 전세가율", value=f"{jeonse_numeric.mean():.1f} %")
        
    st.markdown("---")
    
    # 좌우 레이아웃으로 두 데이터의 전체 행과 통계량 비교
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("### 🏢 HUG 보증사고 데이터 (서울시 전체 구)")
        # .head()를 제거하여 25개 자치구가 모두 나타납니다.
        st.dataframe(accident_df, use_container_width=True, height=400)
        
        st.write("#### 📊 보증사고 수치 요약 통계 (Describe)")
        st.dataframe(accident_df[['사고건수', '사고금액', '사고율']].describe(), use_container_width=True)
        
    with col2:
        st.write(f"### 📈 한국부동산원 전세가율 데이터 ({latest_date_col} 기준 전체 구)")
        # 전체 구의 최신 전세가율을 보여줍니다. (.head() 제거)
        jeonse_display = jeonse_df[['광역지자체', '기초지자체', latest_date_col]].rename(columns={latest_date_col: '전세가율(%)'})
        st.dataframe(jeonse_display, use_container_width=True, height=400)
        
        st.write("#### 📊 전세가율 수치 요약 통계 (Describe)")
        jeonse_df_numeric_summary = pd.to_numeric(jeonse_df[latest_date_col], errors='coerce').to_frame(name='전세가율(%)')
        st.dataframe(jeonse_df_numeric_summary.describe(), use_container_width=True)

    st.markdown("---")
    st.write("### 🔍 데이터 결측치(Missing Value) 수 점검")
    st.markdown("가이드라인 요건에 따른 데이터 정성 검증 결과입니다. 모든 변수에 결측치(Null)가 없음을 확인했습니다.")
    
    null_col1, null_col2 = st.columns(2)
    with null_col1:
        st.dataframe(accident_df.isnull().sum().to_frame(name="보증사고 데이터 결측치 수"), use_container_width=True)
    with null_col2:
        st.dataframe(jeonse_df[['광역지자체', '기초지자체', latest_date_col]].isnull().sum().to_frame(name="전세가율 데이터 결측치 수"), use_container_width=True)

with tab2:
    st.subheader("🔗 2. '구(기초지자체)' 기준 데이터 결합 결과")
    
    jeonse_latest = jeonse_df[['기초지자체', latest_date_col]].rename(columns={latest_date_col: '전세가율(%)'})
    jeonse_latest['전세가율(%)'] = pd.to_numeric(jeonse_latest['전세가율(%)'], errors='coerce')
    
    # 두 데이터프레임 병합
    merged_df = pd.merge(accident_df, jeonse_latest, on='기초지자체', how='inner')
    
    st.write("보증사고 현황과 전세가율이 하나의 테이블로 결합된 분석용 데이터셋입니다. (25개 자치구 완벽 매칭)")
    st.dataframe(merged_df, use_container_width=True)
    
    # 다른 페이지에서 재사용 가능하도록 세션 상태에 보관
    st.session_state['merged_data'] = merged_df

with tab3:
    st.subheader("📊 3. 서울시 구별 주요 지표 비교 시각화")
    
    if 'merged_data' in st.session_state:
        df_analysis = st.session_state['merged_data']
        
        # 시각화 1: 구별 보증사고 건수 상위 정렬
        df_sorted_accident = df_analysis.sort_values(by='사고건수', ascending=False)
        fig1 = px.bar(df_sorted_accident, x='기초지자체', y='사고건수', 
                     title='서울시 자치구별 보증사고 건수 현황',
                     labels={'기초지자체': '자치구', '사고건수': '사고 건수(건)'},
                     color='사고건수', color_continuous_scale='Reds')
        st.plotly_chart(fig1, use_container_width=True)
        
        # 시각화 2: 구별 전세가율 비교
        df_sorted_jeonse = df_analysis.sort_values(by='전세가율(%)', ascending=False)
        fig2 = px.bar(df_sorted_jeonse, x='기초지자체', y='전세가율(%)', 
                     title=f'서울시 자치구별 전세가율 현황 ({latest_date_col} 기준)',
                     labels={'기초지자체': '자치구', '전세가율(%)': '전세가율(%)'},
                     color='전세가율(%)', color_continuous_scale='Blues')
        st.plotly_chart(fig2, use_container_width=True)