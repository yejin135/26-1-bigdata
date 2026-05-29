import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_loader import load_ratio_data, load_accident_data

# 한글 깨짐 방지 설정 (노트북 환경용 기본 폰트 셋팅)
plt.rcParams['font.family'] = 'Malgun Gothic' 
plt.rcParams['axes.unicode_minus'] = False

st.title("📊 1차 작업: 데이터 요약 및 EDA")
st.markdown("---")

# 1. 데이터 로드 (캐싱 적용됨)
ratio_df = load_ratio_data()
accident_df = load_accident_data()

# 2. 전세가율 데이터 섹션
st.header("💡 1. 한국부동산원 전세가율 데이터")
col1, col2 = st.columns(2)
with col1:
    st.metric("데이터 행/열 크기", f"{ratio_df.shape[0]}행 × {ratio_df.shape[1]}열")
with col2:
    st.metric("결측치 개수", f"{ratio_df.isnull().sum().sum()}개")

st.subheader("📋 데이터 상위 5개 행 확인")
st.dataframe(ratio_df.head())

st.subheader("🔍 변수별 결측치 상세 현황")
st.bar_chart(ratio_df.isnull().sum())


st.markdown("---")


# 3. 보증사고 데이터 섹션
st.header("🚨 2. HUG 보증사고 현황 데이터")
col3, col4 = st.columns(2)
with col3:
    st.metric("데이터 행/열 크기", f"{accident_df.shape[0]}행 × {accident_df.shape[1]}열")
with col4:
    st.metric("결측치 개수", f"{accident_df.isnull().sum().sum()}개")

st.subheader("📋 데이터 상위 5개 행 확인")
st.dataframe(accident_df.head())

st.subheader("🔍 변수별 결측치 상세 현황")
st.bar_chart(accident_df.isnull().sum())