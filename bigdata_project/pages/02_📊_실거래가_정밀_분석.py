import streamlit as st
import plotly.express as px
import pandas as pd
from src.features import load_and_process_real_estate

# 1. 페이지 기본 설정 및 브라우저 탭 데코레이션
st.set_page_config(
    page_title="서울시 빌라 전세사기 위험도 대시보드",
    page_icon="🚨",
    layout="wide"
)

# 2. 사이드바 내부 추가 정돈
with st.sidebar:
    st.markdown("### ⚙️ 분석 시스템 안내")
    st.caption("본 대시보드는 국토교통부 실거래가 17만 건을 실시간으로 추적하는 정밀 분석 시스템입니다.")
    st.markdown("---")
    st.markdown("📅 **데이터 기준:** 2025년 전체 계약분")

# 3. 메인 화면 상단 프리미엄 배너 
st.markdown("""
    <div style="background-color:#F8F9FA; padding:22px; border-radius:12px; border-left: 6px solid #FF4B4B; margin-bottom: 25px;">
        <h1 style="margin:0; color:#31333F; font-size:26px; font-weight:700;">🚨 서울시 연립다세대 전세 사기 위험도 정밀 분석</h1>
        <p style="margin:6px 0 0 0; color:#555555; font-size:14px;">국토교통부 대용량 로우 데이터를 기반으로 행정동별 <b>깡통전세 위험 지수(Risk Score)</b>를 실시간 연산합니다.</p>
    </div>
""", unsafe_allow_html=True)

# 4. 데이터 로드 및 로딩바 정돈
@st.cache_data
def get_processed_data():
    return load_and_process_real_estate()

with st.status("⚙️ 대용량 엔진 가공 중...", expanded=False) as status:
    try:
        df = get_processed_data()
        status.update(label="✅ 빅데이터 연산 및 분석 엔진 로드 완료", state="complete", expanded=False)
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        st.stop()

# 5. 핵심 지표 요약 
st.markdown("### 📌 핵심 관리 지표")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.container(border=True).metric(
        label="🎯 총 분석 관측치", 
        value=f"{len(df):,} 개", 
        help="구-동-연월 조건이 매칭된 총 결합 데이터 수"
    )
with col2:
    mean_val = df['전세가율'].mean()
    st.container(border=True).metric(
        label="📈 서울시 평균 전세가율", 
        value=f"{mean_val:.1f}%",
        delta=f"{mean_val - 80:.1f}%" if mean_val > 80 else f"{mean_val - 80:.1f}%",
        delta_color="inverse"
    )
with col3:
    st.container(border=True).metric(
        label="🔥 최고 위험지역 점수", 
        value=f"{df['위험점수'].max():.1f} 점"
    )
with col4:
    danger_count = len(df[df['위험점수'] >= 80])
    st.container(border=True).metric(
        label="🚨 위험 관리 동(Dong) 수", 
        value=f"{danger_count} 개 지역",
        help="위험 점수가 학계 기준인 80점 이상을 초과한 지역 수"
    )

st.markdown("")

# 6. 메인 콘텐츠를 '탭(Tabs)' 구조로 분리하여 시각적 피로도 감소 
tab1, tab2 = st.tabs(["🔥 위험 지역 Top 10 시각화", "🔍 우리 동네 트렌드 검색기"])

with tab1:
    st.markdown("#### 📊 전세 사기 위험도 상위 10개 행정동")
    df['지역명'] = df['구'] + " " + df['동']
    top_10 = df.groupby('지역명')['위험점수'].mean().sort_values(ascending=False).head(10).reset_index()
    
    # 깔끔한 테마의 Plotly 차트
    fig_bar = px.bar(
        top_10, 
        x='위험점수', 
        y='지역명', 
        orientation='h',
        color='위험점수', 
        color_continuous_scale='Reds',
        labels={'위험점수': '위험 점수', '지역명': '행정동명'},
        template="plotly_white"
    )
    fig_bar.update_layout(
        yaxis={'categoryorder':'total ascending'}, 
        showlegend=False,
        margin=dict(l=10, r=10, t=10, b=10)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    st.markdown("#### 📍 지역별 전세가율 시계열 추이 상세 검색")
    c1, c2 = st.columns(2)
    with c1:
        gu_list = sorted(df['구'].unique())
        selected_gu = st.selectbox("분석할 '구' 선택", gu_list)
    with c2:
        dong_list = sorted(df[df['구'] == selected_gu]['동'].unique())
        selected_dong = st.selectbox("분석할 '동' 선택", dong_list)
        
    dong_df = df[(df['구'] == selected_gu) & (df['동'] == selected_dong)].sort_values(by='계약년월')
    dong_df['계약년월'] = dong_df['계약년월'].astype(str)

    if not dong_df.empty:
        fig_line = px.line(
            dong_df, 
            x='계약년월', 
            y='전세가율', 
            markers=True,
            text=dong_df['전세가율'].round(1),
            labels={'전세가율': '전세가율 (%)', '계약년월': '계약 연월'},
            template="plotly_white"
        )
        # 붉은색 경계 점선 추가
        fig_line.add_hline(y=80, line_dash="dash", line_color="#FF4B4B", annotation_text="깡통전세 위험선 (80%)")
        fig_line.update_traces(textposition="top center", line_color="#1F77B4")
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.warning("선택하신 동은 분석 가능한 충분한 쌍(매매+전세) 데이터가 존재하지 않습니다.")

# 7. 하단 로우 데이터는 접이식 창(Expander)으로 숨겨서 깔끔하게 정돈
st.markdown("---")
with st.expander("📋 정밀 전처리 완료된 5,000건의 데이터셋 원본 보기"):
    display_df = df[['구', '동', '계약년월', '평균매매가', '평균전세가', '전세가율', '위험점수']].copy()
    display_df['평균매매가'] = display_df['평균매매가'].apply(lambda x: f"{int(x):,} 만원")
    display_df['평균전세가'] = display_df['평균전세가'].apply(lambda x: f"{int(x):,} 만원")
    display_df['전세가율'] = display_df['전세가율'].round(1).astype(str) + " %"
    st.dataframe(display_df, use_container_width=True)