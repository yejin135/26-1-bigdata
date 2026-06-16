"""
🏠_메인_화면.py
──────────────────────────────────────────────────────────
[화면 1] 메인 화면
- 서울시 전체 거시적 위험 규모를 한눈에 파악
- 누적 보증사고 건수 / 최고 전세가율 구 / 위험 자치구 수
- 자치구별 현재 전세가율 현황 (수평 막대)
- 서울 전체 월별 보증사고 추이
──────────────────────────────────────────────────────────
실행: streamlit run 🏠_메인_화면.py
"""

import sys
from pathlib import Path

# src 모듈 경로 추가
sys.path.append(str(Path(__file__).parent / "src"))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils import (
    load_data, render_sidebar_info, rate_to_label,
    COL_DATE, COL_DISTRICT, COL_RATE,
    COL_ACC_CNT, COL_ACC_AMT, RATE_COLOR
)

# ─────────────────────────────────────────
# 0. 페이지 설정
# ─────────────────────────────────────────
st.set_page_config(
    page_title="서울 전세 위험 분석 대시보드",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────
# 1. 데이터 로드
# ─────────────────────────────────────────
df = load_data()

latest_month = df[COL_DATE].max()
latest_df    = df[df[COL_DATE] == latest_month].copy()

render_sidebar_info(df)

# ─────────────────────────────────────────
# 2. 헤더
# ─────────────────────────────────────────
st.title("🏠 서울 전세 위험 분석 대시보드")
st.markdown(
    f"**기준 시점:** {latest_month.strftime('%Y년 %m월')}  |  "
    f"**분석 대상:** 서울시 연립·다세대 주택  |  "
    f"**⚠️ 전세가율 80% 이상 = 깡통전세 위험**"
)
st.divider()

# ─────────────────────────────────────────
# 3. KPI 카드 (4개)
# ─────────────────────────────────────────
total_acc  = int(df[COL_ACC_CNT].sum())                         # 누적 보증사고
month_acc  = int(latest_df[COL_ACC_CNT].sum())                  # 이번 달 사고
danger_cnt = int((latest_df[COL_RATE] >= 80).sum())             # 위험 구 수

_rate_valid  = latest_df[latest_df[COL_RATE].notna()]   # NaN 행 제외
if len(_rate_valid) > 0:
    max_row      = _rate_valid.loc[_rate_valid[COL_RATE].idxmax()]
    max_district = max_row[COL_DISTRICT]
    max_rate     = max_row[COL_RATE]
else:
    max_district = "데이터 없음"
    max_rate     = 0.0

# 전월 대비 위험 구 수 변화 (delta 표시용)
prev_month = df[df[COL_DATE] < latest_month][COL_DATE].max()
if pd.notna(prev_month):
    prev_df      = df[df[COL_DATE] == prev_month]
    prev_danger  = int((prev_df[COL_RATE] >= 80).sum())
    danger_delta = danger_cnt - prev_danger
else:
    danger_delta = None

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        label="📋 누적 보증사고 건수",
        value=f"{total_acc:,}건",
        help="분석 기간 전체의 HUG 보증사고 합계입니다."
    )
with c2:
    st.metric(
        label="📅 이번 달 보증사고 건수",
        value=f"{month_acc:,}건",
        help=f"{latest_month.strftime('%Y년 %m월')} 기준"
    )
with c3:
    st.metric(
        label="🚨 최고 전세가율 자치구",
        value=f"{max_district}",
        delta=f"{max_rate:.1f}%",
        delta_color="inverse",
        help="현재 시점 전세가율이 가장 높은 자치구"
    )
with c4:
    delta_str = f"{danger_delta:+d}개 (전월 대비)" if danger_delta is not None else None
    st.metric(
        label="⚠️ 위험 자치구 수 (전세가율 ≥ 80%)",
        value=f"{danger_cnt}개",
        delta=delta_str,
        delta_color="inverse",
        help="전세가율 80% 이상 = 깡통전세 위험 구역"
    )

st.divider()

# ─────────────────────────────────────────
# 4. 자치구별 전세가율 현황 (수평 막대)
# ─────────────────────────────────────────
st.subheader(f"📊 {latest_month.strftime('%Y년 %m월')} 서울 자치구별 전세가율 현황")

latest_sorted = latest_df.sort_values(COL_RATE, ascending=True).copy()
latest_sorted["위험구분"] = latest_sorted[COL_RATE].apply(
    lambda x: "🔴 위험 (80% 이상)" if x >= 80
              else ("🟡 주의 (70~80%)" if x >= 70 else "🟢 안전 (70% 미만)")
)
color_map = {
    "🔴 위험 (80% 이상)": "#FF4444",
    "🟡 주의 (70~80%)": "#FFA500",
    "🟢 안전 (70% 미만)": "#44BB44",
}

fig_bar = px.bar(
    latest_sorted,
    x=COL_RATE,
    y=COL_DISTRICT,
    orientation="h",
    color="위험구분",
    color_discrete_map=color_map,
    text=COL_RATE,
    labels={COL_RATE: "전세가율 (%)", COL_DISTRICT: ""},
    title=f"자치구별 전세가율 ({latest_month.strftime('%Y.%m')})",
)
fig_bar.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
fig_bar.add_vline(
    x=80, line_dash="dash", line_color="red",
    annotation_text="⚠️ 위험선 80%",
    annotation_position="top right"
)
fig_bar.update_layout(
    height=700,
    showlegend=True,
    legend_title_text="위험 구분",
    xaxis=dict(range=[0, max(latest_sorted[COL_RATE].max() + 10, 100)]),
    margin=dict(l=10, r=80, t=50, b=20),
)
st.plotly_chart(fig_bar, use_container_width=True)

# ─────────────────────────────────────────
# 5. 서울 전체 월별 보증사고 추이
# ─────────────────────────────────────────
st.subheader("📈 서울 전체 월별 보증사고 추이")

col_left, col_right = st.columns([2, 1])

with col_left:
    monthly_acc = (
        df.groupby(COL_DATE)[COL_ACC_CNT]
        .sum()
        .reset_index()
        .rename(columns={COL_ACC_CNT: "보증사고건수_합계"})
    )
    fig_trend = px.area(
        monthly_acc,
        x=COL_DATE,
        y="보증사고건수_합계",
        title="월별 서울 전체 보증사고 건수",
        color_discrete_sequence=["#FF6B6B"],
        labels={COL_DATE: "기준 년월", "보증사고건수_합계": "보증사고 건수"},
    )
    fig_trend.update_layout(
        xaxis_title="기준 년월",
        yaxis_title="보증사고 건수",
        hovermode="x unified",
    )
    st.plotly_chart(fig_trend, use_container_width=True)

with col_right:
    st.markdown("#### 📌 주요 해설")
    st.info(
        "**HUG 보증사고란?**\n\n"
        "집주인이 전세보증금을 돌려주지 못할 때 "
        "세입자의 요청으로 HUG(주택도시보증공사)가 "
        "대신 변제한 사건입니다.\n\n"
        "→ **실제 피해자가 발생한 팩트 데이터**입니다."
    )
    st.warning(
        "**깡통전세 위험 기준**\n\n"
        "- 🟢 안전: 전세가율 < 70%\n"
        "- 🟡 주의: 70% ≤ 전세가율 < 80%\n"
        "- 🔴 위험: 전세가율 ≥ 80%\n\n"
        "경매 시 낙찰가는 감정가의 70~80% 수준 → "
        "80% 초과 시 세입자는 **무조건 손실**"
    )



# ─────────────────────────────────────────
# 7. 안내 문구
# ─────────────────────────────────────────
st.divider()
st.markdown(
    """
    > **📌 분석 흐름**
    > 서울 전체의 판 보여주기 → **[기본 데이터 분석]** 가장 심한 동네의 역사적 흐름
    > → **[실거래가 정밀 분석]** 전세가율과 사고의 인과관계 증명 → **[모델 서비스]** 미래 예측
    """
)