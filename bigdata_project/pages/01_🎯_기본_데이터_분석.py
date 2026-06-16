"""
pages/01_🎯_기본_데이터_분석.py
──────────────────────────────────────────────────────────
[화면 2] 기본 데이터 분석
- 어느 동네가 가장 심하고 과거부터 지금까지 어떻게 변해왔나?
- 위험 자치구 Top 5 랭킹 차트
- 개별 자치구 시계열 추이 (전세가율 + 보증사고)
- 최악의 시점 기록 테이블
──────────────────────────────────────────────────────────
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils import (
    load_data, render_sidebar_info, rate_to_label,
    COL_DATE, COL_DISTRICT, COL_RATE,
    COL_ACC_CNT, COL_ACC_AMT, COL_RATE_CHG, COL_RATE_MA,
    GRADE_COLOR
)

# ─────────────────────────────────────────
# 0. 페이지 설정
# ─────────────────────────────────────────
st.set_page_config(
    page_title="기본 데이터 분석",
    page_icon="🎯",
    layout="wide",
)

df = load_data()
render_sidebar_info(df)

latest_month = df[COL_DATE].max()

# ─────────────────────────────────────────
# 1. 헤더
# ─────────────────────────────────────────
st.title("🎯 기본 데이터 분석")
st.markdown("어느 자치구가 가장 심각하며, 시간의 흐름에 따라 어떻게 변해왔는지 파악합니다.")
st.divider()

# ═══════════════════════════════════════════
# [섹션 1] 위험 자치구 Top 5 랭킹
# ═══════════════════════════════════════════
st.subheader("🏆 위험 자치구 Top 5 랭킹")
st.caption("기준: 분석 기간 전체 누적 보증사고 건수 + 평균 전세가율 종합")

col_rank, col_exp = st.columns([3, 1])

with col_rank:
    # 자치구별 집계
    district_agg = (
        df.groupby(COL_DISTRICT)
        .agg(
            누적_보증사고건수=(COL_ACC_CNT, "sum"),
            평균_전세가율=(COL_RATE, "mean"),
            최고_전세가율=(COL_RATE, "max"),
        )
        .reset_index()
    )

    # Top5 (누적 보증사고 기준)
    top5_acc = district_agg.nlargest(5, "누적_보증사고건수").copy()
    top5_acc["순위"] = range(1, 6)
    top5_acc["순위_레이블"] = top5_acc["순위"].apply(
        lambda x: {1: "🥇", 2: "🥈", 3: "🥉"}.get(x, f"{x}위")
    )

    fig_top5 = go.Figure()
    fig_top5.add_trace(go.Bar(
        y=top5_acc[COL_DISTRICT][::-1],
        x=top5_acc["누적_보증사고건수"][::-1],
        orientation="h",
        name="누적 보증사고 건수",
        marker_color=["#FF4444", "#FF6666", "#FF8888", "#FFAAAA", "#FFCCCC"][::-1],
        text=top5_acc["누적_보증사고건수"][::-1],
        textposition="outside",
    ))
    fig_top5.update_layout(
        title="누적 보증사고 건수 Top 5",
        xaxis_title="누적 보증사고 건수",
        yaxis_title="",
        height=350,
        margin=dict(l=10, r=60, t=50, b=20),
    )
    st.plotly_chart(fig_top5, use_container_width=True)

    # Top5 상세 테이블
    display_df = top5_acc[["순위_레이블", COL_DISTRICT, "누적_보증사고건수", "평균_전세가율", "최고_전세가율"]].copy()
    display_df.columns = ["순위", "자치구", "누적 보증사고 건수", "평균 전세가율(%)", "최고 전세가율(%)"]
    display_df["평균 전세가율(%)"] = display_df["평균 전세가율(%)"].round(1)
    display_df["최고 전세가율(%)"] = display_df["최고 전세가율(%)"].round(1)
    st.dataframe(display_df, hide_index=True, use_container_width=True)

with col_exp:
    st.markdown("#### 📌 해석 방법")
    st.info(
        "**누적 보증사고 건수**는\n"
        "HUG가 실제 대위변제한\n"
        "사건 수의 합계입니다.\n\n"
        "이 지표는 **예측값이 아닌\n"
        "실제 피해 팩트**입니다.\n\n"
        "→ 누적 사고가 많은 구는\n"
        "악성 임대인이 집중된\n"
        "가능성이 높습니다."
    )

st.divider()

# ═══════════════════════════════════════════
# [섹션 2] 개별 자치구 시계열 추이
# ═══════════════════════════════════════════
st.subheader("📈 개별 자치구 시계열 추이")
st.caption("자치구를 선택하면 전세가율과 보증사고 건수의 시간 흐름을 확인할 수 있습니다.")

# 자치구 선택 (기본값: 누적 사고 1위)
districts = sorted(df[COL_DISTRICT].unique().tolist())
default_district = top5_acc[COL_DISTRICT].iloc[0] if len(top5_acc) > 0 else districts[0]

col_sel1, col_sel2 = st.columns([2, 2])
with col_sel1:
    selected = st.selectbox("🏙️ 자치구 선택", districts, index=districts.index(default_district))
with col_sel2:
    compare = st.selectbox(
        "📊 비교할 자치구 선택 (선택 사항)",
        ["없음"] + districts,
        index=0
    )

# 선택 구 데이터
sel_df = df[df[COL_DISTRICT] == selected].sort_values(COL_DATE)

# ── 이중 Y축 차트 (전세가율 + 보증사고 건수) ──
fig_ts = make_subplots(specs=[[{"secondary_y": True}]])

# 전세가율 - 선
fig_ts.add_trace(
    go.Scatter(
        x=sel_df[COL_DATE], y=sel_df[COL_RATE],
        name=f"{selected} 전세가율",
        line=dict(color="#2196F3", width=2),
        mode="lines+markers",
    ),
    secondary_y=False,
)

# 3개월 이동평균 - 점선 (컬럼 존재 시)
if COL_RATE_MA in sel_df.columns:
    fig_ts.add_trace(
        go.Scatter(
            x=sel_df[COL_DATE], y=sel_df[COL_RATE_MA],
            name=f"{selected} 3개월 이동평균",
            line=dict(color="#2196F3", width=1.5, dash="dot"),
            mode="lines",
        ),
        secondary_y=False,
    )

# 보증사고 건수 - 막대
fig_ts.add_trace(
    go.Bar(
        x=sel_df[COL_DATE], y=sel_df[COL_ACC_CNT],
        name=f"{selected} 보증사고 건수",
        marker_color="rgba(255, 100, 100, 0.5)",
    ),
    secondary_y=True,
)

# 비교 구 추가
if compare != "없음":
    cmp_df = df[df[COL_DISTRICT] == compare].sort_values(COL_DATE)
    fig_ts.add_trace(
        go.Scatter(
            x=cmp_df[COL_DATE], y=cmp_df[COL_RATE],
            name=f"{compare} 전세가율",
            line=dict(color="#FF9800", width=2, dash="dash"),
            mode="lines+markers",
        ),
        secondary_y=False,
    )

# 위험선 80%
fig_ts.add_hline(
    y=80, line_dash="dash", line_color="red",
    annotation_text="⚠️ 위험선 80%",
    annotation_position="right",
    secondary_y=False,
)

fig_ts.update_yaxes(title_text="전세가율 (%)", secondary_y=False)
fig_ts.update_yaxes(title_text="보증사고 건수", secondary_y=True)
fig_ts.update_xaxes(title_text="기준 년월")
fig_ts.update_layout(
    title=f"{selected} 전세가율 및 보증사고 추이",
    height=480,
    hovermode="x unified",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)
st.plotly_chart(fig_ts, use_container_width=True)

# ── 선택 구 전세가율 변화율 (전월 대비) ──
if COL_RATE_CHG in df.columns:
    chg_df = sel_df[[COL_DATE, COL_RATE_CHG]].dropna()
    fig_chg = px.bar(
        chg_df,
        x=COL_DATE, y=COL_RATE_CHG,
        title=f"{selected} 전세가율 변화율 (전월 대비)",
        color=COL_RATE_CHG,
        color_continuous_scale=["#44BB44", "#FFFFFF", "#FF4444"],
        color_continuous_midpoint=0,
        labels={COL_DATE: "기준 년월", COL_RATE_CHG: "변화율 (%p)"},
    )
    fig_chg.add_hline(y=0, line_color="gray")
    fig_chg.update_layout(height=300, coloraxis_showscale=False)
    st.plotly_chart(fig_chg, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════
# [섹션 3] 최악의 시점 기록 테이블
# ═══════════════════════════════════════════
st.subheader("📋 최악의 시점 기록")
st.caption("자치구별로 전세가율이 최고였던 시점과 보증사고 최다 시점을 정리합니다.")

tab1, tab2 = st.tabs(["📈 전세가율 최고 시점", "🚨 보증사고 최다 시점"])

with tab1:
    # NaN이 아닌 행만 대상으로 자치구별 전세가율 최대 인덱스 추출
    _rate_valid = df[df[COL_RATE].notna()]
    _rate_idx   = _rate_valid.groupby(COL_DISTRICT)[COL_RATE].idxmax().dropna()
    worst_rate = (
        df.loc[_rate_idx]
        [[COL_DISTRICT, COL_DATE, COL_RATE, COL_ACC_CNT]]
        .sort_values(COL_RATE, ascending=False)
        .reset_index(drop=True)
    )
    worst_rate.index += 1
    worst_rate[COL_DATE] = worst_rate[COL_DATE].dt.strftime("%Y년 %m월")
    worst_rate.columns = ["자치구", "최고 전세가율 기록 월", "전세가율(%)", "해당 월 보증사고 건수"]
    worst_rate["전세가율(%)"] = worst_rate["전세가율(%)"].round(1)

    def highlight_danger(val):
        if isinstance(val, float) and val >= 80:
            return "background-color: #FFE0E0; color: #CC0000; font-weight: bold"
        return ""

    st.dataframe(
        worst_rate.style.applymap(highlight_danger, subset=["전세가율(%)"]),
        use_container_width=True,
        height=500,
    )

with tab2:
    # NaN이 아닌 행만 대상으로 자치구별 보증사고건수 최대 인덱스 추출
    _acc_valid = df[df[COL_ACC_CNT].notna()]
    _acc_idx   = _acc_valid.groupby(COL_DISTRICT)[COL_ACC_CNT].idxmax().dropna()
    worst_acc = (
        df.loc[_acc_idx]
        [[COL_DISTRICT, COL_DATE, COL_ACC_CNT, COL_RATE]]
        .sort_values(COL_ACC_CNT, ascending=False)
        .reset_index(drop=True)
    )
    worst_acc.index += 1
    worst_acc[COL_DATE] = worst_acc[COL_DATE].dt.strftime("%Y년 %m월")
    worst_acc.columns = ["자치구", "보증사고 최다 월", "보증사고 건수", "해당 월 전세가율(%)"]
    worst_acc["해당 월 전세가율(%)"] = worst_acc["해당 월 전세가율(%)"].round(1)

    st.dataframe(worst_acc, use_container_width=True, height=500)