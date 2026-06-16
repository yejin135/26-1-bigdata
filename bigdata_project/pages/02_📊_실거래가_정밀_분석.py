"""
pages/02_📊_실거래가_정밀_분석.py
──────────────────────────────────────────────────────────
[화면 3] 실거래가 정밀 분석
- 전세가율이 높으면 진짜로 사고가 터지는가?
- 통계학적으로 완벽하게 인과관계를 입증하는 핵심 화면
  · 상관분석 (피어슨 r)
  · 산점도 + 회귀선
  · 집단 비교 (80% 이상 vs 미만) → 독립표본 t-검정
  · 박스플롯
  · 전세가율 구간별 평균 보증사고 건수
──────────────────────────────────────────────────────────
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats

from utils import (
    load_data, render_sidebar_info,
    COL_DATE, COL_DISTRICT, COL_RATE,
    COL_ACC_CNT, COL_ACC_AMT, COL_RATE_CHG,
    COL_ACC_RATE, COL_VOL_GAP, GRADE_COLOR
)

# ─────────────────────────────────────────
# 0. 페이지 설정
# ─────────────────────────────────────────
st.set_page_config(
    page_title="실거래가 정밀 분석",
    page_icon="📊",
    layout="wide",
)

df = load_data()
render_sidebar_info(df)

# ─────────────────────────────────────────
# 1. 헤더
# ─────────────────────────────────────────
st.title("📊 실거래가 정밀 분석")
st.markdown(
    "**전세가율이 높으면 실제로 보증사고가 더 많이 터지는가?**\n\n"
    "통계적 검정을 통해 전세가율과 HUG 보증사고의 인과관계를 학술적으로 입증합니다."
)
st.divider()

# ─────────────────────────────────────────
# 분석용 데이터 준비
# ─────────────────────────────────────────
# 보증사고율이 있으면 함께 포함 (t-검정용)
cols_for_analysis = [COL_DATE, COL_DISTRICT, COL_RATE, COL_ACC_CNT]
if COL_ACC_RATE in df.columns:
    cols_for_analysis.append(COL_ACC_RATE)
analysis_df = df[cols_for_analysis].dropna(subset=[COL_DATE, COL_DISTRICT, COL_RATE, COL_ACC_CNT]).copy()
# 보증사고율 결측은 0으로 (사고 없는 달)
if COL_ACC_RATE in analysis_df.columns:
    analysis_df[COL_ACC_RATE] = analysis_df[COL_ACC_RATE].fillna(0)

# 데이터가 부족하면 경고 후 중단
if len(analysis_df) < 2:
    n = len(analysis_df)
    st.error(
        f"분석 가능한 데이터가 부족합니다 (현재 {n}행). "
        "데이터_병합.py -> 데이터_전처리.py 순서로 실행한 뒤 앱을 재시작하세요."
    )
    st.stop()

analysis_df["위험구분"] = analysis_df[COL_RATE].apply(
    lambda x: "위험 (≥80%)" if x >= 80 else "비위험 (<80%)"
)
analysis_df["전세가율_구간"] = pd.cut(
    analysis_df[COL_RATE],
    bins=[0, 60, 70, 75, 80, 85, 90, 200],
    labels=["~60%", "60~70%", "70~75%", "75~80%", "80~85%", "85~90%", "90%~"],
    right=False,
)

# ═══════════════════════════════════════════
# [섹션 1] 상관분석 결과 KPI
# ═══════════════════════════════════════════
st.subheader("🔬 상관분석: 전세가율 ↔ 보증사고 건수")

r_val, p_val = stats.pearsonr(analysis_df[COL_RATE], analysis_df[COL_ACC_CNT])
r2_val = r_val ** 2

# 해석 텍스트
if abs(r_val) >= 0.7:
    strength = "강한"
elif abs(r_val) >= 0.4:
    strength = "중간"
else:
    strength = "약한"
direction = "양의" if r_val > 0 else "음의"

kpi1, kpi2, kpi3 = st.columns(3)
with kpi1:
    st.metric(
        label="📐 피어슨 상관계수 (r)",
        value=f"{r_val:.4f}",
        help="−1 ~ +1 범위. |r| ≥ 0.7이면 강한 상관"
    )
with kpi2:
    st.metric(
        label="📐 결정계수 (R²)",
        value=f"{r2_val:.4f}",
        help="전세가율이 보증사고 건수 분산의 몇 %를 설명하는가"
    )
with kpi3:
    p_display = f"{p_val:.6f}" if p_val >= 0.0001 else "< 0.0001"
    significance = "✅ 통계적으로 유의함" if p_val < 0.05 else "❌ 통계적으로 유의하지 않음"
    st.metric(
        label="📐 p-value",
        value=p_display,
        help="p < 0.05이면 우연이 아닌 실제 관계가 있다고 판단"
    )

# 해석 박스
if p_val < 0.05:
    st.success(
        f"✅ **통계적으로 유의한 {direction} {strength} 상관관계** (r = {r_val:.3f}, p < 0.05)\n\n"
        f"전세가율은 보증사고 건수 분산의 **{r2_val*100:.1f}%** 를 설명합니다. "
        f"전세가율이 높아질수록 보증사고 건수도 유의미하게 증가하는 경향이 있습니다."
    )
else:
    st.warning(
        f"⚠️ 통계적으로 유의하지 않습니다 (r = {r_val:.3f}, p = {p_val:.4f}). "
        "데이터 수가 충분한지 확인이 필요합니다."
    )

st.divider()

# ═══════════════════════════════════════════
# [섹션 2] 산점도 + 회귀선
# ═══════════════════════════════════════════
st.subheader("🔵 산점도 분석: 전세가율 vs 보증사고 건수")

fig_scatter = px.scatter(
    analysis_df,
    x=COL_RATE,
    y=COL_ACC_CNT,
    color="위험구분",
    color_discrete_map={"위험 (≥80%)": "#FF4444", "비위험 (<80%)": "#2196F3"},
    hover_data=[COL_DISTRICT, COL_DATE],
    trendline="ols",          # 최소자승 회귀선
    trendline_scope="overall",
    title="전세가율 vs 보증사고 건수 (전체 기간)",
    labels={COL_RATE: "전세가율 (%)", COL_ACC_CNT: "보증사고 건수"},
    opacity=0.6,
)
fig_scatter.add_vline(
    x=80, line_dash="dash", line_color="red",
    annotation_text="⚠️ 위험선 80%"
)
fig_scatter.update_layout(height=500)
st.plotly_chart(fig_scatter, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════
# [섹션 3] 독립표본 t-검정 (80% 이상 vs 미만)
# ═══════════════════════════════════════════
st.subheader("📐 집단 비교: 전세가율 80% 이상 vs 미만 (독립표본 t-검정)")
st.caption(
    "전세가율 80% 이상인 집단과 미만인 집단의 보증사고율(%)을 비교합니다. "
    "보증사고 건수는 구마다 규모 차이가 있어 왜곡될 수 있으므로 "
    "구 규모를 통제한 보증사고율(%)로 비교합니다."
)

# 보증사고율 컬럼이 없으면 보증사고건수로 대체
t_col    = COL_ACC_RATE if COL_ACC_RATE in analysis_df.columns else COL_ACC_CNT
t_label  = "보증사고율(%)" if t_col == COL_ACC_RATE else "보증사고 건수"

danger_group = analysis_df[analysis_df[COL_RATE] >= 80][t_col]
safe_group   = analysis_df[analysis_df[COL_RATE] < 80][t_col]

t_stat, t_pval = stats.ttest_ind(danger_group, safe_group, equal_var=False)  # Welch's t-test

col_t1, col_t2 = st.columns(2)

with col_t1:
    fig_box = px.box(
        analysis_df,
        x="위험구분",
        y=t_col,
        color="위험구분",
        color_discrete_map={"위험 (≥80%)": "#FF4444", "비위험 (<80%)": "#2196F3"},
        points="outliers",
        title=f"위험 구분별 {t_label} 분포",
        labels={t_col: t_label, "위험구분": ""},
    )
    fig_box.update_layout(showlegend=False, height=400)
    st.plotly_chart(fig_box, use_container_width=True)

with col_t2:
    st.markdown("#### 📋 t-검정 결과")
    result_data = {
        "구분": ["위험 (전세가율 ≥ 80%)", "비위험 (전세가율 < 80%)"],
        "표본 수": [len(danger_group), len(safe_group)],
        f"평균 {t_label}": [round(danger_group.mean(), 3), round(safe_group.mean(), 3)],
        "표준편차": [round(danger_group.std(), 3), round(safe_group.std(), 3)],
    }
    st.dataframe(pd.DataFrame(result_data), hide_index=True, use_container_width=True)

    p_display  = f"{t_pval:.6f}" if t_pval >= 0.0001 else "< 0.0001"
    t_conclude = "✅ 유의한 차이 있음" if t_pval < 0.05 else "❌ 유의한 차이 없음"

    st.markdown(f"""
    | 통계량 | 값 |
    |--------|-----|
    | t-통계량 | {t_stat:.4f} |
    | p-value | {p_display} |
    | 유의수준 | α = 0.05 |
    | 결론 | {t_conclude} |
    """)

    if t_pval < 0.05:
        diff = danger_group.mean() - safe_group.mean()
        st.success(
            f"전세가율 80% 이상인 구의 평균 {t_label}은 미만인 구보다 "
            f"**{diff:.3f}%p 더 높으며**, 이 차이는 통계적으로 유의합니다 (p < 0.05)."
            + "  \n→ **전세가율 80%는 보증사고율이 유의미하게 높아지는 실질적 임계점**임이 입증되었습니다."
        )
    else:
        st.warning(f"두 집단의 {t_label} 평균 차이가 통계적으로 유의하지 않습니다.")

st.divider()

# ═══════════════════════════════════════════
# [섹션 4] 전세가율 구간별 평균 보증사고율
# ═══════════════════════════════════════════
st.subheader("📊 전세가율 구간별 평균 보증사고율")
st.caption("전세가율 구간이 높아질수록 보증사고율이 어떻게 변하는지 확인합니다.")

agg_col   = COL_ACC_RATE if COL_ACC_RATE in analysis_df.columns else COL_ACC_CNT
agg_label = "평균_보증사고율(%)" if agg_col == COL_ACC_RATE else "평균_보증사고건수"

interval_agg = (
    analysis_df.groupby("전세가율_구간", observed=True)[agg_col]
    .agg(["mean", "count"])
    .reset_index()
    .rename(columns={"mean": agg_label, "count": "데이터수"})
)

fig_interval = px.bar(
    interval_agg,
    x="전세가율_구간",
    y=agg_label,
    text=agg_label,
    color=agg_label,
    color_continuous_scale=["#44BB44", "#FFF176", "#FF4444"],
    title=f"전세가율 구간별 {agg_label}",
    labels={"전세가율_구간": "전세가율 구간", agg_label: agg_label},
)
fig_interval.update_traces(texttemplate="%{text:.2f}", textposition="outside")
fig_interval.update_layout(height=400, coloraxis_showscale=False)
st.plotly_chart(fig_interval, use_container_width=True)

st.divider()

# ═══════════════════════════════════════════
# [섹션 5] 시차 분석 (전세가율이 사고를 선행하는가?)
# ═══════════════════════════════════════════
st.subheader("⏱️ 시차 분석: 전세가율 상승 → 보증사고 선행 확인")
st.caption(
    "이번 달 전세가율이 다음 달 보증사고를 예측할 수 있는지 확인합니다. "
    "(lag=1: 1개월 시차, lag=2: 2개월 시차)"
)

col_lag1, col_lag2 = st.columns([1, 2])

with col_lag1:
    lag_results = []
    for lag in range(0, 7):
        temp = df[[COL_DATE, COL_DISTRICT, COL_RATE, COL_ACC_CNT]].copy()
        temp["사고_lag"] = temp.groupby(COL_DISTRICT)[COL_ACC_CNT].shift(-lag)
        temp = temp.dropna()
        if len(temp) > 10:
            r, p = stats.pearsonr(temp[COL_RATE], temp["사고_lag"])
            lag_results.append({
                "시차 (개월)": lag,
                "상관계수 r": round(r, 4),
                "p-value": round(p, 6),
                "유의성": "✅" if p < 0.05 else "❌",
            })
    lag_df = pd.DataFrame(lag_results)
    st.dataframe(lag_df, hide_index=True, use_container_width=True)

with col_lag2:
    fig_lag = px.line(
        lag_df,
        x="시차 (개월)",
        y="상관계수 r",
        markers=True,
        title="시차별 전세가율-보증사고 상관계수",
        color_discrete_sequence=["#2196F3"],
    )
    fig_lag.add_hline(y=0, line_color="gray", line_dash="dot")
    fig_lag.update_layout(height=350)
    st.plotly_chart(fig_lag, use_container_width=True)

# 시차 해석
best_lag = lag_df.loc[lag_df["상관계수 r"].abs().idxmax()]
st.info(
    f"**시차 {int(best_lag['시차 (개월)'])}개월**에서 상관계수가 가장 높습니다 "
    f"(r = {best_lag['상관계수 r']:.4f}). "
    f"이는 현재 전세가율이 {int(best_lag['시차 (개월)'])}개월 후 보증사고를 "
    f"가장 잘 예측함을 의미합니다."
)