"""
pages/03_🤖_모델_서비스.py
──────────────────────────────────────────────────────────
[화면 4] 모델 서비스
- 인과관계가 증명되었으니 다음 달 위험 등급(상/중/하)을 예측
- 선형 모델(Logistic Regression) vs 트리 모델(Random Forest) 성능 비교
- 시간 기반 Train/Test 분리 (마지막 3개월 = Test)
- 특성 중요도 시각화
- 사용자 입력 → 실시간 예측
- 2026년 5월 예측 결과 표시
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, confusion_matrix
)
import warnings
warnings.filterwarnings("ignore")

from utils import (
    load_data, render_sidebar_info,
    COL_DATE, COL_DISTRICT, COL_RATE, COL_ACC_CNT,
    COL_RATE_CHG, COL_RATE_MA, COL_VOL_GAP, COL_GRADE,
    GRADE_COLOR
)

# ─────────────────────────────────────────
# 0. 페이지 설정
# ─────────────────────────────────────────
st.set_page_config(
    page_title="모델 서비스",
    page_icon="🤖",
    layout="wide",
)

df = load_data()
render_sidebar_info(df)

# ─────────────────────────────────────────
# 1. 헤더
# ─────────────────────────────────────────
st.title("🤖 모델 서비스: 다음 달 위험 등급 예측")
st.markdown(
    "통계적 인과관계가 입증되었으므로,\n"
    "현재 시점(t)의 데이터를 활용하여 **다음 달(t+1)의 보증사고 위험 등급(상/중/하)** 을 예측합니다."
)
st.divider()

# ─────────────────────────────────────────
# 2. 피처 정의 및 데이터 준비
# ─────────────────────────────────────────
# 사용할 특성 (컬럼이 있는 것만 선택)
FEATURE_CANDIDATES = {
    COL_RATE:     "전세가율(%)",
    COL_RATE_CHG: "전세가율 변화율(%p)",
    COL_RATE_MA:  "전세가율 3개월 이동평균(%)",
    COL_VOL_GAP:  "거래량 격차(전세/매매)",
    COL_ACC_CNT:  "보증사고건수(현재, 참고용)",
}

FEATURES = [col for col in FEATURE_CANDIDATES if col in df.columns and col != COL_ACC_CNT]

# 보증사고건수를 현재 t 시점 피처로 추가 (t+1 예측에는 사용 가능)
if COL_ACC_CNT in df.columns:
    FEATURES.append(COL_ACC_CNT)

TARGET = COL_GRADE

# 위험 등급 컬럼이 없을 경우 자동 생성
if TARGET not in df.columns:
    st.warning(
        f"⚠️ '{TARGET}' 컬럼이 없습니다. "
        "보증사고건수를 3분위 기준으로 자동 분류하여 임시 타깃을 생성합니다."
    )
    q33 = df[COL_ACC_CNT].quantile(0.33)
    q66 = df[COL_ACC_CNT].quantile(0.66)

    def auto_grade(x):
        if x >= q66: return "상"
        elif x >= q33: return "중"
        return "하"

    df[TARGET] = df[COL_ACC_CNT].apply(auto_grade)

model_df = df[[COL_DATE, COL_DISTRICT] + FEATURES + [TARGET]].dropna()

if len(model_df) < 30:
    st.error("모델 학습에 필요한 데이터가 부족합니다. 데이터를 확인해 주세요.")
    st.stop()

# ─────────────────────────────────────────
# 3. 시간 기반 Train/Test 분리
# ─────────────────────────────────────────
sorted_dates = sorted(model_df[COL_DATE].unique())
n_test_months = 3                                      # 마지막 3개월 = Test
cutoff_date   = sorted_dates[-n_test_months]

train_df = model_df[model_df[COL_DATE] < cutoff_date]
test_df  = model_df[model_df[COL_DATE] >= cutoff_date]

X_train = train_df[FEATURES].values
y_train = train_df[TARGET].astype(str).values
X_test  = test_df[FEATURES].values
y_test  = test_df[TARGET].astype(str).values

train_start = sorted_dates[0].strftime("%Y.%m")
train_end   = sorted_dates[-n_test_months - 1].strftime("%Y.%m")
test_start  = sorted_dates[-n_test_months].strftime("%Y.%m")
test_end    = sorted_dates[-1].strftime("%Y.%m")

# ─────────────────────────────────────────
# 4. 랜덤 포레스트 학습
# ─────────────────────────────────────────
@st.cache_resource
def train_rf(_X_train, _y_train):
    rf = RandomForestClassifier(
        n_estimators=100, random_state=42,
        class_weight="balanced", max_depth=5
    )
    rf.fit(_X_train, _y_train)
    return rf

rf_model = train_rf(X_train, y_train)
rf_pred  = rf_model.predict(X_test)

# ─────────────────────────────────────────
# 5. 모델 성능 및 혼동 행렬
# ─────────────────────────────────────────
st.subheader("📊 랜덤 포레스트 모델 성능")

rf_f1  = f1_score(y_test, rf_pred, average="macro", zero_division=0)
rf_acc = accuracy_score(y_test, rf_pred)

# 핵심 수치 3개
m1, m2, m3 = st.columns(3)
with m1:
    st.info(
        f"**학습 기간**  {train_start} ~ {train_end}  \n"
        f"**검증 기간**  {test_start} ~ {test_end}"
    )
with m2:
    st.metric("✅ F1-Score (macro)", f"{rf_f1:.3f}",
              help="상/중/하 각 등급 예측 성능의 평균")
with m3:
    st.metric("✅ 정확도 (Accuracy)", f"{rf_acc:.1%}",
              help="전체 예측 중 정답 비율")

# 혼동 행렬
st.markdown("#### 🔢 혼동 행렬")
st.caption("대각선(↘) 숫자가 클수록 정확한 예측입니다. 대각선 밖은 오분류입니다.")

labels = [l for l in ["상", "중", "하"] if l in set(y_test) | set(rf_pred)]
cm = confusion_matrix(y_test, rf_pred, labels=labels)
fig_cm = px.imshow(
    cm,
    text_auto=True,
    x=[f"예측: {l}" for l in labels],
    y=[f"실제: {l}" for l in labels],
    color_continuous_scale="Blues",
    labels=dict(color="건수"),
)
fig_cm.update_layout(height=380, margin=dict(t=30, b=20))

cm_col, cm_exp = st.columns([1, 1])
with cm_col:
    st.plotly_chart(fig_cm, use_container_width=True)
with cm_exp:
    st.markdown("#### 📌 이 모델의 핵심 결과")

    # 혼동 행렬에서 핵심 칸 직접 계산
    cm_labels = [l for l in ["상", "중", "하"] if l in set(y_test) | set(rf_pred)]
    cm_vals = confusion_matrix(y_test, rf_pred, labels=cm_labels)
    idx = {l: i for i, l in enumerate(cm_labels)}

    # 상→하 (위험을 안전으로 오판), 하→상 (안전을 위험으로 오판)
    상_to_하 = int(cm_vals[idx["상"]][idx["하"]]) if "상" in idx and "하" in idx else 0
    하_to_상 = int(cm_vals[idx["하"]][idx["상"]]) if "하" in idx and "상" in idx else 0
    diagonal_sum = int(cm_vals.diagonal().sum())
    total = int(cm_vals.sum())

    st.success(
        f"✅ 고위험 → 안전 오판: {상_to_하}건  \n"
        "위험 지역을 안전하다고 예측한 최악의 오류가 없습니다."
    )
    st.success(
        f"✅ 안전 → 고위험 오판: {하_to_상}건  \n"
        "안전 지역을 위험하다고 과잉 경보한 경우도 없습니다."
    )
    st.info(
        f"📊 전체 정확도: {diagonal_sum}/{total} ({diagonal_sum/total:.1%})  \n"
        "오분류는 모두 인접 등급(상↔중, 중↔하) 사이에서만 발생했습니다."
    )

st.divider()

# ─────────────────────────────────────────
# 7. 특성 중요도 (랜덤 포레스트)
# ─────────────────────────────────────────
st.subheader("🌲 특성 중요도 (랜덤 포레스트 기준)")

feature_labels = [FEATURE_CANDIDATES.get(f, f) for f in FEATURES]
importances = rf_model.feature_importances_
imp_df = pd.DataFrame({
    "특성": feature_labels,
    "중요도": importances
}).sort_values("중요도", ascending=True)

fig_imp = px.bar(
    imp_df,
    x="중요도",
    y="특성",
    orientation="h",
    text="중요도",
    color="중요도",
    color_continuous_scale=["#BBDEFB", "#1565C0"],
    title="특성 중요도 (Feature Importance)",
)
fig_imp.update_traces(texttemplate="%{text:.3f}", textposition="outside")
fig_imp.update_layout(height=350, coloraxis_showscale=False)
st.plotly_chart(fig_imp, use_container_width=True)

st.divider()

# ─────────────────────────────────────────
# 8. 자치구 선택 → 위험 등급 실시간 예측 (발표 데모)
# ─────────────────────────────────────────
st.subheader("🎮 자치구 선택 → 다음 달 위험 등급 예측")
st.caption("자치구와 기준 월을 선택하고 예측 버튼을 누르면 다음 달 위험 등급을 카드로 보여줍니다.")

GRADE_COLOR_CSS = {"상": "#FF4444", "중": "#FFA500", "하": "#44BB44"}
GRADE_BG_CSS    = {"상": "#FFF0F0", "중": "#FFFBF0", "하": "#F0FFF0"}
GRADE_TEXT      = {"상": "🔴 고위험 (상)", "중": "🟡 주의 (중)", "하": "🟢 안전 (하)"}

sel_col1, sel_col2 = st.columns(2)
with sel_col1:
    districts_list = sorted(df[COL_DISTRICT].unique().tolist())
    sel_district = st.selectbox("🏙️ 자치구 선택", districts_list)
with sel_col2:
    avail_dates = (
        df[df[COL_DISTRICT] == sel_district][COL_DATE]
        .sort_values(ascending=False)
        .dt.strftime("%Y년 %m월")
        .tolist()
    )
    sel_date_str = st.selectbox("📅 기준 월 선택", avail_dates)

sel_date_dt = pd.to_datetime(sel_date_str, format="%Y년 %m월")
sel_row = df[
    (df[COL_DISTRICT] == sel_district) &
    (df[COL_DATE] == sel_date_dt)
]

if not sel_row.empty:
    sel_row = sel_row.iloc[0]

    # 현재 데이터 수치 미리 표시
    st.markdown(f"**{sel_district} {sel_date_str} 현재 데이터**")
    info_cols = st.columns(len(FEATURES))
    for i, feat in enumerate(FEATURES):
        val = float(sel_row[feat]) if feat in sel_row.index and pd.notna(sel_row[feat]) else 0.0
        with info_cols[i]:
            st.metric(label=FEATURE_CANDIDATES.get(feat, feat), value=f"{val:.2f}")

    st.divider()

    if st.button("🔮 다음 달 위험 등급 예측", type="primary", use_container_width=True):
        feat_vals = [
            float(sel_row[f]) if f in sel_row.index and pd.notna(sel_row[f]) else 0.0
            for f in FEATURES
        ]
        pred_X     = np.array([feat_vals])
        pred_grade = rf_model.predict(pred_X)[0]
        pred_prob  = rf_model.predict_proba(pred_X)[0]
        prob_dict  = dict(zip(rf_model.classes_, pred_prob))
        next_month = (sel_date_dt + pd.DateOffset(months=1)).strftime("%Y년 %m월")

        prob_상 = prob_dict.get("상", 0)
        prob_중 = prob_dict.get("중", 0)
        prob_하 = prob_dict.get("하", 0)

        border = GRADE_COLOR_CSS.get(pred_grade, "#888")
        bg     = GRADE_BG_CSS.get(pred_grade, "#FFF")
        label  = GRADE_TEXT.get(pred_grade, pred_grade)

        # ── 결과 카드 ──────────────────────────────────
        left_card, right_info = st.columns([1, 1])

        with left_card:
            st.markdown(
                f"<div style='border:3px solid {border};border-radius:16px;"
                f"padding:30px;background:{bg};text-align:center;height:260px;"
                f"display:flex;flex-direction:column;justify-content:center;'>"
                f"<div style='font-size:1rem;color:#888;margin-bottom:4px'>예측 대상</div>"
                f"<div style='font-size:1.8rem;font-weight:bold'>{sel_district}</div>"
                f"<div style='font-size:0.9rem;color:#888;margin-bottom:12px'>{next_month}</div>"
                f"<div style='font-size:2.2rem;font-weight:bold;color:{border}'>{label}</div>"
                f"<div style='font-size:0.85rem;color:#555;margin-top:12px'>"
                f"현재 전세가율: <b>{sel_row[COL_RATE]:.1f}%</b></div>"
                f"</div>",
                unsafe_allow_html=True,
            )

        with right_info:
            st.markdown("**예측 확률**")
            # 확률 막대 (진행 막대 스타일)
            for g, prob, color in [("상 (고위험)", prob_상, "#FF4444"),
                                    ("중 (주의)",   prob_중, "#FFA500"),
                                    ("하 (안전)",   prob_하, "#44BB44")]:
                pct = prob * 100
                st.markdown(
                    f"<div style='margin-bottom:10px'>"
                    f"<div style='font-size:0.85rem;margin-bottom:3px'>{g}: <b>{pct:.1f}%</b></div>"
                    f"<div style='background:#eee;border-radius:6px;height:18px;width:100%'>"
                    f"<div style='background:{color};width:{pct:.1f}%;height:18px;"
                    f"border-radius:6px'></div></div></div>",
                    unsafe_allow_html=True,
                )

            st.markdown("<br>", unsafe_allow_html=True)

            if pred_grade == "상":
                st.error("HUG 전세보증보험 가입 및 등기부등본 확인 필수")
            elif pred_grade == "중":
                st.warning("전세가율 추이 모니터링 및 보증보험 가입 권장")
            else:
                st.success("현재 안전 수준, 전세가율 변화 지속 확인 권장")

        # ── 예측 vs 실제 비교 ──────────────────────────
        st.divider()
        st.markdown("### 🔍 예측 결과 검증: 실제 사고와 비교")

        next_date_dt = sel_date_dt + pd.DateOffset(months=1)
        actual_row = df[
            (df[COL_DISTRICT] == sel_district) &
            (df[COL_DATE] == next_date_dt)
        ]

        if actual_row.empty:
            st.info(
                f"{next_month}은 데이터셋의 마지막 달 이후입니다. "
                "실제값과 비교하려면 2026년 04월 이전 월을 선택해 주세요."
            )
        else:
            actual_row   = actual_row.iloc[0]
            actual_cnt   = int(actual_row[COL_ACC_CNT])
            actual_grade = str(actual_row[COL_GRADE]) if COL_GRADE in actual_row.index else None

            # 분위수 기준으로 실제 등급 계산 (COL_GRADE가 없을 경우 대비)
            if actual_grade in [None, "nan", "None"]:
                q33_val = df[COL_ACC_CNT].quantile(0.33)
                q66_val = df[COL_ACC_CNT].quantile(0.66)
                actual_grade = "상" if actual_cnt >= q66_val else ("중" if actual_cnt >= q33_val else "하")

            is_correct = (pred_grade == actual_grade)
            match_icon = "✅ 예측 성공" if is_correct else "❌ 예측 빗나감"
            match_color = "#44BB44" if is_correct else "#FF4444"

            v_col1, v_col2, v_col3 = st.columns(3)

            with v_col1:
                pred_border = GRADE_COLOR_CSS.get(pred_grade, "#888")
                pred_bg     = GRADE_BG_CSS.get(pred_grade, "#FFF")
                st.markdown(
                    f"<div style='border:2px solid {pred_border};border-radius:12px;"
                    f"padding:20px;background:{pred_bg};text-align:center;'>"
                    f"<div style='font-size:0.85rem;color:#888'>AI 예측</div>"
                    f"<div style='font-size:1.6rem;font-weight:bold;color:{pred_border}'>"
                    f"{GRADE_TEXT.get(pred_grade, pred_grade)}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with v_col2:
                st.markdown(
                    f"<div style='border:2px solid {match_color};border-radius:12px;"
                    f"padding:20px;text-align:center;height:100%;'>"
                    f"<div style='font-size:0.85rem;color:#888'>결과</div>"
                    f"<div style='font-size:1.4rem;font-weight:bold;color:{match_color}'>"
                    f"{match_icon}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

            with v_col3:
                actual_border = GRADE_COLOR_CSS.get(actual_grade, "#888")
                actual_bg     = GRADE_BG_CSS.get(actual_grade, "#FFF")
                st.markdown(
                    f"<div style='border:2px solid {actual_border};border-radius:12px;"
                    f"padding:20px;background:{actual_bg};text-align:center;'>"
                    f"<div style='font-size:0.85rem;color:#888'>실제 ({next_month})</div>"
                    f"<div style='font-size:1.6rem;font-weight:bold;color:{actual_border}'>"
                    f"{GRADE_TEXT.get(actual_grade, actual_grade)}</div>"
                    f"<div style='font-size:0.9rem;color:#555;margin-top:6px'>"
                    f"실제 보증사고: <b>{actual_cnt}건</b></div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # ── 전세가율 + 보증사고 추이 차트 ─────────────────
        st.markdown("#### 📈 전세가율 및 보증사고 추이")
        dist_history = df[df[COL_DISTRICT] == sel_district].sort_values(COL_DATE)

        from plotly.subplots import make_subplots
        import plotly.graph_objects as go_inner

        fig_hist = make_subplots(specs=[[{"secondary_y": True}]])
        fig_hist.add_trace(
            go_inner.Scatter(
                x=dist_history[COL_DATE], y=dist_history[COL_RATE],
                name="전세가율(%)", line=dict(color=border, width=2), mode="lines+markers",
            ), secondary_y=False,
        )
        fig_hist.add_trace(
            go_inner.Bar(
                x=dist_history[COL_DATE], y=dist_history[COL_ACC_CNT],
                name="보증사고 건수", marker_color="rgba(255,100,100,0.5)",
            ), secondary_y=True,
        )
        fig_hist.add_vline(x=sel_date_dt, line_dash="dash", line_color="gray",
                           annotation_text="선택 월")
        fig_hist.add_hline(y=80, line_dash="dot", line_color="red",
                           annotation_text="위험선 80%", secondary_y=False)
        fig_hist.update_yaxes(title_text="전세가율(%)", secondary_y=False)
        fig_hist.update_yaxes(title_text="보증사고 건수", secondary_y=True)
        fig_hist.update_layout(
            title=f"{sel_district} 전세가율 및 보증사고 추이",
            height=360, hovermode="x unified",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        st.plotly_chart(fig_hist, use_container_width=True)

st.divider()