"""
src/utils.py
──────────────────────────────────────────────────────────
공통 유틸리티: 데이터 로딩, 전처리, 색상 정의
모든 페이지에서 공유하여 사용합니다.
──────────────────────────────────────────────────────────
실제 데이터 컬럼: 자치구 | 년월 | 전세가율 | 보증사고건수
없는 파생 컬럼은 load_data() 안에서 자동 계산합니다.
──────────────────────────────────────────────────────────
"""

import pandas as pd
import streamlit as st
from pathlib import Path

# ═══════════════════════════════════════════════════════
# ▶ 실제 데이터 컬럼명 (확인된 값으로 수정 완료)
# ═══════════════════════════════════════════════════════
COL_DATE       = "년월"              # YYYYMM 숫자형 → datetime으로 변환
COL_DISTRICT   = "자치구"            # 예) 강남구
COL_RATE       = "전세가율"          # float, %
COL_ACC_CNT    = "보증사고건수"      # int

# ── 파생 컬럼 (load_data에서 자동 생성) ──────────────
COL_RATE_CHG   = "전세가율_변화율"   # 전월 대비 증감(%p)
COL_RATE_MA    = "이동평균_전세가율" # 3개월 이동평균
COL_GRADE      = "위험등급"          # 상/중/하 (t+1 타깃)

# ── 원본에 없는 컬럼 (페이지에서 if 체크 후 사용) ───
COL_ACC_AMT    = "보증사고금액"
COL_SALE_VOL   = "매매거래량"
COL_RENT_VOL   = "전세거래량"
COL_VOL_GAP    = "거래량격차"
COL_ACC_RATE   = "보증사고율"

# ═══════════════════════════════════════════════════════
# ▶ 색상 / 레이블 상수
# ═══════════════════════════════════════════════════════
GRADE_COLOR = {"상": "#FF4444", "중": "#FFA500", "하": "#44BB44"}
GRADE_LABEL = {"상": "🔴 상 (고위험)", "중": "🟡 중 (주의)", "하": "🟢 하 (안전)"}
RATE_COLOR  = {"위험": "#FF4444", "주의": "#FFA500", "안전": "#44BB44"}

# ═══════════════════════════════════════════════════════
# ▶ 데이터 경로
# ═══════════════════════════════════════════════════════
BASE_DIR  = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "seoul_market_preprocessed.xlsx"


# ═══════════════════════════════════════════════════════
# ▶ 데이터 로드 + 파생 컬럼 자동 생성
# ═══════════════════════════════════════════════════════
@st.cache_data
def load_data() -> pd.DataFrame:
    """
    전처리 엑셀을 로드하고 파생 컬럼을 자동 생성합니다.

    원본 컬럼: 자치구 | 년월 | 전세가율 | 보증사고건수
    생성 컬럼:
        - 전세가율_변화율  : 자치구별 전월 대비 전세가율 증감(%p)
        - 이동평균_전세가율: 자치구별 최근 3개월 이동평균
        - 위험등급         : t+1 보증사고건수 기준 상/중/하 분류
    """
    df = pd.read_excel(DATA_PATH)

    # ── 1. 날짜 변환: 202405 → datetime ────────────────
    df[COL_DATE] = pd.to_datetime(df[COL_DATE].astype(str), format="%Y%m")

    # ── 2. 결측치 처리 ──────────────────────────────────
    # 전세가율 NaN: 해당 자치구 전체 평균으로 대체
    df[COL_RATE] = df.groupby(COL_DISTRICT)[COL_RATE].transform(
        lambda x: x.fillna(x.mean())
    )
    # 그래도 남은 NaN(데이터 없는 구 전체)은 전체 평균으로
    df[COL_RATE] = df[COL_RATE].fillna(df[COL_RATE].mean())

    # ── 3. 자치구별 정렬 (시계열 파생 컬럼 계산 전 필수) ─
    df = df.sort_values([COL_DISTRICT, COL_DATE]).reset_index(drop=True)

    # ── 4. 파생 컬럼: 전세가율 변화율 (전월 대비) ────────
    df[COL_RATE_CHG] = df.groupby(COL_DISTRICT)[COL_RATE].diff()

    # ── 5. 파생 컬럼: 3개월 이동평균 ─────────────────────
    df[COL_RATE_MA] = (
        df.groupby(COL_DISTRICT)[COL_RATE]
        .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    )

    # ── 6. 타깃 변수: t+1 보증사고건수 기준 위험등급 ──────
    # t+1 보증사고건수 = 다음달 실제 사고 수
    df["_next_acc"] = df.groupby(COL_DISTRICT)[COL_ACC_CNT].shift(-1)

    # 전체 분포를 3분위로 나눠 상/중/하 분류
    q33 = df["_next_acc"].quantile(0.33)
    q66 = df["_next_acc"].quantile(0.66)

    def assign_grade(x):
        if pd.isna(x):
            return None       # 마지막 달은 다음달 데이터 없음
        if x >= q66:
            return "상"
        elif x >= q33:
            return "중"
        return "하"

    df[COL_GRADE] = df["_next_acc"].apply(assign_grade)
    df.drop(columns=["_next_acc"], inplace=True)

    # ── 7. 위험등급 순서형 범주 지정 ──────────────────────
    grade_order = pd.CategoricalDtype(["하", "중", "상"], ordered=True)
    df[COL_GRADE] = df[COL_GRADE].astype(grade_order)

    return df


# ═══════════════════════════════════════════════════════
# ▶ 전세가율 → 위험 레이블
# ═══════════════════════════════════════════════════════
def rate_to_label(rate: float) -> str:
    if rate >= 80:
        return "위험"
    elif rate >= 70:
        return "주의"
    return "안전"


# ═══════════════════════════════════════════════════════
# ▶ 사이드바 공통 렌더링
# ═══════════════════════════════════════════════════════
def render_sidebar_info(df: pd.DataFrame):
    with st.sidebar:
        st.markdown("## 📌 프로젝트 정보")
        st.info(
            "**서울 전세 위험 조기경보 시스템**\n\n"
            "연립·다세대 주택 전세가율과\n"
            "HUG 보증사고 데이터를 결합해\n"
            "자치구별 위험도를 분석·예측합니다."
        )
        st.markdown("---")
        st.markdown("**📂 데이터 출처**")
        st.markdown("- 전세가율: 한국부동산원")
        st.markdown("- 보증사고: 주택도시보증공사(HUG)")
        st.markdown("---")
        start = df[COL_DATE].min().strftime("%Y.%m")
        end   = df[COL_DATE].max().strftime("%Y.%m")
        st.markdown(f"**🗓️ 분석 기간:** {start} ~ {end}")
        st.markdown(f"**🏙️ 분석 대상:** 서울시 25개 자치구")