"""
데이터_전처리.py
──────────────────────────────────────────────────────────
병합된 데이터(seoul_market_merged.xlsx)를 가져와서
모델 학습에 필요한 형태로 가공합니다.

[전처리 단계]
  1. 날짜 정렬 및 결측치 처리
  2. 파생 피처 계산
       - 전세가율_변화율  : 자치구별 전월 대비 전세가율 증감(%p)
       - 이동평균_전세가율: 자치구별 최근 3개월 이동평균
       - 전세가율_가속도  : 변화율의 변화 (급등 감지)
  3. 타깃 변수 생성 (t+1 예측용)
       - 다음달_보증사고건수: t+1 시점의 실제 보증사고건수
       - 위험등급          : 다음달 보증사고건수를 상/중/하로 분류

입력: data/seoul_market_merged.xlsx
출력: data/seoul_market_preprocessed.xlsx
──────────────────────────────────────────────────────────
실행: python data/데이터_전처리.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

# ─────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
INPUT_PATH  = BASE_DIR / "seoul_market_merged.xlsx"
OUTPUT_PATH = BASE_DIR / "seoul_market_preprocessed.xlsx"


def preprocess():
    # ════════════════════════════════════════
    # STEP 0. 데이터 로드
    # ════════════════════════════════════════
    print("=" * 50)
    print("STEP 0. 데이터 로드")
    print("=" * 50)

    if not INPUT_PATH.exists():
        print(f"❌ '{INPUT_PATH}' 파일이 없습니다.")
        print("   먼저 데이터_병합.py를 실행해 주세요.")
        return

    df = pd.read_excel(INPUT_PATH)
    print(f"✅ 로드 완료: {len(df):,}행 × {len(df.columns)}열")
    print(f"   컬럼: {df.columns.tolist()}")
    print(f"   기간: {df['년월'].min()} ~ {df['년월'].max()}")
    print()

    # ════════════════════════════════════════
    # STEP 1. 날짜 변환 및 정렬
    # ════════════════════════════════════════
    print("=" * 50)
    print("STEP 1. 날짜 변환 및 정렬")
    print("=" * 50)

    # 년월: 202405 → datetime (시계열 파생 컬럼 계산을 위해 필수)
    df["년월"] = pd.to_datetime(df["년월"].astype(str), format="%Y%m")

    # 자치구별 시간 순서 정렬 (shift/diff/rolling 계산 전 반드시 필요)
    df = df.sort_values(["자치구", "년월"]).reset_index(drop=True)
    print(f"✅ 정렬 완료 (자치구 → 년월 순)")
    print()

    # ════════════════════════════════════════
    # STEP 2. 결측치 처리
    # ════════════════════════════════════════
    print("=" * 50)
    print("STEP 2. 결측치 처리")
    print("=" * 50)
    print("처리 전 결측치:")
    print(df.isnull().sum())
    print()

    # 전세가율: 해당 자치구의 앞뒤 값으로 보간 (선형)
    # → "거래 없어서 산정 불가" 상황이므로 인접 월 값으로 추정
    df["전세가율"] = df.groupby("자치구")["전세가율"].transform(
        lambda x: x.interpolate(method="linear", limit_direction="both")
    )
    df["전세가율_3개월"] = df.groupby("자치구")["전세가율_3개월"].transform(
        lambda x: x.interpolate(method="linear", limit_direction="both")
    )

    # 보증사고율 결측 → 0으로 (사고 없는 달)
    df["보증사고율"] = df["보증사고율"].fillna(0)

    print("처리 후 결측치:")
    print(df.isnull().sum())
    print()

    # ════════════════════════════════════════
    # STEP 3. 파생 피처 계산
    # ════════════════════════════════════════
    print("=" * 50)
    print("STEP 3. 파생 피처 계산")
    print("=" * 50)

    # ── 3-1. 전세가율 변화율 (전월 대비 증감, %p) ──────────
    # 예) 이번달 72.5% - 지난달 70.0% = +2.5%p
    # → 전세가율이 급등하는 지역을 포착하는 핵심 신호
    df["전세가율_변화율"] = df.groupby("자치구")["전세가율"].diff()
    print("✅ 전세가율_변화율 계산 완료 (전월 대비 증감)")

    # ── 3-2. 전세가율 3개월 이동평균 ──────────────────────
    # 단기 노이즈를 제거하고 추세를 부드럽게 파악
    # min_periods=1: 첫달처럼 3개월이 안 쌓인 경우도 계산
    df["이동평균_전세가율"] = df.groupby("자치구")["전세가율"].transform(
        lambda x: x.rolling(window=3, min_periods=1).mean()
    )
    print("✅ 이동평균_전세가율 계산 완료 (3개월 이동평균)")

    # ── 3-3. 전세가율 가속도 (변화율의 변화) ──────────────
    # 예) 지난달 +1%p 상승했는데 이번달 +4%p → 가속도 +3
    # → 단순 상승보다 "점점 빨리 오르는" 위험 신호 감지
    df["전세가율_가속도"] = df.groupby("자치구")["전세가율_변화율"].diff()
    print("✅ 전세가율_가속도 계산 완료 (변화율의 변화)")

    # ── 3-4. 위험구분 (현재 시점 기준 참고용) ─────────────
    # 80% 이상이면 깡통전세 위험
    df["위험구분"] = df["전세가율"].apply(
        lambda x: "위험" if x >= 80 else ("주의" if x >= 70 else "안전")
    )
    print("✅ 위험구분 계산 완료 (안전/주의/위험)")
    print()

    # ════════════════════════════════════════
    # STEP 4. 타깃 변수 생성 (모델 정답 레이블)
    # ════════════════════════════════════════
    print("=" * 50)
    print("STEP 4. 타깃 변수 생성 (t+1 예측 타깃)")
    print("=" * 50)

    # ── 4-1. 다음달 보증사고건수 (t+1) ────────────────────
    # shift(-1): 같은 자치구에서 한 달 앞의 값을 현재 행으로 당겨옴
    # 예) 2024년 5월 강서구 행에 → 2024년 6월 강서구 보증사고건수를 붙임
    # 마지막 달은 다음달 데이터가 없으므로 NaN이 됨 (정상)
    df["다음달_보증사고건수"] = df.groupby("자치구")["보증사고건수"].shift(-1)
    print("✅ 다음달_보증사고건수 생성 (t+1 보증사고건수)")

    # ── 4-2. 위험등급 (상/중/하) ──────────────────────────
    # 다음달 보증사고건수를 전체 분포 기준으로 3등분
    # 상위 33% → 상 (고위험), 중간 33% → 중 (주의), 하위 33% → 하 (안전)
    #
    # 왜 절대값이 아닌 분위수로 나누는가?
    # → 지역마다, 시기마다 사고 건수 규모가 달라
    #   "강서구 100건"과 "종로구 5건"을 같은 기준으로 보면 왜곡됨
    # → 전체 분포 내 상대적 위치로 판단하는 것이 더 공정
    valid_acc = df["다음달_보증사고건수"].dropna()
    q33 = valid_acc.quantile(0.33)
    q66 = valid_acc.quantile(0.66)

    print(f"   위험등급 기준: 하(0~{q33:.0f}건) / 중({q33:.0f}~{q66:.0f}건) / 상({q66:.0f}건~)")

    def assign_grade(x):
        if pd.isna(x):
            return np.nan      # 마지막 달: 다음달 데이터 없음 → NaN
        if x >= q66:
            return "상"        # 고위험
        elif x >= q33:
            return "중"        # 주의
        return "하"            # 안전

    df["위험등급"] = df["다음달_보증사고건수"].apply(assign_grade)

    grade_counts = df["위험등급"].value_counts()
    print(f"✅ 위험등급 생성 완료")
    print(f"   상: {grade_counts.get('상', 0)}개, "
          f"중: {grade_counts.get('중', 0)}개, "
          f"하: {grade_counts.get('하', 0)}개, "
          f"NaN: {df['위험등급'].isna().sum()}개")
    print()

    # ════════════════════════════════════════
    # STEP 5. 최종 컬럼 정리 및 저장
    # ════════════════════════════════════════
    print("=" * 50)
    print("STEP 5. 최종 컬럼 정리 및 저장")
    print("=" * 50)

    # 년월을 다시 YYYYMM 숫자형으로 변환 (저장 후 읽기 편하게)
    df["년월"] = df["년월"].dt.strftime("%Y%m").astype(int)

    # 최종 컬럼 순서 정의
    final_cols = [
        # ── 기본 정보
        "년월", "자치구",
        # ── 원본 피처
        "전세가율", "전세가율_3개월",
        "보증사고건수", "보증사고금액", "보증사고율",
        # ── 파생 피처 (모델 입력)
        "전세가율_변화율", "이동평균_전세가율", "전세가율_가속도",
        # ── 참고용 레이블
        "위험구분",
        # ── 타깃 변수 (모델 정답)
        "다음달_보증사고건수", "위험등급",
    ]
    df = df[final_cols]

    df.to_excel(OUTPUT_PATH, index=False)

    print(f"✅ 전처리 완료! 저장 경로: {OUTPUT_PATH}")
    print(f"   총 행 수: {len(df):,}행 × {len(df.columns)}열")
    print(f"\n📋 최종 컬럼 목록:")
    for col in df.columns:
        role = ""
        if col in ["년월", "자치구"]:
            role = "← 기본 정보"
        elif col in ["전세가율", "전세가율_3개월", "보증사고건수", "보증사고금액", "보증사고율"]:
            role = "← 원본 피처"
        elif col in ["전세가율_변화율", "이동평균_전세가율", "전세가율_가속도"]:
            role = "← 파생 피처 (모델 입력)"
        elif col == "위험등급":
            role = "← 타깃 변수 (모델 정답 레이블)"
        elif col == "다음달_보증사고건수":
            role = "← 타깃 변수 (수치)"
        print(f"   {col:20s} {role}")

    print(f"\n📋 상위 5행 미리보기:")
    print(df.head().to_string())


if __name__ == "__main__":
    preprocess()