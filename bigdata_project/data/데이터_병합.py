"""
데이터_병합.py
──────────────────────────────────────────────────────────
임대시장데이터 폴더의 월별 xlsx 파일을 하나로 합칩니다.

추출 컬럼 (연립·다세대 기준):
  - 년월          : 파일명에서 추출 (예: market_202405.xlsx → 202405)
  - 자치구         : 시군구 (서울특별시만 필터)
  - 전세가율       : 연립·다세대 최근 1년 전세가율(%)
  - 전세가율_3개월 : 연립·다세대 최근 3개월 전세가율(%)
  - 보증사고건수   : 보증사고 사고건수(건)
  - 보증사고금액   : 보증사고 사고금액(원)
  - 보증사고율     : 보증사고 사고율(%)

출력: data/seoul_market_merged.xlsx
──────────────────────────────────────────────────────────
실행: python data/데이터_병합.py
"""

import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────
# 경로 설정
# ─────────────────────────────────────────
BASE_DIR    = Path(__file__).parent
INPUT_DIR   = BASE_DIR / "임대시장데이터"
OUTPUT_PATH = BASE_DIR / "seoul_market_merged.xlsx"

# ─────────────────────────────────────────
# 추출할 컬럼 인덱스 (0-based, 원본 파일 구조 기준)
# ─────────────────────────────────────────
COL_SIDO      = 0   # 시도
COL_DISTRICT  = 3   # 시군구 (자치구)
COL_RATE_1Y   = 12  # 연립·다세대 전세가율 최근 1년(%)
COL_RATE_3M   = 15  # 연립·다세대 전세가율 최근 3개월(%)
COL_ACC_CNT   = 18  # 보증사고 건수(건)
COL_ACC_AMT   = 20  # 보증사고 금액(원)
COL_ACC_RATE  = 22  # 보증사고율(%)


def parse_monthly_file(file_path: Path) -> pd.DataFrame:
    """
    월별 xlsx 한 파일에서 서울특별시 자치구 데이터를 추출합니다.
    헤더가 복잡(4행)하므로 header=None으로 읽고 직접 파싱합니다.
    """
    year_month = int(file_path.stem.split("_")[1])  # market_202405 → 202405

    df_raw = pd.read_excel(file_path, header=None)

    # 서울특별시이면서 소계가 아닌 행 (= 25개 자치구)
    seoul_df = df_raw[
        (df_raw[COL_SIDO] == "서울특별시") &
        (df_raw[COL_DISTRICT] != "소계")
    ].copy()

    if seoul_df.empty:
        print(f"  ⚠️  {file_path.name}: 데이터 없음, 건너뜀")
        return pd.DataFrame()

    result = pd.DataFrame({
        "년월"          : year_month,
        "자치구"        : seoul_df[COL_DISTRICT].values,
        "전세가율"      : pd.to_numeric(seoul_df[COL_RATE_1Y],  errors="coerce"),
        "전세가율_3개월" : pd.to_numeric(seoul_df[COL_RATE_3M],  errors="coerce"),
        "보증사고건수"   : pd.to_numeric(seoul_df[COL_ACC_CNT],  errors="coerce").fillna(0).astype(int),
        "보증사고금액"   : pd.to_numeric(seoul_df[COL_ACC_AMT],  errors="coerce").fillna(0).astype(int),
        "보증사고율"     : pd.to_numeric(seoul_df[COL_ACC_RATE], errors="coerce"),
    })

    return result


def merge_all_files():
    files = sorted(INPUT_DIR.glob("market_*.xlsx"))

    if not files:
        print(f"❌ '{INPUT_DIR}' 폴더에 market_*.xlsx 파일이 없습니다.")
        return

    print(f"📂 총 {len(files)}개 파일 처리 시작\n")

    all_dfs = []
    for f in files:
        print(f"  처리 중: {f.name}")
        monthly = parse_monthly_file(f)
        if not monthly.empty:
            all_dfs.append(monthly)

    if not all_dfs:
        print("❌ 처리된 데이터가 없습니다.")
        return

    merged = pd.concat(all_dfs, ignore_index=True)
    merged = merged.sort_values(["년월", "자치구"]).reset_index(drop=True)

    merged.to_excel(OUTPUT_PATH, index=False)

    print(f"\n✅ 병합 완료!")
    print(f"   저장 경로: {OUTPUT_PATH}")
    print(f"   총 행 수 : {len(merged):,}행")
    print(f"   기간     : {merged['년월'].min()} ~ {merged['년월'].max()}")
    print(f"   자치구 수: {merged['자치구'].nunique()}개")
    print(f"\n📋 결측치 현황:")
    print(merged.isnull().sum())
    print(f"\n📋 상위 5행 미리보기:")
    print(merged.head())


if __name__ == "__main__":
    merge_all_files()