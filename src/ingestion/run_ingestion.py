"""
전체 raw 데이터 수집을 한 번에 실행하는 엔트리포인트.

실행 (프로젝트 루트 `soybean-oil-poc` 에서):

    python src/ingestion/run_ingestion.py

뉴스(GDELT): 환경변수 `NEWS_INGESTION_API_KEY` 가 비어 있으면 뉴스 단계는 스킵한다.
(공개 API만 사용하더라도, 파이프라인에서 의도적으로 켤 때 아무 값이나 설정 — 예: `export NEWS_INGESTION_API_KEY=1`)
"""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

# 프로젝트 루트를 path에 넣어 `python src/ingestion/run_ingestion.py` 지원
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

try:
    from dotenv import load_dotenv

    load_dotenv(_ROOT / ".env")
except ImportError:
    pass

DB_PATH = _ROOT / "data" / "db" / "soybean.db"
ENV_NEWS_KEY = "NEWS_INGESTION_API_KEY"

# 요약 출력 순서 (존재하지 않는 테이블은 COUNT 시 건너뜀)
SUMMARY_TABLES: list[str] = [
    "raw_price_futures",
    "raw_crude_oil",
    "raw_soybean_futures",
    "raw_soymeal_futures",
    "raw_palm_oil",
    "raw_exchange_rate",
    "raw_price_spot",
    "raw_dollar_index",
    "raw_canola_oil",
    "raw_sunflower_oil",
    "raw_eia_biodiesel",
    "raw_fed_rate",
    "raw_vix",
    "raw_cftc",
    "raw_wasde",
    "raw_news_scored",
    "raw_sap_inventory",
    "raw_sap_po_history",
    "raw_sap_production",
]


def _zero_row_note(table: str, news_skipped: bool) -> str:
    if table == "raw_wasde":
        return " (수동 적재 필요)"
    if table == "raw_news_scored":
        return " (API 키 필요)" if news_skipped else " (GDELT 응답 없음·레이트리밋 등)"
    if table.startswith("raw_sap_"):
        return " (엑셀 파일 필요)"
    if table == "raw_price_spot":
        return " (World Bank 엑셀·CEPEA 등)"
    if table == "raw_dollar_index":
        return " (yfinance DX-Y.NYB)"
    if table == "raw_canola_oil":
        return " (RS=F 실패)"
    if table == "raw_sunflower_oil":
        return " (World Bank 엑셀)"
    if table == "raw_eia_biodiesel":
        return " (EIA DNAV XLS 응답/파싱 확인 필요)"
    if table == "raw_fed_rate":
        return " (FRED CSV 응답/파싱 확인 필요)"
    if table == "raw_vix":
        return " (FRED CSV 응답/파싱 확인 필요)"
    if table == "raw_palm_oil":
        return " (데이터 없음)"
    return ""


def print_summary(db_path: Path, news_skipped: bool) -> None:
    print()
    print("=== 데이터 수집 완료 ===")
    with sqlite3.connect(db_path) as conn:
        for t in SUMMARY_TABLES:
            try:
                cur = conn.execute(f"SELECT COUNT(*) FROM {t}")
                n = cur.fetchone()[0]
            except sqlite3.OperationalError:
                print(f"{t:<22}: — (테이블 없음)")
                continue
            extra = _zero_row_note(t, news_skipped) if n == 0 else ""
            print(f"{t:<22}: {n}행{extra}")


def main() -> None:
    db_path = DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)

    from src.ingestion.price import (
        CANOLA_OIL_CSV_DEFAULT_PATH,
        RAW_CANOLA_OIL_TABLE,
        ensure_raw_canola_oil_table,
        ingest_all_futures_to_sqlite,
        load_canola_oil,
        load_palm_oil_from_investing_csv,
        PALM_OIL_CSV_DEFAULT_PATH,
    )
    from src.ingestion.macro import (
        WORLDBANK_COMMODITY_MONTHLY_DEFAULT_PATH,
        load_dollar_index,
        load_eia_biodiesel,
        load_exchange_rate,
        load_fed_rate,
        load_price_spot,
        load_sunflower_oil,
        load_vix,
        load_worldbank_pink_sheet_soybean_oil,
    )
    from src.ingestion.cftc import load_cftc
    from src.ingestion.wasde import RAW_WASDE_TABLE, load_wasde_from_csv, load_wasde_from_usda_api
    from src.ingestion.news_scorer import load_news_to_db, run_news_pipeline
    from src.ingestion.sap_internal import (
        load_inventory,
        load_po_history,
        load_production_plan,
    )

    print(
        "[0] raw_canola_oil 기존 행 삭제 (과거 오적재·옥수수 ZC=F 데이터 제거) …"
    )
    with sqlite3.connect(db_path) as conn:
        ensure_raw_canola_oil_table(conn)
        conn.execute(f"DELETE FROM {RAW_CANOLA_OIL_TABLE}")
        conn.commit()
    print(f"    DELETE FROM {RAW_CANOLA_OIL_TABLE} 완료 → 파이프라인 중 CSV로 재적재·행 수 출력")

    print("[1/6] 선물 가격 (ZL, CL, ZS, ZM, 팜유 시도) …")
    ingest_all_futures_to_sqlite(db_path, print_counts=False)

    palm_csv = PALM_OIL_CSV_DEFAULT_PATH
    if palm_csv.is_file():
        print("    팜유 CSV (Investing CPOc1) …")
        with sqlite3.connect(db_path) as conn:
            load_palm_oil_from_investing_csv(conn, palm_csv)
    else:
        print(f"    팜유 CSV 없음 — 건너뜀: {palm_csv}")

    print("[2/6] 환율(BRL=X)·현물 테이블 …")
    load_exchange_rate(db_path, print_count=False)
    load_price_spot(db_path, print_message=False)
    if WORLDBANK_COMMODITY_MONTHLY_DEFAULT_PATH.is_file():
        print("    World Bank Pink Sheet (Soybean oil) …")
        load_worldbank_pink_sheet_soybean_oil(
            db_path, WORLDBANK_COMMODITY_MONTHLY_DEFAULT_PATH, print_summary=True
        )
    else:
        print(f"    World Bank 엑셀 없음 — 건너뜀: {WORLDBANK_COMMODITY_MONTHLY_DEFAULT_PATH}")

    with sqlite3.connect(db_path) as conn:
        print("    달러 지수 (DX-Y.NYB) …")
        load_dollar_index(conn)
        print("    카놀라 선물 CSV (Investing RSc1, 테이블 비우고 재적재) …")
        load_canola_oil(conn, CANOLA_OIL_CSV_DEFAULT_PATH)
        if WORLDBANK_COMMODITY_MONTHLY_DEFAULT_PATH.is_file():
            print("    World Bank Pink Sheet (Sunflower oil) …")
            load_sunflower_oil(conn)
        else:
            print(
                "    World Bank 엑셀 없음 — Sunflower oil 건너뜀: "
                f"{WORLDBANK_COMMODITY_MONTHLY_DEFAULT_PATH}"
            )
        print("    EIA DNAV XLS (Total Soybean Oil, 월간) …")
        load_eia_biodiesel(conn)
        print("    FRED FEDFUNDS (월별) …")
        load_fed_rate(conn)
        print("    FRED VIXCLS (일별) …")
        load_vix(conn)

    news_skipped = not (os.environ.get(ENV_NEWS_KEY, "").strip())

    with sqlite3.connect(db_path) as conn:
        print("[3/6] CFTC …")
        load_cftc(conn)

        print("[4/6] WASDE/PSD …")
        wasde_csv = Path(__file__).resolve().parents[2] / "data" / "raw" / "wasde_psd.csv"
        if wasde_csv.is_file():
            conn.execute(f"DELETE FROM {RAW_WASDE_TABLE}")
            conn.commit()
            print(f"    WASDE CSV 재적재: {wasde_csv}")
            load_wasde_from_csv(wasde_csv, conn)
        else:
            print("    WASDE CSV 없음 — USDA API 경로 사용")
            load_wasde_from_usda_api(conn)

        print("[5/6] 뉴스(GDELT, 최근 7일) …")
        if news_skipped:
            load_news_to_db([], conn)
            print(f"    스킵: {ENV_NEWS_KEY} 미설정")
        else:
            run_news_pipeline(conn, days_back=7)

        print("[6/6] SAP 엑셀 (data/raw/ 있으면 적재) …")
        load_inventory(None, conn)
        load_po_history(None, conn)
        load_production_plan(None, conn)

    print_summary(db_path, news_skipped=news_skipped)


if __name__ == "__main__":
    main()
