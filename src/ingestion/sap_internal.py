from __future__ import annotations

import sqlite3
import warnings
from pathlib import Path
from typing import Any

import pandas as pd

# 프로젝트 루트 기준 수동 적재 폴더
DATA_RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"

DEFAULT_STEMS = {
    "inventory": "sap_inventory",
    "po": "sap_po_history",
    "production": "sap_production_plan",
}

CREATE_INVENTORY_SQL = """
CREATE TABLE IF NOT EXISTS raw_sap_inventory (
    date TEXT PRIMARY KEY,
    inventory_ton REAL,
    safety_stock_ton REAL,
    tank_capacity_ton REAL,
    warehouse_code TEXT
);
"""

CREATE_PO_SQL = """
CREATE TABLE IF NOT EXISTS raw_sap_po_history (
    po_number TEXT PRIMARY KEY,
    po_date TEXT,
    delivery_date TEXT,
    quantity_ton REAL,
    unit_price REAL,
    contract_type TEXT,
    supplier_code TEXT
);
"""

CREATE_PRODUCTION_SQL = """
CREATE TABLE IF NOT EXISTS raw_sap_production (
    plan_date TEXT,
    production_quantity_ton REAL,
    product_code TEXT,
    plant_code TEXT,
    UNIQUE (plan_date, product_code, plant_code)
);
"""

INSERT_INVENTORY_SQL = """
INSERT INTO raw_sap_inventory
    (date, inventory_ton, safety_stock_ton, tank_capacity_ton, warehouse_code)
VALUES (?, ?, ?, ?, ?);
"""

INSERT_PO_SQL = """
INSERT INTO raw_sap_po_history
    (po_number, po_date, delivery_date, quantity_ton, unit_price, contract_type, supplier_code)
VALUES (?, ?, ?, ?, ?, ?, ?);
"""

INSERT_PRODUCTION_SQL = """
INSERT INTO raw_sap_production
    (plan_date, production_quantity_ton, product_code, plant_code)
VALUES (?, ?, ?, ?);
"""


def _resolve_input_path(filepath: str | Path | None, stem: str) -> Path | None:
    """
    filepath가 주어지면 그 경로만 사용.
    None이면 data/raw/{기본파일명}.xlsx → .xls → .csv 순으로 존재하는 첫 파일.
    """
    if filepath is not None:
        p = Path(filepath)
        return p if p.exists() else None
    name = DEFAULT_STEMS[stem]
    for ext in (".xlsx", ".xls", ".csv"):
        p = DATA_RAW_DIR / f"{name}{ext}"
        if p.exists():
            return p
    return None


def _read_tabular(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path)
    if suf == ".csv":
        return pd.read_csv(path, encoding="utf-8-sig")
    raise ValueError(f"지원하지 않는 확장자: {path.suffix}")


def _apply_column_map(df: pd.DataFrame, korean_to_eng: dict[str, str]) -> pd.DataFrame:
    """엑셀 헤더(한글·공백) → 내부 영문 컬럼명."""
    strip_map = {str(k).strip(): v for k, v in korean_to_eng.items()}
    rename: dict[str, str] = {}
    allowed_eng = set(korean_to_eng.values())
    for c in df.columns:
        key = str(c).strip()
        if key in strip_map:
            rename[c] = strip_map[key]
        elif key in allowed_eng:
            rename[c] = key
        else:
            rename[c] = key
    return df.rename(columns=rename)


def _normalize_date_columns(df: pd.DataFrame, date_cols: list[str]) -> pd.DataFrame:
    """지정 컬럼명이 존재하면 YYYY-MM-DD 문자열로 통일 (자동 감지: 이름 + 값 파싱)."""
    out = df.copy()
    for col in date_cols:
        if col not in out.columns:
            continue
        s = out[col]
        parsed = pd.to_datetime(s, errors="coerce")
        out[col] = parsed.dt.strftime("%Y-%m-%d")
        out.loc[parsed.isna(), col] = ""
    return out


def _auto_detect_extra_date_columns(df: pd.DataFrame, already: set[str]) -> list[str]:
    """이름에 날짜·일자 힌트가 있거나 datetime으로 대부분 파싱되는 컬럼."""
    extra: list[str] = []
    hints = ("date", "일", "날짜", "일자", "dt")
    for c in df.columns:
        if c in already:
            continue
        cl = str(c).lower()
        if any(h in str(c) for h in ("날짜", "일자")) or any(h in cl for h in hints):
            extra.append(str(c))
            continue
        try:
            parsed = pd.to_datetime(df[c].head(20), errors="coerce")
            if parsed.notna().sum() >= max(3, len(parsed) // 2):
                extra.append(str(c))
        except (TypeError, ValueError):
            continue
    return extra


def _coerce_float(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def _normalize_contract_type(v: Any) -> str:
    s = str(v).strip().lower() if v is not None and not (isinstance(v, float) and pd.isna(v)) else ""
    if s in ("spot", "현물"):
        return "spot"
    if s in ("forward", "fwd", "futures", "선물"):
        return "forward"
    if not s or s == "nan":
        return "other"
    return "other"


def load_inventory(filepath: str | Path | None, conn: sqlite3.Connection) -> int:
    """
    재고 엑셀 → `raw_sap_inventory`.

    한글 컬럼명 매핑 (필요 시 엑셀 헤더와 맞춰 수정):
    """
    COL_MAP_KO_TO_EN: dict[str, str] = {
        "날짜": "date",
        "일자": "date",
        "재고량(톤)": "inventory_ton",
        "재고(톤)": "inventory_ton",
        "재고량": "inventory_ton",
        "안전재고(톤)": "safety_stock_ton",
        "안전재고": "safety_stock_ton",
        "탱크용량(톤)": "tank_capacity_ton",
        "탱크용량": "tank_capacity_ton",
        "창고코드": "warehouse_code",
        "창고": "warehouse_code",
    }

    conn.execute(CREATE_INVENTORY_SQL)
    conn.commit()

    path = _resolve_input_path(filepath, "inventory")
    if path is None:
        warnings.warn(
            f"재고 파일 없음 — `raw_sap_inventory` 테이블만 유지합니다. "
            f"기본 경로: {DATA_RAW_DIR / (DEFAULT_STEMS['inventory'] + '.xlsx')} 등",
            UserWarning,
            stacklevel=2,
        )
        return 0

    df = _read_tabular(path)
    df = _apply_column_map(df, COL_MAP_KO_TO_EN)
    need = ["date", "inventory_ton", "safety_stock_ton", "tank_capacity_ton", "warehouse_code"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"필수 컬럼 누락: {c} (파일: {path})")

    # 날짜: 매핑된 date + (헤더/값 기준) 추가 날짜형 컬럼 통일
    extra = [c for c in _auto_detect_extra_date_columns(df, {"date"}) if c in df.columns]
    df = _normalize_date_columns(df, list(dict.fromkeys(["date"] + extra)))
    df = _coerce_float(df, ["inventory_ton", "safety_stock_ton", "tank_capacity_ton"])
    df["warehouse_code"] = df["warehouse_code"].astype(str).str.strip()
    df = df[df["date"].astype(str).str.len() >= 10]

    existing = {
        r[0]
        for r in conn.execute("SELECT date FROM raw_sap_inventory").fetchall()
    }
    df_new = df[~df["date"].isin(existing)]
    rows = []
    for _, r in df_new.iterrows():
        rows.append(
            (
                str(r["date"]),
                float(r["inventory_ton"]) if pd.notna(r["inventory_ton"]) else None,
                float(r["safety_stock_ton"]) if pd.notna(r["safety_stock_ton"]) else None,
                float(r["tank_capacity_ton"]) if pd.notna(r["tank_capacity_ton"]) else None,
                str(r["warehouse_code"]) if str(r["warehouse_code"]) != "nan" else "",
            )
        )
    if rows:
        conn.executemany(INSERT_INVENTORY_SQL, rows)
        conn.commit()
    return len(rows)


def load_po_history(filepath: str | Path | None, conn: sqlite3.Connection) -> int:
    """
    발주 이력 → `raw_sap_po_history`.

    contract_type: spot / forward 만 허용, 그 외는 other.
    """
    COL_MAP_KO_TO_EN: dict[str, str] = {
        "PO번호": "po_number",
        "발주번호": "po_number",
        "발주일": "po_date",
        "PO일자": "po_date",
        "납품일": "delivery_date",
        "입고예정일": "delivery_date",
        "수량(톤)": "quantity_ton",
        "수량": "quantity_ton",
        "단가": "unit_price",
        "계약유형": "contract_type",
        "계약구분": "contract_type",
        "공급업체코드": "supplier_code",
        "공급업체": "supplier_code",
    }

    conn.execute(CREATE_PO_SQL)
    conn.commit()

    path = _resolve_input_path(filepath, "po")
    if path is None:
        warnings.warn(
            f"발주 이력 파일 없음 — `raw_sap_po_history` 테이블만 유지합니다. "
            f"기본 경로: {DATA_RAW_DIR / (DEFAULT_STEMS['po'] + '.xlsx')} 등",
            UserWarning,
            stacklevel=2,
        )
        return 0

    df = _read_tabular(path)
    df = _apply_column_map(df, COL_MAP_KO_TO_EN)
    need = [
        "po_number",
        "po_date",
        "delivery_date",
        "quantity_ton",
        "unit_price",
        "contract_type",
        "supplier_code",
    ]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"필수 컬럼 누락: {c} (파일: {path})")

    df = _normalize_date_columns(df, ["po_date", "delivery_date"])
    df = _coerce_float(df, ["quantity_ton", "unit_price"])
    df["contract_type"] = df["contract_type"].map(_normalize_contract_type)
    df["po_number"] = df["po_number"].astype(str).str.strip()
    df["supplier_code"] = df["supplier_code"].astype(str).str.strip()
    df = df[df["po_number"].astype(str).str.len() > 0]

    existing = {r[0] for r in conn.execute("SELECT po_number FROM raw_sap_po_history").fetchall()}
    df_new = df[~df["po_number"].isin(existing)]
    rows = []
    for _, r in df_new.iterrows():
        rows.append(
            (
                str(r["po_number"]),
                str(r["po_date"]) if pd.notna(r["po_date"]) and str(r["po_date"]) else "",
                str(r["delivery_date"]) if pd.notna(r["delivery_date"]) and str(r["delivery_date"]) else "",
                float(r["quantity_ton"]) if pd.notna(r["quantity_ton"]) else None,
                float(r["unit_price"]) if pd.notna(r["unit_price"]) else None,
                str(r["contract_type"]),
                str(r["supplier_code"]) if str(r["supplier_code"]) != "nan" else "",
            )
        )
    if rows:
        conn.executemany(INSERT_PO_SQL, rows)
        conn.commit()
    return len(rows)


def load_production_plan(filepath: str | Path | None, conn: sqlite3.Connection) -> int:
    """
    생산 계획 → `raw_sap_production`.

    중복 키: (plan_date, product_code, plant_code).
    """
    COL_MAP_KO_TO_EN: dict[str, str] = {
        "계획일": "plan_date",
        "생산일": "plan_date",
        "계획일자": "plan_date",
        "생산수량(톤)": "production_quantity_ton",
        "생산수량": "production_quantity_ton",
        "제품코드": "product_code",
        "자재코드": "product_code",
        "공장코드": "plant_code",
        "플랜트": "plant_code",
    }

    conn.execute(CREATE_PRODUCTION_SQL)
    conn.commit()

    path = _resolve_input_path(filepath, "production")
    if path is None:
        warnings.warn(
            f"생산계획 파일 없음 — `raw_sap_production` 테이블만 유지합니다. "
            f"기본 경로: {DATA_RAW_DIR / (DEFAULT_STEMS['production'] + '.xlsx')} 등",
            UserWarning,
            stacklevel=2,
        )
        return 0

    df = _read_tabular(path)
    df = _apply_column_map(df, COL_MAP_KO_TO_EN)
    need = ["plan_date", "production_quantity_ton", "product_code", "plant_code"]
    for c in need:
        if c not in df.columns:
            raise ValueError(f"필수 컬럼 누락: {c} (파일: {path})")

    df = _normalize_date_columns(df, ["plan_date"])
    df = _coerce_float(df, ["production_quantity_ton"])
    df["product_code"] = df["product_code"].astype(str).str.strip()
    df["plant_code"] = df["plant_code"].astype(str).str.strip()
    df = df[df["plan_date"].astype(str).str.len() >= 10]

    existing = {
        (r[0], r[1], r[2])
        for r in conn.execute(
            "SELECT plan_date, product_code, plant_code FROM raw_sap_production"
        ).fetchall()
    }
    df = df.drop_duplicates(subset=["plan_date", "product_code", "plant_code"], keep="last")

    def _prod_key(r: pd.Series) -> tuple[str, str, str]:
        return (str(r["plan_date"]), str(r["product_code"]), str(r["plant_code"]))

    keys = df.apply(_prod_key, axis=1)
    df_new = df[~keys.isin(existing)]
    rows = []
    for _, r in df_new.iterrows():
        rows.append(
            (
                str(r["plan_date"]),
                float(r["production_quantity_ton"]) if pd.notna(r["production_quantity_ton"]) else None,
                str(r["product_code"]),
                str(r["plant_code"]),
            )
        )
    if rows:
        conn.executemany(INSERT_PRODUCTION_SQL, rows)
        conn.commit()
    return len(rows)


# --- 초기 스켈레톤 호환 ---


def load_sap_export(path: str) -> None:
    """사용하지 않음. `load_inventory` / `load_po_history` / `load_production_plan` 사용."""
    raise NotImplementedError("SAP 수동 적재는 load_inventory 등 개별 함수를 사용하세요.")


def map_sap_columns(df: pd.DataFrame) -> pd.DataFrame:
    """내부용이 아닌 경우 `_apply_column_map`을 직접 호출하세요."""
    return df


def sync_raw_internal_to_db(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    """미구현 — 테이블별 load_* 함수를 사용하세요."""
    raise NotImplementedError("sync_raw_internal_to_db 대신 load_inventory 등을 사용하세요.")
