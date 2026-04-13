"""
역할: 매크로·현물 관련 raw 테이블 적재 (1단계 수집).

1) `load_exchange_rate`: Yahoo Finance BRL=X로 USD/BRL 일별 시세 → `raw_exchange_rate`
2) `load_price_spot`: `raw_price_spot` 테이블 보장(빈 경우 생성).
3) `load_worldbank_pink_sheet_soybean_oil`: World Bank Pink Sheet 엑셀
   `data/raw/worldbank_commodity_monthly.xlsx` 시트 `Monthly Prices`에서 대두유(USD) 월별 적재.
   CEPEA 등 다른 출처는 별도 적재 시 `source`로 구분(PK는 `date` 단일).
4) `load_dollar_index`: Yahoo ``DX-Y.NYB`` → ``raw_dollar_index``
5) `load_sunflower_oil`: 동일 엑셀에서 Sunflower oil 월별 → ``raw_sunflower_oil``
6) `load_eia_biodiesel`: EIA DNAV XLS 월간 대두유 바이오연료 투입량 → ``raw_eia_biodiesel``

Look-ahead bias (환율):
- Yahoo 관측일 데이터는 해당 일자 확정분으로 본다. 주말·휴장일은 직전 관측일로 forward-fill 하며
  `is_interpolated=1`로 표시한다 (신규 시장 정보 아님).
- master_daily에서는 관측일만 쓸지·보간 행을 제외할지 정책으로 통제한다.
- ffill은 과거→현재만 사용한다.
"""

from __future__ import annotations

import re
import sqlite3
import warnings
from io import BytesIO
from io import StringIO
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "db" / "soybean.db"
DEFAULT_START_DATE = "2010-01-01"
TICKER_USD_BRL = "BRL=X"
RAW_EXCHANGE_TABLE = "raw_exchange_rate"
RAW_PRICE_SPOT_TABLE = "raw_price_spot"
WORLDBANK_COMMODITY_MONTHLY_DEFAULT_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "raw" / "worldbank_commodity_monthly.xlsx"
)
WORLDBANK_PINK_SHEET_SOURCE = "WorldBank"
WORLDBANK_MONTHLY_SHEET_NAME = "Monthly Prices"
TICKER_DOLLAR_INDEX = "DX-Y.NYB"
RAW_DOLLAR_INDEX_TABLE = "raw_dollar_index"
RAW_SUNFLOWER_OIL_TABLE = "raw_sunflower_oil"
RAW_EIA_BIODIESEL_TABLE = "raw_eia_biodiesel"
RAW_FED_RATE_TABLE = "raw_fed_rate"
RAW_VIX_TABLE = "raw_vix"
EIA_BIODIESEL_SOURCE = "EIA_DNAV"
EIA_DNAV_XLS_URL = "https://www.eia.gov/dnav/pet/xls/PET_PNP_FEEDBIOFUEL_DCU_NUS_M.xls"
EIA_DNAV_SHEET_NAME = "Data 1"
EIA_DNAV_SERIES_ID = "M_EPOOBDSO_YIFBP_NUS_MMLB"
FRED_SOURCE = "FRED"
FRED_FEDFUNDS_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=FEDFUNDS"
FRED_VIX_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=VIXCLS"

CREATE_RAW_EIA_BIODIESEL_SQL = f"""
CREATE TABLE IF NOT EXISTS {RAW_EIA_BIODIESEL_TABLE} (
    date TEXT PRIMARY KEY,
    soybean_oil_biofuel_mmlb REAL,
    source TEXT DEFAULT '{EIA_BIODIESEL_SOURCE}'
);
"""

INSERT_EIA_BIODIESEL_IGNORE_SQL = f"""
INSERT OR IGNORE INTO {RAW_EIA_BIODIESEL_TABLE}
    (date, soybean_oil_biofuel_mmlb, source)
VALUES (?, ?, ?);
"""

CREATE_RAW_FED_RATE_SQL = f"""
CREATE TABLE IF NOT EXISTS {RAW_FED_RATE_TABLE} (
    date TEXT PRIMARY KEY,
    fed_rate REAL,
    source TEXT DEFAULT '{FRED_SOURCE}'
);
"""

INSERT_FED_RATE_IGNORE_SQL = f"""
INSERT OR IGNORE INTO {RAW_FED_RATE_TABLE}
    (date, fed_rate, source)
VALUES (?, ?, ?);
"""

CREATE_RAW_VIX_SQL = f"""
CREATE TABLE IF NOT EXISTS {RAW_VIX_TABLE} (
    date TEXT PRIMARY KEY,
    vix_close REAL,
    source TEXT DEFAULT '{FRED_SOURCE}'
);
"""

INSERT_VIX_IGNORE_SQL = f"""
INSERT OR IGNORE INTO {RAW_VIX_TABLE}
    (date, vix_close, source)
VALUES (?, ?, ?);
"""

CREATE_RAW_DOLLAR_INDEX_SQL = f"""
CREATE TABLE IF NOT EXISTS {RAW_DOLLAR_INDEX_TABLE} (
    date TEXT PRIMARY KEY,
    dxy_close REAL,
    dxy_open REAL,
    dxy_high REAL,
    dxy_low REAL
);
"""

INSERT_DOLLAR_INDEX_IGNORE_SQL = f"""
INSERT OR IGNORE INTO {RAW_DOLLAR_INDEX_TABLE}
    (date, dxy_close, dxy_open, dxy_high, dxy_low)
VALUES (?, ?, ?, ?, ?);
"""

CREATE_RAW_SUNFLOWER_OIL_SQL = f"""
CREATE TABLE IF NOT EXISTS {RAW_SUNFLOWER_OIL_TABLE} (
    date TEXT PRIMARY KEY,
    sunflower_close REAL,
    source TEXT DEFAULT '{WORLDBANK_PINK_SHEET_SOURCE}'
);
"""

INSERT_SUNFLOWER_OIL_IGNORE_SQL = f"""
INSERT OR IGNORE INTO {RAW_SUNFLOWER_OIL_TABLE}
    (date, sunflower_close, source)
VALUES (?, ?, ?);
"""

CREATE_RAW_EXCHANGE_SQL = f"""
CREATE TABLE IF NOT EXISTS {RAW_EXCHANGE_TABLE} (
    date TEXT PRIMARY KEY,
    usd_brl_close REAL,
    usd_brl_open REAL,
    usd_brl_high REAL,
    usd_brl_low REAL,
    is_interpolated INTEGER
);
"""

INSERT_EXCHANGE_IGNORE_SQL = f"""
INSERT OR IGNORE INTO {RAW_EXCHANGE_TABLE}
    (date, usd_brl_close, usd_brl_open, usd_brl_high, usd_brl_low, is_interpolated)
VALUES (?, ?, ?, ?, ?, ?);
"""

# 현물/월별 인용가: PK date. USD·BRL·출처 구분 (World Bank Pink Sheet, CEPEA 수동 등).
CREATE_RAW_PRICE_SPOT_SQL = f"""
CREATE TABLE IF NOT EXISTS {RAW_PRICE_SPOT_TABLE} (
    date TEXT PRIMARY KEY,
    spot_price_usd REAL,
    spot_price_brl REAL DEFAULT NULL,
    source TEXT DEFAULT '{WORLDBANK_PINK_SHEET_SOURCE}'
);
"""

INSERT_PRICE_SPOT_IGNORE_SQL = f"""
INSERT OR IGNORE INTO {RAW_PRICE_SPOT_TABLE}
    (date, spot_price_usd, spot_price_brl, source)
VALUES (?, ?, ?, ?);
"""


def _yfinance_end_exclusive_through(today: date | None = None) -> str:
    """일봉 다운로드 시 오늘까지 포함되도록 yfinance 배타적 end 날짜."""
    d = today or date.today()
    return (d + timedelta(days=1)).isoformat()


def ensure_raw_exchange_rate_table(conn: sqlite3.Connection) -> None:
    """`raw_exchange_rate` 테이블이 없으면 생성한다."""
    conn.execute(CREATE_RAW_EXCHANGE_SQL)
    conn.commit()


def ensure_raw_price_spot_table(conn: sqlite3.Connection) -> None:
    """`raw_price_spot` 테이블이 없으면 생성한다."""
    conn.execute(CREATE_RAW_PRICE_SPOT_SQL)
    conn.commit()


def _normalize_fx_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance FX 결과 컬럼을 open, high, low, close 소문자로 맞춘다 (volume 없음)."""
    if df.empty:
        return df
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(c[0]).lower() for c in out.columns]
    else:
        out.columns = [str(c).lower() for c in out.columns]
    wanted = ["open", "high", "low", "close"]
    missing = [c for c in wanted if c not in out.columns]
    if missing:
        raise ValueError(f"yfinance FX 컬럼 부족: {missing}, 실제={list(out.columns)}")
    return out[wanted]


def _fetch_brl_x_raw_ohlc(
    start: str | None = None,
    end_exclusive: str | None = None,
    *,
    auto_adjust: bool = False,
    progress: bool = False,
) -> pd.DataFrame:
    """BRL=X 일별 원본 OHLC (거래 있는 날만). Yahoo BRL=X 인용 규칙을 따른다."""
    df = yf.download(
        TICKER_USD_BRL,
        start=start or DEFAULT_START_DATE,
        end=end_exclusive,
        interval="1d",
        auto_adjust=auto_adjust,
        progress=progress,
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close"])
    df = _normalize_fx_ohlc(df)
    df = df.reset_index()
    date_col = df.columns[0]
    df["date"] = pd.to_datetime(df[date_col], utc=True).dt.strftime("%Y-%m-%d")
    df = df.drop(columns=[date_col])
    df = df[["date", "open", "high", "low", "close"]].drop_duplicates(subset=["date"], keep="last")
    df = df.dropna(subset=["close", "open", "high", "low"], how="any")
    return df


def _build_usd_brl_daily_with_ffill(
    df_raw: pd.DataFrame,
    start: str,
    end_inclusive: str,
) -> pd.DataFrame:
    """
    캘린더 일별로 reindex 후 주말·공휴일 결측을 forward-fill.
    보간 전 결측이었던 날 → is_interpolated=1, 원본 관측일 → 0.
    """
    empty_cols = [
        "date",
        "usd_brl_close",
        "usd_brl_open",
        "usd_brl_high",
        "usd_brl_low",
        "is_interpolated",
    ]
    if df_raw.empty:
        return pd.DataFrame(columns=empty_cols)

    daily_index = pd.date_range(start=start, end=end_inclusive, freq="D")
    base = df_raw.copy()
    base["dt"] = pd.to_datetime(base["date"])
    base = base.set_index("dt").sort_index()
    ohlc = base[["open", "high", "low", "close"]].rename(
        columns={
            "open": "usd_brl_open",
            "high": "usd_brl_high",
            "low": "usd_brl_low",
            "close": "usd_brl_close",
        }
    )
    expanded = ohlc.reindex(daily_index)
    needs_fill = expanded["usd_brl_close"].isna()
    filled = expanded.ffill()
    filled = filled.dropna(how="any")
    is_interp = needs_fill.reindex(filled.index).fillna(False).astype(int)
    filled = filled.copy()
    filled["is_interpolated"] = is_interp
    filled = filled.reset_index()
    filled["date"] = filled["index"].dt.strftime("%Y-%m-%d")
    filled = filled.drop(columns=["index"])
    column_order = [
        "date",
        "usd_brl_close",
        "usd_brl_open",
        "usd_brl_high",
        "usd_brl_low",
        "is_interpolated",
    ]
    return filled[column_order]


def _insert_raw_exchange_rate_ignore(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    """INSERT OR IGNORE."""
    if df.empty:
        return
    rows = list(
        zip(
            df["date"].astype(str),
            df["usd_brl_close"].astype(float),
            df["usd_brl_open"].astype(float),
            df["usd_brl_high"].astype(float),
            df["usd_brl_low"].astype(float),
            df["is_interpolated"].astype(int),
        )
    )
    conn.executemany(INSERT_EXCHANGE_IGNORE_SQL, rows)
    conn.commit()


def load_exchange_rate(
    db_path: str | Path | None = None,
    start: str | None = None,
    end_exclusive: str | None = None,
    *,
    print_count: bool = True,
) -> dict:
    """
    yfinance BRL=X로 USD/BRL 환율을 수집해 `raw_exchange_rate`에 적재한다.

    스키마: date TEXT PK, usd_brl_close/open/high/low REAL, is_interpolated INTEGER.
    주말·공휴일은 forward-fill 후 is_interpolated=1. 기간 기본: 2019-01-01 ~ 오늘(포함).
    중복 date: INSERT OR IGNORE.

    Returns
    -------
    dict
        rows_fetched, db_path
    """
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    start_s = start or DEFAULT_START_DATE
    end_ex = end_exclusive or _yfinance_end_exclusive_through()
    end_inclusive = (date.fromisoformat(end_ex) - timedelta(days=1)).isoformat()

    raw = _fetch_brl_x_raw_ohlc(start=start_s, end_exclusive=end_ex)
    built = _build_usd_brl_daily_with_ffill(raw, start=start_s, end_inclusive=end_inclusive)

    with sqlite3.connect(path) as conn:
        ensure_raw_exchange_rate_table(conn)
        _insert_raw_exchange_rate_ignore(conn, built)

    out = {"rows_fetched": len(built), "db_path": str(path.resolve())}
    if print_count:
        n_interp = int(built["is_interpolated"].sum()) if not built.empty else 0
        print(
            f"[{RAW_EXCHANGE_TABLE}] BRL=X: 적재 시도 {len(built)}행 "
            f"(보간 is_interpolated=1: {n_interp}행) | DB: {path.resolve()}"
        )
    return out


def load_price_spot(
    db_path: str | Path | None = None,
    *,
    print_message: bool = True,
) -> dict:
    """
    `raw_price_spot` 테이블만 보장한다(행은 넣지 않음).

    실제 데이터는 `load_worldbank_pink_sheet_soybean_oil` 또는 CEPEA 등 수동 적재로 넣는다.
    """
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        ensure_raw_price_spot_table(conn)
    if print_message:
        print(
            f"[{RAW_PRICE_SPOT_TABLE}] 테이블만 보장(행 없음). "
            f"World Bank·CEPEA 등 별도 적재 | DB: {path.resolve()}"
        )
    return {"rows_fetched": 0, "db_path": str(path.resolve()), "table": RAW_PRICE_SPOT_TABLE}


def _normalize_spot_column_names(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip() for c in out.columns]
    return out


def _pink_sheet_ym_code_ratio(s: pd.Series, *, sample: int = 300) -> float:
    """Pink Sheet 월 코드 `1960M01` 형태 비율."""
    sub = s.dropna().head(sample)
    if sub.empty:
        return 0.0
    pat = re.compile(r"^\d{4}[Mm]\d{2}$")
    return float(sum(pat.match(str(v).strip()) is not None for v in sub) / len(sub))


def _detect_date_column_for_worldbank(df: pd.DataFrame) -> str:
    """날짜 열 추정: Pink Sheet `YYYYMmm` / 일반 datetime / 이름에 month 등."""
    if df.empty or len(df.columns) == 0:
        raise ValueError("World Bank 시트가 비었거나 열이 없습니다.")

    def score_series(s: pd.Series) -> float:
        if _pink_sheet_ym_code_ratio(s) >= 0.5:
            return 1.0
        dt = pd.to_datetime(s, errors="coerce")
        return float(dt.notna().mean())

    first = df.columns[0]
    if score_series(df[first]) >= 0.5:
        return first

    for c in df.columns[1:]:
        cl = str(c).lower()
        if any(k in cl for k in ("date", "month", "period", "year", "unnamed")):
            if score_series(df[c]) >= 0.5:
                return c
    return first


def _find_soybean_oil_column(columns: list[str]) -> str:
    """Pink Sheet 표기 'Soybean oil' 등 대소문자·공백 무시 매칭."""
    for c in columns:
        s = str(c).strip().lower().replace("_", " ")
        if s == "soybean oil":
            return c
    for c in columns:
        s = str(c).strip().lower()
        if "soybean" in s and "oil" in s and "meal" not in s:
            return c
    raise ValueError(
        f"'Soybean oil'에 해당하는 열을 찾을 수 없습니다. 컬럼: {columns}"
    )


def _monthly_dates_to_month_first(dt_series: pd.Series) -> pd.Series:
    """
    월별 시계열 → 해당 월 1일 `YYYY-MM-DD` 문자열.
    World Bank Pink Sheet 코드 `1960M01` 지원.
    """
    out: list[str | None] = []
    for v in dt_series:
        if v is None or (isinstance(v, float) and pd.isna(v)):
            out.append(None)
            continue
        s = str(v).strip()
        m = re.match(r"^(\d{4})[Mm](\d{2})$", s)
        if m:
            out.append(f"{m.group(1)}-{int(m.group(2)):02d}-01")
            continue
        dt = pd.to_datetime(v, errors="coerce")
        if pd.isna(dt):
            out.append(None)
        else:
            ts = pd.Timestamp(dt).to_period("M").start_time
            out.append(ts.strftime("%Y-%m-%d"))
    return pd.Series(out, index=dt_series.index, dtype=object)


def _load_worldbank_monthly_prices_dataframe(
    path_xlsx: Path,
    sheet_name: str,
) -> pd.DataFrame:
    """
    공식 Pink Sheet xlsx: 상단 4행은 제목·단위 안내, 5행이 상품 헤더, 6행이 ($/…) 단위, 7행부터 데이터.
    """
    probe = pd.read_excel(path_xlsx, sheet_name=sheet_name, engine="openpyxl", nrows=1)
    c0 = str(probe.columns[0])
    if "World Bank Commodity Price Data" in c0 or "Pink Sheet" in c0:
        df = pd.read_excel(path_xlsx, sheet_name=sheet_name, engine="openpyxl", header=4)
        df = _normalize_spot_column_names(df)
        if len(df) > 0:
            df = df.iloc[1:].reset_index(drop=True)
        return df
    df = pd.read_excel(path_xlsx, sheet_name=sheet_name, engine="openpyxl")
    return _normalize_spot_column_names(df)


def load_worldbank_pink_sheet_soybean_oil(
    db_path: str | Path | None = None,
    filepath: str | Path | None = None,
    *,
    min_date: str = DEFAULT_START_DATE,
    sheet_name: str = WORLDBANK_MONTHLY_SHEET_NAME,
    print_summary: bool = True,
) -> dict:
    """
    World Bank Pink Sheet 월별 상품가 엑셀에서 대두유(Soybean oil) 열을 읽어 `raw_price_spot`에 적재한다.

    - 시트 기본: `Monthly Prices`
    - 월별 관측은 `YYYY-MM-01`로 저장
    - `min_date`(기본 2019-01-01) 이후만 INSERT OR IGNORE
    """
    path_xlsx = Path(filepath) if filepath is not None else WORLDBANK_COMMODITY_MONTHLY_DEFAULT_PATH
    if not path_xlsx.is_file():
        raise FileNotFoundError(f"World Bank 엑셀 없음: {path_xlsx.resolve()}")

    df = _load_worldbank_monthly_prices_dataframe(path_xlsx, sheet_name)
    print("World Bank Pink Sheet columns:", list(df.columns))

    date_col = _detect_date_column_for_worldbank(df)
    price_col = _find_soybean_oil_column(list(df.columns))

    dates = _monthly_dates_to_month_first(df[date_col])
    prices = pd.to_numeric(df[price_col], errors="coerce")

    out = pd.DataFrame({"date": dates, "spot_price_usd": prices})
    out = out.dropna(subset=["date", "spot_price_usd"])
    out = out[out["date"] >= min_date[:10]]
    out = out.drop_duplicates(subset=["date"], keep="last")

    db = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    db.parent.mkdir(parents=True, exist_ok=True)

    rows = [
        (r["date"], float(r["spot_price_usd"]), None, WORLDBANK_PINK_SHEET_SOURCE)
        for _, r in out.iterrows()
    ]

    with sqlite3.connect(db) as conn:
        ensure_raw_price_spot_table(conn)
        if rows:
            conn.executemany(INSERT_PRICE_SPOT_IGNORE_SQL, rows)
            conn.commit()

    n = len(rows)
    if print_summary and n:
        d0, d1 = out["date"].min(), out["date"].max()
        print(
            f"[{RAW_PRICE_SPOT_TABLE}] World Bank Pink Sheet (Soybean oil): "
            f"적재 시도 {n}행 | 날짜: {d0} ~ {d1} | DB: {db.resolve()}"
        )
    elif print_summary:
        print(
            f"[{RAW_PRICE_SPOT_TABLE}] World Bank Pink Sheet: 적재 0행 "
            f"(데이터 없음 또는 min_date={min_date} 이후 없음) | DB: {db.resolve()}"
        )

    return {"rows_fetched": n, "db_path": str(db.resolve()), "table": RAW_PRICE_SPOT_TABLE}


def ensure_raw_dollar_index_table(conn: sqlite3.Connection) -> None:
    conn.execute(CREATE_RAW_DOLLAR_INDEX_SQL)
    conn.commit()


def _normalize_dxy_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance 달러 지수 결과를 open, high, low, close 소문자로 맞춘다."""
    if df.empty:
        return df
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(c[0]).lower() for c in out.columns]
    else:
        out.columns = [str(c).lower() for c in out.columns]
    wanted = ["open", "high", "low", "close"]
    missing = [c for c in wanted if c not in out.columns]
    if missing:
        raise ValueError(f"yfinance DXY 컬럼 부족: {missing}, 실제={list(out.columns)}")
    return out[wanted]


def load_dollar_index(
    conn: sqlite3.Connection,
    start: str | None = None,
    end_exclusive: str | None = None,
    *,
    print_count: bool = True,
) -> int:
    """
    Yahoo Finance ``DX-Y.NYB`` 일별 OHLC → ``raw_dollar_index`` (INSERT OR IGNORE).

    컬럼: ``date``, ``dxy_close/open/high/low``. 기간 기본 2019-01-01 ~ 오늘(포함).
    """
    start_s = start or DEFAULT_START_DATE
    end_ex = end_exclusive or _yfinance_end_exclusive_through()
    df = yf.download(
        TICKER_DOLLAR_INDEX,
        start=start_s,
        end=end_ex,
        interval="1d",
        auto_adjust=False,
        progress=False,
    )
    if df is None or df.empty:
        ensure_raw_dollar_index_table(conn)
        if print_count:
            print(f"[{RAW_DOLLAR_INDEX_TABLE}] {TICKER_DOLLAR_INDEX}: 데이터 없음 → 테이블만 보장")
        return 0
    df = _normalize_dxy_ohlc(df)
    df = df.reset_index()
    date_col = df.columns[0]
    df["date"] = pd.to_datetime(df[date_col], utc=True).dt.strftime("%Y-%m-%d")
    df = df.drop(columns=[date_col])
    df = df.rename(
        columns={
            "open": "dxy_open",
            "high": "dxy_high",
            "low": "dxy_low",
            "close": "dxy_close",
        }
    )
    df = df.dropna(subset=["dxy_close", "dxy_open", "dxy_high", "dxy_low"], how="any")
    df = df.drop_duplicates(subset=["date"], keep="last")

    ensure_raw_dollar_index_table(conn)
    rows = list(
        zip(
            df["date"].astype(str),
            df["dxy_close"].astype(float),
            df["dxy_open"].astype(float),
            df["dxy_high"].astype(float),
            df["dxy_low"].astype(float),
        )
    )
    if rows:
        conn.executemany(INSERT_DOLLAR_INDEX_IGNORE_SQL, rows)
        conn.commit()
    n = len(rows)
    if print_count:
        print(
            f"[{RAW_DOLLAR_INDEX_TABLE}] {TICKER_DOLLAR_INDEX}: 적재 시도 {n}행 "
            f"| {df['date'].min()} ~ {df['date'].max()}"
        )
    return n


def ensure_raw_sunflower_oil_table(conn: sqlite3.Connection) -> None:
    conn.execute(CREATE_RAW_SUNFLOWER_OIL_SQL)
    conn.commit()


def ensure_raw_eia_biodiesel_table(conn: sqlite3.Connection) -> None:
    conn.execute(CREATE_RAW_EIA_BIODIESEL_SQL)
    cols = {
        str(r[1]).strip().lower()
        for r in conn.execute(f"PRAGMA table_info({RAW_EIA_BIODIESEL_TABLE})").fetchall()
    }
    if "soybean_oil_biofuel_mmlb" not in cols:
        conn.execute(f"DROP TABLE IF EXISTS {RAW_EIA_BIODIESEL_TABLE}")
        conn.execute(CREATE_RAW_EIA_BIODIESEL_SQL)
    conn.commit()


def ensure_raw_fed_rate_table(conn: sqlite3.Connection) -> None:
    conn.execute(CREATE_RAW_FED_RATE_SQL)
    conn.commit()


def ensure_raw_vix_table(conn: sqlite3.Connection) -> None:
    conn.execute(CREATE_RAW_VIX_SQL)
    conn.commit()


def load_eia_biodiesel(
    conn: sqlite3.Connection,
    start: str = DEFAULT_START_DATE,
    *,
    print_summary: bool = True,
) -> int:
    """
    EIA DNAV XLS 공개 파일에서 Total Soybean Oil 월간 시리즈를 읽어 적재한다.
    - URL: ``PET_PNP_FEEDBIOFUEL_DCU_NUS_M.xls``
    - 시리즈 ID: ``M_EPOOBDSO_YIFBP_NUS_MMLB``
    - 단위: Million Pounds
    - ``INSERT OR IGNORE``

    Returns
    -------
    int
        적재 후 ``SELECT COUNT(*)`` 행 수.
    """
    ensure_raw_eia_biodiesel_table(conn)
    start_s = (start or DEFAULT_START_DATE)[:10]
    try:
        resp = requests.get(
            EIA_DNAV_XLS_URL,
            headers={"User-Agent": "Mozilla/5.0 (compatible; soybean-oil-poc/1.0)"},
            timeout=90,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        warnings.warn(f"EIA DNAV XLS 다운로드 실패: {e}", UserWarning, stacklevel=2)
        if print_summary:
            print(f"[{RAW_EIA_BIODIESEL_TABLE}] XLS 다운로드 실패 — 적재 0행")
        return int(conn.execute(f"SELECT COUNT(*) FROM {RAW_EIA_BIODIESEL_TABLE}").fetchone()[0])

    workbook = pd.ExcelFile(BytesIO(resp.content), engine="xlrd")
    sheet_candidates = [EIA_DNAV_SHEET_NAME] + [
        s for s in workbook.sheet_names if str(s).startswith("Data") and s != EIA_DNAV_SHEET_NAME
    ]
    data = None
    value_col = None
    for sh in sheet_candidates:
        raw = pd.read_excel(BytesIO(resp.content), sheet_name=sh, header=None, engine="xlrd")
        if raw.empty or len(raw) < 3:
            continue
        source_keys = [str(v).strip() for v in raw.iloc[1].tolist() if pd.notna(v)]
        if not any(EIA_DNAV_SERIES_ID in v for v in source_keys):
            continue
        headers = [str(v).strip() for v in raw.iloc[2].tolist()]
        tmp = raw.iloc[3:].copy()
        tmp.columns = headers
        tmp = tmp.dropna(how="all")
        date_col = next((c for c in tmp.columns if "date" in str(c).lower()), tmp.columns[0])
        for c in tmp.columns:
            cl = str(c).lower()
            if "total soybean oil" in cl or "feedstocks consumed for production of biofuels" in cl:
                value_col = c
                break
        if value_col is None:
            idx = source_keys.index(EIA_DNAV_SERIES_ID)
            if idx < len(tmp.columns):
                value_col = tmp.columns[idx]
        data = tmp[[date_col, value_col]].copy()
        data.columns = ["date", "value"]
        break

    if data is None or value_col is None:
        raise ValueError(f"EIA DNAV XLS에서 시리즈 {EIA_DNAV_SERIES_ID}를 찾지 못했습니다.")

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(data["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    out["soybean_oil_biofuel_mmlb"] = pd.to_numeric(
        data["value"].astype(str).str.replace(",", "", regex=False),
        errors="coerce",
    )
    out = out.dropna(subset=["date", "soybean_oil_biofuel_mmlb"])
    out = out[out["date"] >= start_s]
    out = out.drop_duplicates(subset=["date"], keep="last").sort_values("date")

    rows = [
        (r["date"], float(r["soybean_oil_biofuel_mmlb"]), EIA_BIODIESEL_SOURCE)
        for _, r in out.iterrows()
    ]
    if rows:
        conn.executemany(INSERT_EIA_BIODIESEL_IGNORE_SQL, rows)
        conn.commit()

    n_final = int(conn.execute(f"SELECT COUNT(*) FROM {RAW_EIA_BIODIESEL_TABLE}").fetchone()[0])
    if print_summary and rows:
        d0, d1 = rows[0][0], rows[-1][0]
        print(
            f"[{RAW_EIA_BIODIESEL_TABLE}] EIA DNAV Total Soybean Oil: 적재 시도 {len(rows)}행 "
            f"(고유일자) | {d0} ~ {d1} | 테이블 행 수 {n_final}"
        )
        month_index = pd.DatetimeIndex(pd.to_datetime(out["date"], errors="coerce").dropna()).to_period("M")
        if len(month_index) > 0:
            full_months = pd.period_range(start=month_index.min(), end=month_index.max(), freq="M")
            missing = [p.strftime("%Y-%m") for p in full_months.difference(month_index.unique())]
            print(f"[{RAW_EIA_BIODIESEL_TABLE}] 데이터 시작일: {d0} | 종료일: {d1}")
            if missing:
                print(f"[{RAW_EIA_BIODIESEL_TABLE}] 누락 월({len(missing)}): {', '.join(missing)}")
            else:
                print(f"[{RAW_EIA_BIODIESEL_TABLE}] 중간 누락 월 없음")
    elif print_summary and not rows:
        print(f"[{RAW_EIA_BIODIESEL_TABLE}] 적재 0행 (DNAV 파싱 불가 또는 데이터 없음).")
    return n_final


def load_fed_rate(
    conn: sqlite3.Connection,
    start: str = DEFAULT_START_DATE,
    *,
    print_summary: bool = True,
) -> int:
    """FRED FEDFUNDS(월별) CSV를 ``raw_fed_rate``에 적재한다."""
    ensure_raw_fed_rate_table(conn)
    start_s = (start or DEFAULT_START_DATE)[:10]
    try:
        resp = requests.get(
            FRED_FEDFUNDS_URL,
            headers={"User-Agent": "Mozilla/5.0 (compatible; soybean-oil-poc/1.0)"},
            timeout=60,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        warnings.warn(f"FRED FEDFUNDS 다운로드 실패: {e}", UserWarning, stacklevel=2)
        if print_summary:
            print(f"[{RAW_FED_RATE_TABLE}] 다운로드 실패 — 적재 0행")
        return int(conn.execute(f"SELECT COUNT(*) FROM {RAW_FED_RATE_TABLE}").fetchone()[0])

    df = pd.read_csv(StringIO(resp.text))
    if df.empty or len(df.columns) < 2:
        if print_summary:
            print(f"[{RAW_FED_RATE_TABLE}] CSV 파싱 실패 — 적재 0행")
        return int(conn.execute(f"SELECT COUNT(*) FROM {RAW_FED_RATE_TABLE}").fetchone()[0])
    date_col = df.columns[0]
    value_col = df.columns[1]

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    out["fed_rate"] = pd.to_numeric(df[value_col], errors="coerce")
    out = out.dropna(subset=["date", "fed_rate"])
    out = out[out["date"] >= start_s]
    out = out.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    rows = [(r["date"], float(r["fed_rate"]), FRED_SOURCE) for _, r in out.iterrows()]
    if rows:
        conn.executemany(INSERT_FED_RATE_IGNORE_SQL, rows)
        conn.commit()

    n_final = int(conn.execute(f"SELECT COUNT(*) FROM {RAW_FED_RATE_TABLE}").fetchone()[0])
    if print_summary and rows:
        print(
            f"[{RAW_FED_RATE_TABLE}] FRED FEDFUNDS: 적재 시도 {len(rows)}행 | "
            f"{rows[0][0]} ~ {rows[-1][0]} | 테이블 행 수 {n_final}"
        )
    elif print_summary:
        print(f"[{RAW_FED_RATE_TABLE}] 적재 0행 (응답 없음 또는 파싱 불가)")
    return n_final


def load_vix(
    conn: sqlite3.Connection,
    start: str = DEFAULT_START_DATE,
    *,
    print_summary: bool = True,
) -> int:
    """FRED VIXCLS(일별) CSV를 ``raw_vix``에 적재한다."""
    ensure_raw_vix_table(conn)
    start_s = (start or DEFAULT_START_DATE)[:10]
    try:
        resp = requests.get(
            FRED_VIX_URL,
            headers={"User-Agent": "Mozilla/5.0 (compatible; soybean-oil-poc/1.0)"},
            timeout=60,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        warnings.warn(f"FRED VIX 다운로드 실패: {e}", UserWarning, stacklevel=2)
        if print_summary:
            print(f"[{RAW_VIX_TABLE}] 다운로드 실패 — 적재 0행")
        return int(conn.execute(f"SELECT COUNT(*) FROM {RAW_VIX_TABLE}").fetchone()[0])

    df = pd.read_csv(StringIO(resp.text))
    if df.empty or len(df.columns) < 2:
        if print_summary:
            print(f"[{RAW_VIX_TABLE}] CSV 파싱 실패 — 적재 0행")
        return int(conn.execute(f"SELECT COUNT(*) FROM {RAW_VIX_TABLE}").fetchone()[0])
    date_col = df.columns[0]
    value_col = df.columns[1]

    out = pd.DataFrame()
    out["date"] = pd.to_datetime(df[date_col], errors="coerce").dt.strftime("%Y-%m-%d")
    out["vix_close"] = pd.to_numeric(df[value_col], errors="coerce")
    out = out.dropna(subset=["date", "vix_close"])
    out = out[out["date"] >= start_s]
    out = out.drop_duplicates(subset=["date"], keep="last").sort_values("date")
    rows = [(r["date"], float(r["vix_close"]), FRED_SOURCE) for _, r in out.iterrows()]
    if rows:
        conn.executemany(INSERT_VIX_IGNORE_SQL, rows)
        conn.commit()

    n_final = int(conn.execute(f"SELECT COUNT(*) FROM {RAW_VIX_TABLE}").fetchone()[0])
    if print_summary:
        print(f"[{RAW_VIX_TABLE}] FRED VIXCLS: 적재 시도 {len(rows)}행 | 테이블 행 수 {n_final}")
    return n_final


def _find_sunflower_oil_column(columns: list[str]) -> str:
    """Pink Sheet 'Sunflower oil' 열 탐색."""
    for c in columns:
        s = str(c).strip().lower().replace("_", " ")
        if s == "sunflower oil":
            return c
    for c in columns:
        s = str(c).strip().lower()
        if "sunflower" in s and "oil" in s:
            return c
    raise ValueError(
        f"'Sunflower oil'에 해당하는 열을 찾을 수 없습니다. 컬럼: {columns}"
    )


def load_sunflower_oil(
    conn: sqlite3.Connection,
    filepath: str | Path | None = None,
    *,
    min_date: str = DEFAULT_START_DATE,
    sheet_name: str = WORLDBANK_MONTHLY_SHEET_NAME,
    print_summary: bool = True,
) -> int:
    """
    World Bank Pink Sheet 월별 엑셀에서 **Sunflower oil** 열을 읽어 ``raw_sunflower_oil``에 적재한다.

    ``min_date``(기본 2019-01-01) 이후만 INSERT OR IGNORE.
    """
    path_xlsx = Path(filepath) if filepath is not None else WORLDBANK_COMMODITY_MONTHLY_DEFAULT_PATH
    if not path_xlsx.is_file():
        raise FileNotFoundError(f"World Bank 엑셀 없음: {path_xlsx.resolve()}")

    df = _load_worldbank_monthly_prices_dataframe(path_xlsx, sheet_name)
    date_col = _detect_date_column_for_worldbank(df)
    price_col = _find_sunflower_oil_column(list(df.columns))

    dates = _monthly_dates_to_month_first(df[date_col])
    prices = pd.to_numeric(df[price_col], errors="coerce")

    out = pd.DataFrame({"date": dates, "sunflower_close": prices})
    out = out.dropna(subset=["date", "sunflower_close"])
    out = out[out["date"] >= min_date[:10]]
    out = out.drop_duplicates(subset=["date"], keep="last")

    ensure_raw_sunflower_oil_table(conn)
    rows = [
        (r["date"], float(r["sunflower_close"]), WORLDBANK_PINK_SHEET_SOURCE)
        for _, r in out.iterrows()
    ]
    if rows:
        conn.executemany(INSERT_SUNFLOWER_OIL_IGNORE_SQL, rows)
        conn.commit()

    n = len(rows)
    if print_summary and n:
        d0, d1 = out["date"].min(), out["date"].max()
        print(
            f"[{RAW_SUNFLOWER_OIL_TABLE}] World Bank (Sunflower oil): 적재 시도 {n}행 "
            f"| {d0} ~ {d1}"
        )
    elif print_summary:
        print(
            f"[{RAW_SUNFLOWER_OIL_TABLE}] 적재 0행 "
            f"(min_date={min_date} 이후 없음 또는 열 없음)"
        )
    return n


def ingest_usd_brl_to_sqlite(
    db_path: str | Path | None = None,
    start: str | None = None,
    end_exclusive: str | None = None,
    *,
    print_count: bool = True,
) -> dict:
    """하위 호환: `load_exchange_rate`와 동일."""
    return load_exchange_rate(
        db_path=db_path,
        start=start,
        end_exclusive=end_exclusive,
        print_count=print_count,
    )


# 이전 이름 호환 (외부에서 import 했을 수 있음)
fetch_brl_x_raw_ohlc = _fetch_brl_x_raw_ohlc
build_usd_brl_daily_with_ffill = _build_usd_brl_daily_with_ffill
insert_raw_exchange_rate_ignore = _insert_raw_exchange_rate_ignore


if __name__ == "__main__":
    load_exchange_rate(print_count=True)
    load_price_spot(print_message=True)
    p = WORLDBANK_COMMODITY_MONTHLY_DEFAULT_PATH
    if p.is_file():
        load_worldbank_pink_sheet_soybean_oil(filepath=p, print_summary=True)
    else:
        print(f"World Bank 엑셀 없음 — 건너뜀: {p}")
