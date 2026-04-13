from __future__ import annotations

import logging
import sqlite3
import warnings
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yfinance as yf

logger = logging.getLogger("src.ingestion.price")

# .cursorrules 기본 DB
DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "db" / "soybean.db"
DEFAULT_START_DATE = "2010-01-01"

# (SQLite 테이블명, Yahoo 티커, commodity 컬럼 값, 실패 시 빈 테이블만 허용 여부)
FUTURES_SPECS: list[dict[str, Any]] = [
    {"table": "raw_price_futures", "ticker": "ZL=F", "commodity": "ZL=F", "optional": False},
    {"table": "raw_crude_oil", "ticker": "CL=F", "commodity": "CL=F", "optional": False},
    {"table": "raw_soybean_futures", "ticker": "ZS=F", "commodity": "ZS=F", "optional": False},
    {"table": "raw_soymeal_futures", "ticker": "ZM=F", "commodity": "ZM=F", "optional": False},
    # 팜유(FCPO): Yahoo 심볼 KPO=F 시도 — 없거나 오류 시 빈 테이블만 두고 로그만 남김
    {"table": "raw_palm_oil", "ticker": "KPO=F", "commodity": "KPO=F", "optional": True},
]

# 하위 호환용 별칭
TICKER_ZL = "ZL=F"
RAW_PRICE_FUTURES_TABLE = "raw_price_futures"

RAW_PALM_OIL_TABLE = "raw_palm_oil"
PALM_OIL_CSV_COMMODITY = "CPOc1"
PALM_OIL_CSV_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "palm_oil.csv"

RAW_CANOLA_OIL_TABLE = "raw_canola_oil"
CANOLA_COMMODITY_DEFAULT = "RSc1"
CANOLA_OIL_CSV_DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "canola_oil.csv"

RAW_CANOLA_OIL_DDL = f"""
CREATE TABLE IF NOT EXISTS {RAW_CANOLA_OIL_TABLE} (
    date TEXT PRIMARY KEY,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    commodity TEXT DEFAULT '{CANOLA_COMMODITY_DEFAULT}'
);
"""

RAW_PALM_OIL_DDL = f"""
CREATE TABLE IF NOT EXISTS {RAW_PALM_OIL_TABLE} (
    date TEXT PRIMARY KEY,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    commodity TEXT DEFAULT '{PALM_OIL_CSV_COMMODITY}'
);
"""


def _table_ddl(table: str) -> str:
    return f"""
CREATE TABLE IF NOT EXISTS {table} (
    date TEXT PRIMARY KEY,
    open REAL,
    high REAL,
    low REAL,
    close REAL,
    volume REAL,
    commodity TEXT
);
"""


def _insert_sql(table: str) -> str:
    return f"""
INSERT OR IGNORE INTO {table}
    (date, open, high, low, close, volume, commodity)
VALUES (?, ?, ?, ?, ?, ?, ?);
"""


def ensure_futures_table(conn: sqlite3.Connection, table: str) -> None:
    """지정 raw 테이블이 없으면 생성한다 (date PK + OHLCV + commodity)."""
    if table == RAW_PALM_OIL_TABLE:
        conn.execute(RAW_PALM_OIL_DDL)
    elif table == RAW_CANOLA_OIL_TABLE:
        conn.execute(RAW_CANOLA_OIL_DDL)
    else:
        conn.execute(_table_ddl(table))
    conn.commit()


def ensure_raw_palm_oil_table(conn: sqlite3.Connection) -> None:
    """`raw_palm_oil` 보장 (commodity 기본값 CPOc1)."""
    ensure_futures_table(conn, RAW_PALM_OIL_TABLE)


def ensure_raw_canola_oil_table(conn: sqlite3.Connection) -> None:
    """`raw_canola_oil` 보장 (commodity 기본값 RSc1)."""
    conn.execute(RAW_CANOLA_OIL_DDL)
    conn.commit()


def ensure_raw_price_futures_table(conn: sqlite3.Connection) -> None:
    """`raw_price_futures` 테이블 보장 (레거시 호출명)."""
    ensure_futures_table(conn, RAW_PRICE_FUTURES_TABLE)


def _normalize_yahoo_columns(df: pd.DataFrame) -> pd.DataFrame:
    """yfinance 단일/멀티 인덱스 컬럼을 open, high, low, close, volume 소문자로 맞춘다."""
    if df.empty:
        return df
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(c[0]).lower() for c in out.columns]
    else:
        out.columns = [str(c).lower() for c in out.columns]
    wanted = ["open", "high", "low", "close", "volume"]
    missing = [c for c in wanted if c not in out.columns]
    if missing:
        raise ValueError(f"yfinance 컬럼 부족: {missing}, 실제 컬럼={list(out.columns)}")
    return out[wanted]


def fetch_yahoo_futures_ohlcv(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    *,
    auto_adjust: bool = False,
    progress: bool = False,
) -> pd.DataFrame:
    """
    Yahoo Finance에서 단일 티커 일별 OHLCV를 내려받는다.

    end는 yfinance 규칙상 배타적이므로, 오늘 거래일을 포함하려면 end를 내일 날짜로 넘긴다.
    """
    df = yf.download(
        ticker,
        start=start,
        end=end,
        interval="1d",
        auto_adjust=auto_adjust,
        progress=progress,
    )
    if df is None or df.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    df = _normalize_yahoo_columns(df)
    df = df.reset_index()
    date_col = df.columns[0]
    df["date"] = pd.to_datetime(df[date_col], utc=True).dt.strftime("%Y-%m-%d")
    df = df.drop(columns=[date_col])
    df = df[["date", "open", "high", "low", "close", "volume"]]
    df = df.dropna(subset=["open", "high", "low", "close"])
    return df


def fetch_zl_futures_ohlcv(
    start: str | None = None,
    end: str | None = None,
    *,
    auto_adjust: bool = False,
    progress: bool = False,
) -> pd.DataFrame:
    """ZL=F 일별 OHLCV (레거시 호출명)."""
    return fetch_yahoo_futures_ohlcv(
        TICKER_ZL, start=start, end=end, auto_adjust=auto_adjust, progress=progress
    )


def validate_price_futures_df(df: pd.DataFrame) -> bool:
    """필수 컬럼, date 형식, OHLC 논리(high>=low) 검증."""
    required = {"date", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        return False
    if df["date"].isna().any():
        return False
    if not bool(df["date"].astype(str).str.match(r"^\d{4}-\d{2}-\d{2}$").all()):
        return False
    if (df["high"] < df["low"]).any():
        return False
    return True


def _real_or_null(series: pd.Series) -> list[float | None]:
    out: list[float | None] = []
    for v in series:
        out.append(None if pd.isna(v) else float(v))
    return out


def insert_futures_ohlcv_ignore(
    conn: sqlite3.Connection,
    table: str,
    df: pd.DataFrame,
    commodity: str,
) -> None:
    """DataFrame을 지정 테이블에 INSERT OR IGNORE로 적재한다 (date 중복 무시)."""
    if df.empty:
        return
    if not validate_price_futures_df(df):
        raise ValueError("OHLCV DataFrame 검증 실패 (validate_price_futures_df).")
    rows = list(
        zip(
            df["date"].astype(str),
            df["open"].astype(float),
            df["high"].astype(float),
            df["low"].astype(float),
            df["close"].astype(float),
            _real_or_null(df["volume"]),
            [commodity] * len(df),
        )
    )
    conn.executemany(_insert_sql(table), rows)
    conn.commit()


def insert_raw_price_futures_ignore(conn: sqlite3.Connection, df: pd.DataFrame) -> None:
    """`raw_price_futures`에 ZL=F commodity로 적재 (레거시 호출명)."""
    insert_futures_ohlcv_ignore(conn, RAW_PRICE_FUTURES_TABLE, df, TICKER_ZL)


def _yfinance_end_exclusive_through(today: date | None = None) -> str:
    """일봉 다운로드 시 오늘까지 포함되도록 yfinance 배타적 end 날짜를 반환한다."""
    d = today or date.today()
    return (d + timedelta(days=1)).isoformat()


def ingest_one_future_to_sqlite(
    conn: sqlite3.Connection,
    table: str,
    ticker: str,
    commodity: str,
    start: str,
    end_exclusive: str,
    *,
    optional: bool = False,
) -> int:
    """
    단일 티커를 내려받아 테이블에 적재한다.

    Returns
    -------
    int
        적재 시도한 행 수(INSERT OR IGNORE로 일부는 무시될 수 있음).
    """
    ensure_futures_table(conn, table)
    if optional:
        try:
            df = fetch_yahoo_futures_ohlcv(ticker, start=start, end=end_exclusive)
        except Exception as e:
            logger.warning(
                "팜유(KPO=F) Yahoo Finance 수집 실패 — 빈 테이블만 유지합니다: %s: %s",
                type(e).__name__,
                e,
            )
            return 0
        if df.empty:
            logger.warning(
                "팜유(KPO=F) Yahoo Finance에서 데이터가 비었습니다. "
                "심볼 미제공 또는 제한일 수 있습니다. 테이블 `%s`는 스키마만 보장합니다.",
                table,
            )
            return 0
    else:
        df = fetch_yahoo_futures_ohlcv(ticker, start=start, end=end_exclusive)

    insert_futures_ohlcv_ignore(conn, table, df, commodity)
    return len(df)


def ingest_all_futures_to_sqlite(
    db_path: str | Path | None = None,
    start: str | None = None,
    end_exclusive: str | None = None,
    *,
    print_counts: bool = True,
) -> list[dict[str, Any]]:
    """
    설정된 모든 티커를 수집·적재한다.

    기본 기간: 2019-01-01 ~ 오늘(포함)까지. end_exclusive 미지정 시 내일 날짜를 사용한다.

    Returns
    -------
    list[dict]
        각 항목: table, ticker, rows_fetched
    """
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    start_s = start or DEFAULT_START_DATE
    end_ex = end_exclusive or _yfinance_end_exclusive_through()

    results: list[dict[str, Any]] = []
    with sqlite3.connect(path) as conn:
        for spec in FUTURES_SPECS:
            table = spec["table"]
            ticker = spec["ticker"]
            commodity = spec["commodity"]
            optional = bool(spec.get("optional", False))
            n = ingest_one_future_to_sqlite(
                conn,
                table,
                ticker,
                commodity,
                start_s,
                end_ex,
                optional=optional,
            )
            results.append(
                {"table": table, "ticker": ticker, "rows_fetched": n, "commodity": commodity}
            )
            if print_counts:
                print(f"[{table}] {ticker}: 수집·적재 시도 {n}행")

    if print_counts:
        total = sum(r["rows_fetched"] for r in results)
        print(f"합계(적재 시도 행 수): {total}행 | DB: {path.resolve()}")

    return results


def ingest_zl_futures_to_sqlite(
    db_path: str | Path,
    start: str | None = None,
    end: str | None = None,
) -> dict:
    """
    ZL=F만 내려받아 `raw_price_futures`에 적재한다 (레거시).

    end가 None이면 오늘까지 포함되도록 내부에서 배타적 end를 계산한다.
    """
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    start_s = start or DEFAULT_START_DATE
    end_ex = end if end is not None else _yfinance_end_exclusive_through()
    with sqlite3.connect(path) as conn:
        ensure_futures_table(conn, RAW_PRICE_FUTURES_TABLE)
        df = fetch_yahoo_futures_ohlcv(TICKER_ZL, start=start_s, end=end_ex)
        insert_futures_ohlcv_ignore(conn, RAW_PRICE_FUTURES_TABLE, df, TICKER_ZL)
    return {"rows_fetched": len(df), "db_path": str(path.resolve())}


def load_price_excel(path: str) -> None:
    """엑셀 가격 파일을 읽어 raw 테이블에 적재한다 (미구현)."""
    raise NotImplementedError("엑셀 적재는 별도 스프린트에서 구현.")


def upsert_raw_prices(conn, df) -> None:
    """DataFrame을 raw_price_futures에 INSERT OR IGNORE (commodity는 ZL=F 고정)."""
    insert_raw_price_futures_ignore(conn, df)


def validate_price_schema(df) -> bool:
    """OHLCV 스키마 검증."""
    return validate_price_futures_df(df)


def _col_key_map(columns: list[Any]) -> dict[str, str]:
    """소문자 정규화 키 → 원본 열 이름."""
    out: dict[str, str] = {}
    for c in columns:
        k = str(c).strip().strip('"').lower()
        out[k] = str(c).strip()
    return out


def _detect_date_column(columns: list[str]) -> str:
    """날짜 열 자동 선택: date/datetime/time 이름 우선, 없으면 첫 열."""
    cmap = _col_key_map(columns)
    for cand in ("date", "datetime", "time"):
        if cand in cmap:
            return cmap[cand]
    for k, orig in cmap.items():
        if "date" in k or k.endswith("time"):
            return orig
    return columns[0]


def _parse_investing_price_number(x: Any) -> float | None:
    """쉼표·공백·따옴표 제거 후 실수."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip().strip('"').replace("\u00a0", " ")
    if not s or s.lower() in ("nan", "-", "—"):
        return None
    s = s.replace(",", "").replace(" ", "")
    try:
        return float(s)
    except ValueError:
        return None


def _parse_investing_volume(x: Any) -> float | None:
    """Investing Vol. 열: 빈 값, 1.5K / 2.3M 등 처리."""
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip().strip('"').replace("\u00a0", " ").upper()
    if not s or s in ("-", "—"):
        return None
    mult = 1.0
    if s.endswith("K"):
        mult = 1_000.0
        s = s[:-1].strip()
    elif s.endswith("M"):
        mult = 1_000_000.0
        s = s[:-1].strip()
    elif s.endswith("B"):
        mult = 1_000_000_000.0
        s = s[:-1].strip()
    s = s.replace(",", "").replace(" ", "")
    try:
        return float(s) * mult
    except ValueError:
        return None


def _resolve_ohlcv_column_names(columns: list[str]) -> tuple[str, str, str, str, str | None]:
    """date, open, high, low, close(또는 price), volume(없으면 None)."""
    cmap = _col_key_map(columns)
    date_col = _detect_date_column(columns)

    def pick(*cands: str) -> str | None:
        for c in cands:
            if c in cmap:
                return cmap[c]
        return None

    o = pick("open")
    h = pick("high")
    lo = pick("low")
    cl = pick("close", "price", "last", "adj close", "adj. close")
    vol = pick("volume", "vol.", "vol")
    if o is None or h is None or lo is None or cl is None:
        raise ValueError(
            f"OHLC/Price 열을 찾을 수 없습니다. 컬럼={columns}"
        )
    return date_col, o, h, lo, cl, vol


def load_canola_oil(
    conn: sqlite3.Connection,
    filepath: str | Path | None = None,
    *,
    print_count: bool = True,
) -> int:
    """
    Investing.com 등에서 받은 카놀라 일별 CSV(``RSc1``)를 ``raw_canola_oil``에 적재한다.

    - 컬럼 목록을 먼저 출력한다.
    - 날짜 열 자동 감지 후 ``YYYY-MM-DD`` 변환.
    - Price/Open/High/Low의 쉼표 제거 후 REAL 변환.
    - Volume의 K/M/B 접미사 확장.
    - 기존 테이블 ``DELETE`` 후 ``INSERT OR IGNORE``.
    - 적재 후 행 수/날짜 범위를 출력한다.
    """
    ensure_raw_canola_oil_table(conn)
    conn.execute(f"DELETE FROM {RAW_CANOLA_OIL_TABLE}")
    conn.commit()

    path = Path(filepath) if filepath is not None else CANOLA_OIL_CSV_DEFAULT_PATH
    if not path.is_file():
        msg = f"카놀라 CSV 없음: {path.resolve()} — raw_canola_oil은 빈 테이블입니다."
        warnings.warn(msg, UserWarning, stacklevel=2)
        if print_count:
            print(f"[{RAW_CANOLA_OIL_TABLE}] 경고: {msg}")
        return int(conn.execute(f"SELECT COUNT(*) FROM {RAW_CANOLA_OIL_TABLE}").fetchone()[0])

    header = pd.read_csv(path, nrows=0)
    cols_list = [str(c).strip() for c in header.columns]
    print("CSV columns:", cols_list)

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    date_c, o_c, h_c, l_c, c_c, v_c = _resolve_ohlcv_column_names(list(df.columns))

    dt = pd.to_datetime(df[date_c], dayfirst=False, errors="coerce")
    if dt.isna().any():
        mask = dt.isna()
        dt = dt.copy()
        dt.loc[mask] = pd.to_datetime(df.loc[mask, date_c], dayfirst=True, errors="coerce")
    df = df.assign(
        date=dt.dt.strftime("%Y-%m-%d"),
        open=df[o_c].map(_parse_investing_price_number),
        high=df[h_c].map(_parse_investing_price_number),
        low=df[l_c].map(_parse_investing_price_number),
        close=df[c_c].map(_parse_investing_price_number),
    )
    if v_c is not None:
        df["volume"] = df[v_c].map(_parse_investing_volume)
    else:
        df["volume"] = None

    df = df.dropna(subset=["date", "open", "high", "low", "close"])
    df = df[df["date"].str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)]
    if (df["high"] < df["low"]).any():
        bad = df[df["high"] < df["low"]]
        raise ValueError(f"canola high < low 인 행이 있습니다: {len(bad)}행")

    out = df[["date", "open", "high", "low", "close", "volume"]].copy()
    if not validate_price_futures_df(out):
        raise ValueError("카놀라 CSV 정규화 후 검증 실패 (validate_price_futures_df).")

    insert_futures_ohlcv_ignore(conn, RAW_CANOLA_OIL_TABLE, out, CANOLA_COMMODITY_DEFAULT)
    if print_count and not out.empty:
        print(
            f"[{RAW_CANOLA_OIL_TABLE}] CSV 적재 시도 {len(out)}행 "
            f"| 날짜 범위: {out['date'].min()} ~ {out['date'].max()} "
            f"| commodity={CANOLA_COMMODITY_DEFAULT}"
        )

    n_final = int(conn.execute(f"SELECT COUNT(*) FROM {RAW_CANOLA_OIL_TABLE}").fetchone()[0])
    if print_count:
        print(f"[{RAW_CANOLA_OIL_TABLE}] 테이블 행 수: {n_final}")
    return n_final


def load_palm_oil_from_investing_csv(
    conn: sqlite3.Connection,
    filepath: str | Path | None = None,
) -> int:
    """
    Investing.com 등에서 받은 팜유 일별 CSV를 `raw_palm_oil`에 INSERT OR IGNORE로 적재한다.

    - 날짜 열 자동 감지 후 `YYYY-MM-DD`로 통일 (Investing.com은 MM/DD/YYYY 우선, 실패 시 DD/MM 시도).
    - 가격 열의 천 단위 쉼표 제거.
    - 거래량 `K`/`M`/`B` 접미사 확장.

    Parameters
    ----------
    filepath
        기본: `data/raw/palm_oil.csv`

    Returns
    -------
    int
        INSERT 시도 행 수.
    """
    path = Path(filepath) if filepath is not None else PALM_OIL_CSV_DEFAULT_PATH
    if not path.is_file():
        raise FileNotFoundError(f"팜유 CSV 없음: {path.resolve()}")

    header = pd.read_csv(path, nrows=0)
    cols_list = [str(c).strip() for c in header.columns]
    print("CSV columns:", cols_list)

    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    date_c, o_c, h_c, l_c, c_c, v_c = _resolve_ohlcv_column_names(list(df.columns))

    # Investing.com 수출은 보통 MM/DD/YYYY. dayfirst=True는 03/31 형태에서 월>12로 NaT가 난다.
    dt = pd.to_datetime(df[date_c], dayfirst=False, errors="coerce")
    if dt.isna().any():
        mask = dt.isna()
        dt = dt.copy()
        dt.loc[mask] = pd.to_datetime(df.loc[mask, date_c], dayfirst=True, errors="coerce")
    df = df.assign(
        date=dt.dt.strftime("%Y-%m-%d"),
        open=df[o_c].map(_parse_investing_price_number),
        high=df[h_c].map(_parse_investing_price_number),
        low=df[l_c].map(_parse_investing_price_number),
        close=df[c_c].map(_parse_investing_price_number),
    )
    if v_c is not None:
        df["volume"] = df[v_c].map(_parse_investing_volume)
    else:
        df["volume"] = None

    df = df.dropna(subset=["date", "open", "high", "low", "close"])
    df = df[df["date"].str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)]
    bad_hilo_mask = df["high"] < df["low"]
    n_bad_hilo = int(bad_hilo_mask.sum())
    if n_bad_hilo > 0:
        df = df.loc[~bad_hilo_mask].copy()
        print(f"[{RAW_PALM_OIL_TABLE}] high<low 이상 행 {n_bad_hilo}개 제거")

    non_positive_mask = (
        (df["open"] <= 0)
        | (df["high"] <= 0)
        | (df["low"] <= 0)
        | (df["close"] <= 0)
    )
    n_non_positive = int(non_positive_mask.sum())
    if n_non_positive > 0:
        df = df.loc[~non_positive_mask].copy()
        print(f"[{RAW_PALM_OIL_TABLE}] OHLC 0/음수 이상 행 {n_non_positive}개 제거")

    out = df[["date", "open", "high", "low", "close", "volume"]].copy()
    if not validate_price_futures_df(out):
        raise ValueError("팜유 CSV 정규화 후 검증 실패 (validate_price_futures_df).")

    ensure_raw_palm_oil_table(conn)
    insert_futures_ohlcv_ignore(conn, RAW_PALM_OIL_TABLE, out, PALM_OIL_CSV_COMMODITY)
    n = len(out)
    dmin, dmax = out["date"].min(), out["date"].max()
    print(
        f"[{RAW_PALM_OIL_TABLE}] CSV 적재 시도 {n}행 | 날짜 범위: {dmin} ~ {dmax} | commodity={PALM_OIL_CSV_COMMODITY}"
    )
    return n


def load_palm_oil_csv(conn: sqlite3.Connection, filepath: str | Path | None = None) -> int:
    """호환용 별칭: Investing CSV 팜유 적재."""
    return load_palm_oil_from_investing_csv(conn, filepath=filepath)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    ingest_all_futures_to_sqlite(print_counts=True)
