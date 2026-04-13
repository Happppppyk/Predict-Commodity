"""
역할: CFTC Commitments of Traders 레거시(Non-commercial) 포지션 중 대두유(SOYBEAN OIL) 행을 수집해 `raw_cftc`에 적재한다.

데이터 소스 (CFTC 공식):
- `https://www.cftc.gov/dea/newcot/deacot.zip` 시도 (구 URL, 404일 수 있음).
- 연도별 레거시 선물 ZIP `https://www.cftc.gov/files/dea/history/deacot{연도}.zip` (Historical Compressed).
- 최신 스냅샷 `https://www.cftc.gov/dea/newcot/deafut.txt` (주간 1회 갱신, 이력 ZIP과 병합·중복 제거).

⚠️ Look-ahead Bias 핵심 주의사항 (master_daily·모델링 시 반드시 준수):
절대로 report_date 기준으로 forward-fill 하지 말 것.
반드시 release_date(금요일) 기준으로만 master_daily에 조인할 것.
화요일~목요일에 그 주 값을 쓰면 미래 정보 유출임.
"""

from __future__ import annotations

import csv
import io
import sqlite3
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

# CFTC 공식: 요청 URL(구 newcot 번들, 현재 404 가능)
DEACOT_ZIP_URL = "https://www.cftc.gov/dea/newcot/deacot.zip"
# Historical Compressed — 레거시 Futures-Only, 연도별 ZIP(내부 annual.txt 등)
DEACOT_HISTORY_YEAR_ZIP = "https://www.cftc.gov/files/dea/history/deacot{year}.zip"
# 최신 주간 스냅샷 (필드 정의 동일: cotvariableslegacy.html)
DEAFUT_TXT_URL = "https://www.cftc.gov/dea/newcot/deafut.txt"

MIN_REPORT_DATE_DEFAULT = "2010-01-01"
HISTORY_START_YEAR = 2010

DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "db" / "soybean.db"
RAW_CFTC_TABLE = "raw_cftc"

# 레거시 콤마 파일: 필드 3=report YYYY-MM-DD, 9=Noncomm Long, 10=Noncomm Short (1-based) → 인덱스 2,8,9
_IDX_REPORT_DATE = 2
_IDX_NONCOMM_LONG = 8
_IDX_NONCOMM_SHORT = 9
_MIN_FIELDS = 10

COMMODITY_SUBSTR = "SOYBEAN OIL"

CREATE_RAW_CFTC_SQL = f"""
CREATE TABLE IF NOT EXISTS {RAW_CFTC_TABLE} (
    report_date TEXT NOT NULL,
    release_date TEXT PRIMARY KEY,
    noncomm_long REAL,
    noncomm_short REAL,
    noncomm_net REAL
);
"""

INSERT_CFTC_IGNORE_SQL = f"""
INSERT OR IGNORE INTO {RAW_CFTC_TABLE}
    (report_date, release_date, noncomm_long, noncomm_short, noncomm_net)
VALUES (?, ?, ?, ?, ?);
"""

USER_AGENT = "Mozilla/5.0 (compatible; soybean-oil-poc/1.0; +research)"


def _http_get(url: str, timeout: int = 120) -> bytes:
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _is_zip_payload(data: bytes) -> bool:
    return len(data) >= 4 and data[:2] == b"PK"


def _text_from_deacot_zip(data: bytes) -> str | None:
    """ZIP 내부 .txt(annual.txt 등)를 이어붙인다. 실패 시 None."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            parts: list[str] = []
            for name in zf.namelist():
                if name.lower().endswith(".txt"):
                    parts.append(zf.read(name).decode("utf-8", errors="replace"))
            return "\n".join(parts) if parts else None
    except (zipfile.BadZipFile, OSError, ValueError):
        return None


def _collect_cftc_legacy_csv_blobs() -> list[str]:
    """
    레거리 선물-only 콤마 구분 텍스트 조각을 모은다 (순서는 파싱 후 release_date로 정렬).

    newcot ZIP → 연도별 history ZIP(2019~올해) → deafut.txt.
    """
    blobs: list[str] = []

    try:
        raw = _http_get(DEACOT_ZIP_URL)
        if _is_zip_payload(raw):
            t = _text_from_deacot_zip(raw)
            if t:
                blobs.append(t)
    except (HTTPError, URLError, OSError, TimeoutError):
        pass

    y_end = date.today().year
    for year in range(HISTORY_START_YEAR, y_end + 1):
        url = DEACOT_HISTORY_YEAR_ZIP.format(year=year)
        try:
            raw_y = _http_get(url)
            if not _is_zip_payload(raw_y):
                continue
            t = _text_from_deacot_zip(raw_y)
            if t:
                blobs.append(t)
        except (HTTPError, URLError, OSError, TimeoutError):
            continue

    try:
        raw_txt = _http_get(DEAFUT_TXT_URL)
        blobs.append(raw_txt.decode("utf-8", errors="replace"))
    except (HTTPError, URLError, OSError, TimeoutError):
        pass

    return blobs


def _parse_cot_legacy_csv_text(text: str) -> pd.DataFrame:
    """
    레거시 deafut/zip 내 txt 한 덩어리를 파싱한다.

    - 시장명에 'SOYBEAN OIL' 포함 행만 (대소문자 무시).
    - report_date: 필드 3 (YYYY-MM-DD, 화요일 기준 보고일).
    - release_date: report_date + 3일 (동주 금요일 공표 가정).
    """
    rows: list[dict[str, object]] = []
    reader = csv.reader(io.StringIO(text))
    for rec in reader:
        if len(rec) < _MIN_FIELDS:
            continue
        market = rec[0].strip().strip('"')
        if COMMODITY_SUBSTR.upper() not in market.upper():
            continue
        report_raw = rec[_IDX_REPORT_DATE].strip()
        try:
            report_d = datetime.strptime(report_raw, "%Y-%m-%d").date()
        except ValueError:
            continue
        release_d = report_d + timedelta(days=3)
        nl = _parse_cftc_number(rec[_IDX_NONCOMM_LONG])
        ns = _parse_cftc_number(rec[_IDX_NONCOMM_SHORT])
        if nl is None or ns is None:
            continue
        net = float(nl) - float(ns)
        rows.append(
            {
                "report_date": report_d.isoformat(),
                "release_date": release_d.isoformat(),
                "noncomm_long": float(nl),
                "noncomm_short": float(ns),
                "noncomm_net": net,
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "report_date",
                "release_date",
                "noncomm_long",
                "noncomm_short",
                "noncomm_net",
            ]
        )
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset=["release_date"], keep="last")
    df = df.sort_values("release_date").reset_index(drop=True)
    return df


def _parse_cftc_number(s: str) -> float | None:
    t = str(s).strip()
    if t in (".", "", "NaN"):
        return None
    try:
        return float(t)
    except ValueError:
        return None


def download_cftc_from_cftc_gov(min_report_date: str = MIN_REPORT_DATE_DEFAULT) -> pd.DataFrame:
    """
    CFTC 공식 소스에서 레거리 선물-only CSV 텍스트를 모아 파싱한다.

    - `deacot.zip`(newcot) 시도 후, `deacot{연도}.zip`(2010~당해) 및 `deafut.txt` 병합.
    - 대두유: 시장명에 \"SOYBEAN OIL\" 포함 행만.
    - report_date(화요일), release_date(금요일=report+3일),
      noncomm_long, noncomm_short, noncomm_net(long-short).
    - `min_report_date` 이전 report_date 행은 제외 (기본 2010-01-01).

    Returns
    -------
    pd.DataFrame
        위 컬럼, release_date 기준 정렬.
    """
    blobs = _collect_cftc_legacy_csv_blobs()
    if not blobs:
        raise RuntimeError(
            "CFTC 레거시 데이터를 받지 못했습니다 (연도 ZIP·deafut 모두 실패)."
        )
    merged = "\n".join(blobs)
    df = _parse_cot_legacy_csv_text(merged)
    if df.empty:
        return df
    df = df[df["report_date"] >= min_report_date].reset_index(drop=True)
    return df


def ensure_raw_cftc_table(conn: sqlite3.Connection) -> None:
    """`raw_cftc` 테이블 생성."""
    conn.execute(CREATE_RAW_CFTC_SQL)
    conn.commit()


def load_cftc(conn: sqlite3.Connection) -> int:
    """
    CFTC 데이터를 내려받아 `raw_cftc`에 INSERT OR IGNORE로 적재한다.

    ⚠️ Look-ahead Bias 핵심 주의사항:
    절대로 report_date 기준으로 forward-fill 하지 말 것.
    반드시 release_date(금요일) 기준으로만 master_daily에 조인할 것.
    화요일~목요일에 그 주 값을 쓰면 미래 정보 유출임.

    적재 후 최신 5행을 출력한다.

    Returns
    -------
    int
        INSERT 시도 행 수(IGNORE로 건너뛴 행 포함 가능).
    """
    df = download_cftc_from_cftc_gov()
    ensure_raw_cftc_table(conn)
    n = 0
    if not df.empty:
        batch = list(
            zip(
                df["report_date"].astype(str),
                df["release_date"].astype(str),
                df["noncomm_long"].astype(float),
                df["noncomm_short"].astype(float),
                df["noncomm_net"].astype(float),
            )
        )
        conn.executemany(INSERT_CFTC_IGNORE_SQL, batch)
        conn.commit()
        n = len(batch)

    cur = conn.execute(
        f"""
        SELECT report_date, release_date, noncomm_long, noncomm_short, noncomm_net
        FROM {RAW_CFTC_TABLE}
        ORDER BY release_date DESC
        LIMIT 5
        """
    )
    print("[raw_cftc] 최신 5행 (release_date 내림차순):")
    for row in cur.fetchall():
        print(row)
    total_rows = int(conn.execute(f"SELECT COUNT(*) FROM {RAW_CFTC_TABLE}").fetchone()[0])
    dmin, dmax = conn.execute(
        f"SELECT MIN(report_date), MAX(report_date) FROM {RAW_CFTC_TABLE}"
    ).fetchone()
    print(f"[raw_cftc] 전체 행 수: {total_rows} | report_date 범위: {dmin} ~ {dmax}")
    return n


def load_cftc_to_sqlite(db_path: str | Path | None = None) -> int:
    """DB 경로를 열어 `load_cftc` 실행 (편의 함수)."""
    path = Path(db_path) if db_path is not None else DEFAULT_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        return load_cftc(conn)


if __name__ == "__main__":
    load_cftc_to_sqlite()
