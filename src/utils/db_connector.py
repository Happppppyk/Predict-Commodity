"""
SQLite 연결을 한곳에 모아 둡니다.
SAP HANA Cloud 등으로 전환할 때는 이 모듈만 수정하면 됩니다.
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DB = _ROOT / "data" / "db" / "soybean.db"
DB_PATH = os.environ.get("DB_PATH", str(_DEFAULT_DB))


def get_connection() -> sqlite3.Connection:
    """
    SQLite 연결 반환.
    추후 SAP HANA Cloud 전환 시
    이 함수만 수정하면 전체 적용됨.

    HANA 전환 예시 (2단계):
    from hdbcli import dbapi
    return dbapi.connect(
        address=os.environ["HANA_HOST"],
        port=int(os.environ["HANA_PORT"]),
        user=os.environ["HANA_USER"],
        password=os.environ["HANA_PASSWORD"]
    )
    """
    return sqlite3.connect(DB_PATH)


def get_latest_date(conn: sqlite3.Connection) -> str | None:
    cur = conn.cursor()
    cur.execute("SELECT MAX(date) FROM master_daily")
    row = cur.fetchone()
    return None if row is None else row[0]
