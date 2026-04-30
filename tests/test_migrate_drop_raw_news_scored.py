"""raw_news_scored DROP 마이그레이션 — 멱등 검증."""
from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT_ROOT / "scripts" / "migrate_drop_raw_news_scored.py"


def _run(db_path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--db", str(db_path)],
        capture_output=True, text=True, cwd=PROJECT_ROOT,
    )


def test_drops_existing_table(tmp_path):
    db = tmp_path / "soybean.db"
    with sqlite3.connect(db) as conn:
        conn.executescript("""
        CREATE TABLE raw_news_scored (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT, sentiment_score REAL, final_score REAL
        );
        INSERT INTO raw_news_scored(url, sentiment_score, final_score)
        VALUES ('u1', 0.5, 0.5), ('u2', -0.3, -0.3);
        """)

    r = _run(db)
    assert r.returncode == 0, r.stderr
    assert "DROP" in r.stdout or "drop" in r.stdout.lower()

    with sqlite3.connect(db) as conn:
        rows = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='raw_news_scored'"
        ).fetchall()
    assert rows == []


def test_idempotent_when_table_missing(tmp_path):
    db = tmp_path / "soybean.db"
    db.touch()  # empty DB
    r1 = _run(db)
    assert r1.returncode == 0, r1.stderr
    # 두 번째 호출도 성공해야 함
    r2 = _run(db)
    assert r2.returncode == 0, r2.stderr
    assert "skip" in r2.stdout.lower() or "없음" in r2.stdout


def test_does_not_touch_other_tables(tmp_path):
    db = tmp_path / "soybean.db"
    with sqlite3.connect(db) as conn:
        conn.executescript("""
        CREATE TABLE raw_news_scored (id INTEGER PRIMARY KEY);
        CREATE TABLE raw_other (id INTEGER PRIMARY KEY, val INTEGER);
        INSERT INTO raw_other VALUES (1, 100), (2, 200);
        """)

    r = _run(db)
    assert r.returncode == 0, r.stderr

    with sqlite3.connect(db) as conn:
        n = conn.execute("SELECT COUNT(*) FROM raw_other").fetchone()[0]
    assert n == 2
