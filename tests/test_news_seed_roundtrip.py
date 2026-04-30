"""dump → gzip → seed 왕복 검증. body 컬럼은 NULL로 비우되 점수/메타는 보존."""
from __future__ import annotations

import gzip
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DUMP_SCRIPT = PROJECT_ROOT / "scripts" / "dump_news_db.py"
SEED_SCRIPT = PROJECT_ROOT / "scripts" / "seed_news_db.py"


def _make_full_news_v2(path: Path) -> None:
    with sqlite3.connect(path) as conn:
        conn.executescript("""
        CREATE TABLE articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            date_int INTEGER NOT NULL,
            title TEXT NOT NULL,
            url TEXT NOT NULL,
            source TEXT,
            dedup_cluster_id INTEGER NOT NULL,
            is_canonical INTEGER NOT NULL,
            lang TEXT,
            gdelt_themes TEXT,
            gdelt_locations TEXT,
            gdelt_organizations TEXT,
            body TEXT,
            body_status TEXT
        );
        CREATE TABLE raw_news_scored_v2 (
            id INTEGER PRIMARY KEY,
            category TEXT,
            is_tradeable INTEGER,
            skip_reason TEXT,
            sentiment INTEGER,
            impact INTEGER,
            certainty INTEGER,
            weight INTEGER,
            final_signal REAL,
            processed_at TEXT,
            scoring_input_mode TEXT,
            original_content_status TEXT
        );
        """)
        conn.execute(
            "INSERT INTO articles VALUES (1,'2026-04-01',20260401000000,'T1','https://a/1','a.com',"
            "100,1,'en','TH','LOC','ORG','VERY_LONG_BODY_TEXT_TO_BE_NULLED','OK')"
        )
        conn.execute(
            "INSERT INTO raw_news_scored_v2 VALUES (1,'demand',1,NULL,2,3,4,5,0.42,"
            "'2026-04-03T00:00:00Z','body','OK')"
        )


def _run(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, cwd=PROJECT_ROOT)


def test_roundtrip_preserves_score_and_nulls_body(tmp_path):
    src_db = tmp_path / "news_v2.db"
    seed_gz = tmp_path / "news_v2_seed.sql.gz"
    restored_db = tmp_path / "restored.db"

    _make_full_news_v2(src_db)

    # dump
    r = _run([sys.executable, str(DUMP_SCRIPT), "--news-db", str(src_db),
             "--out", str(seed_gz)])
    assert r.returncode == 0, r.stderr
    assert seed_gz.exists()
    # body 가 NULL로 비워졌는지 dump 내용에서 직접 확인
    with gzip.open(seed_gz, "rt") as f:
        sql = f.read()
    assert "VERY_LONG_BODY_TEXT_TO_BE_NULLED" not in sql
    # 점수는 살아있어야 함
    assert "0.42" in sql
    assert "demand" in sql

    # seed (restore)
    r = _run([sys.executable, str(SEED_SCRIPT), "--seed", str(seed_gz),
             "--news-db", str(restored_db)])
    assert r.returncode == 0, r.stderr

    with sqlite3.connect(restored_db) as conn:
        n_articles = conn.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        n_scored = conn.execute("SELECT COUNT(*) FROM raw_news_scored_v2").fetchone()[0]
        body, title = conn.execute("SELECT body, title FROM articles WHERE id=1").fetchone()
        score = conn.execute("SELECT final_signal FROM raw_news_scored_v2 WHERE id=1").fetchone()[0]
    assert n_articles == 1
    assert n_scored == 1
    assert body is None
    assert title == "T1"
    assert score == 0.42


def test_seed_idempotent_when_db_already_exists(tmp_path):
    src_db = tmp_path / "news_v2.db"
    seed_gz = tmp_path / "news_v2_seed.sql.gz"
    target_db = tmp_path / "target.db"

    _make_full_news_v2(src_db)
    _run([sys.executable, str(DUMP_SCRIPT), "--news-db", str(src_db),
          "--out", str(seed_gz)])

    # 첫 seed: DB 없음 → 복원
    r1 = _run([sys.executable, str(SEED_SCRIPT), "--seed", str(seed_gz),
              "--news-db", str(target_db)])
    assert r1.returncode == 0, r1.stderr
    assert target_db.exists()

    # 두 번째 seed: DB 이미 있음 → no-op (덮어쓰지 않음)
    target_db.write_bytes(target_db.read_bytes() + b"")  # 크기/내용 그대로
    pre = target_db.read_bytes()
    r2 = _run([sys.executable, str(SEED_SCRIPT), "--seed", str(seed_gz),
              "--news-db", str(target_db)])
    assert r2.returncode == 0, r2.stderr
    post = target_db.read_bytes()
    assert pre == post, "기존 DB가 있으면 seed가 덮어쓰지 말아야 함"
