"""export_to_main: news_v2.db → soybean.db.raw_news_scored_v2 풀 dump 검증."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest


def _make_news_v2(path: Path) -> None:
    """간이 news_v2.db 생성: articles 3행 (canonical 2, member 1) + scored 2행."""
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
        conn.executemany(
            "INSERT INTO articles VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (1, "2026-04-01", 20260401000000, "T1", "https://a/1", "a.com",
                 100, 1, "en", "TH1", "LOC1", "ORG1", "BODY1", "OK"),
                (2, "2026-04-02", 20260402000000, "T2", "https://a/2", "b.com",
                 200, 1, "en", "TH2", "LOC2", "ORG2", None, "PAYWALL"),
                (3, "2026-04-02", 20260402000000, "T3", "https://a/3", "c.com",
                 200, 0, "en", "TH3", "LOC3", "ORG3", None, None),
            ],
        )
        conn.executemany(
            "INSERT INTO raw_news_scored_v2 VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (1, "demand", 1, None, 2, 3, 4, 5, 0.42, "2026-04-03T00:00:00Z", "body", "OK"),
                (2, "supply", 1, None, -1, 2, 3, 4, -0.10, "2026-04-03T00:00:00Z", "title", "PAYWALL"),
            ],
        )


def test_export_creates_table_and_inserts_canonical_only(tmp_path):
    from src.ingestion.news.export_to_main import export

    news_db = tmp_path / "news_v2.db"
    main_db = tmp_path / "soybean.db"
    _make_news_v2(news_db)

    inserted = export(news_db, main_db)

    assert inserted == 2

    with sqlite3.connect(main_db) as conn:
        cols = [r[1] for r in conn.execute("PRAGMA table_info(raw_news_scored_v2)")]
        assert cols == [
            "id", "category", "is_tradeable", "skip_reason",
            "sentiment", "impact", "certainty", "weight", "final_signal",
            "processed_at", "scoring_input_mode", "original_content_status",
            "date", "title", "url", "source", "lang",
            "gdelt_themes", "gdelt_locations", "gdelt_organizations",
        ]
        rows = conn.execute(
            "SELECT id, category, sentiment, final_signal, date, title, gdelt_themes "
            "FROM raw_news_scored_v2 ORDER BY id"
        ).fetchall()
        assert rows == [
            (1, "demand", 2, 0.42, "2026-04-01", "T1", "TH1"),
            (2, "supply", -1, -0.10, "2026-04-02", "T2", "TH2"),
        ]


def test_export_replaces_existing_table(tmp_path):
    from src.ingestion.news.export_to_main import export

    news_db = tmp_path / "news_v2.db"
    main_db = tmp_path / "soybean.db"
    _make_news_v2(news_db)

    export(news_db, main_db)
    # 두 번째 호출도 깨지지 않고 같은 결과
    inserted = export(news_db, main_db)
    assert inserted == 2
    with sqlite3.connect(main_db) as conn:
        n = conn.execute("SELECT COUNT(*) FROM raw_news_scored_v2").fetchone()[0]
        assert n == 2
