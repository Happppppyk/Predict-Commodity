"""news_v2.db → soybean.db.raw_news_scored_v2 풀 dump 어댑터.

운영 정책:
- canonical 기사 (articles.is_canonical=1) 만 export.
- DROP + CREATE + INSERT 한 트랜잭션. 실패 시 자동 롤백.
- 본문(articles.body)은 export 안 함. 필요 시 ATTACH로 직접 join.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

DEST_TABLE = "raw_news_scored_v2"

CREATE_DEST_SQL = f"""
CREATE TABLE {DEST_TABLE} (
    id                       INTEGER PRIMARY KEY,
    category                 TEXT,
    is_tradeable             INTEGER,
    skip_reason              TEXT,
    sentiment                INTEGER,
    impact                   INTEGER,
    certainty                INTEGER,
    weight                   INTEGER,
    final_signal             REAL,
    processed_at             TEXT,
    scoring_input_mode       TEXT,
    original_content_status  TEXT,
    date                     TEXT,
    title                    TEXT,
    url                      TEXT,
    source                   TEXT,
    lang                     TEXT,
    gdelt_themes             TEXT,
    gdelt_locations          TEXT,
    gdelt_organizations      TEXT
);
"""

CREATE_INDEX_SQL = [
    f"CREATE INDEX idx_{DEST_TABLE}_date      ON {DEST_TABLE}(date);",
    f"CREATE INDEX idx_{DEST_TABLE}_processed ON {DEST_TABLE}(processed_at);",
]

SELECT_FROM_NEWS_SQL = """
SELECT s.id, s.category, s.is_tradeable, s.skip_reason,
       s.sentiment, s.impact, s.certainty, s.weight, s.final_signal,
       s.processed_at, s.scoring_input_mode, s.original_content_status,
       a.date, a.title, a.url, a.source, a.lang,
       a.gdelt_themes, a.gdelt_locations, a.gdelt_organizations
FROM raw_news_scored_v2 s
JOIN articles a ON a.id = s.id
WHERE a.is_canonical = 1
"""

INSERT_DEST_SQL = (
    f"INSERT INTO {DEST_TABLE} VALUES ("
    + ",".join(["?"] * 20)
    + ")"
)


def export(news_db: Path, main_db: Path) -> int:
    """news_v2.db → main_db.raw_news_scored_v2 풀 dump. 삽입 행 수 반환."""
    src = sqlite3.connect(str(news_db))
    try:
        rows = src.execute(SELECT_FROM_NEWS_SQL).fetchall()
    finally:
        src.close()

    # isolation_level=None disables Python's automatic commit-before-DDL.
    # We then drive the transaction manually so DROP + CREATE + INSERT
    # all roll back together if any statement fails.
    dst = sqlite3.connect(str(main_db), isolation_level=None)
    try:
        dst.execute("BEGIN")
        try:
            dst.execute(f"DROP TABLE IF EXISTS {DEST_TABLE}")
            dst.execute(CREATE_DEST_SQL)
            for stmt in CREATE_INDEX_SQL:
                dst.execute(stmt)
            dst.executemany(INSERT_DEST_SQL, rows)
            dst.execute("COMMIT")
        except Exception:
            dst.execute("ROLLBACK")
            raise
    finally:
        dst.close()

    return len(rows)


def _summary_line(news_db: Path, main_db: Path, inserted: int) -> str:
    with sqlite3.connect(str(main_db)) as conn:
        tradeable = conn.execute(
            f"SELECT COUNT(*) FROM {DEST_TABLE} WHERE is_tradeable=1"
        ).fetchone()[0]
        body_mode = conn.execute(
            f"SELECT COUNT(*) FROM {DEST_TABLE} WHERE scoring_input_mode='body'"
        ).fetchone()[0]
    return (
        f"[export_to_main] {inserted} rows from {news_db.name} → {main_db.name}"
        f" (tradeable={tradeable}, body_mode={body_mode})"
    )


def main() -> int:
    project_root = Path(__file__).resolve().parents[3]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--news-db", type=Path, default=project_root / "data" / "db" / "news_v2.db")
    p.add_argument("--main-db", type=Path, default=project_root / "data" / "db" / "soybean.db")
    args = p.parse_args()

    if not args.news_db.exists():
        print(f"[export_to_main] news_db 없음: {args.news_db}", file=sys.stderr)
        return 1

    inserted = export(args.news_db, args.main_db)
    print(_summary_line(args.news_db, args.main_db, inserted))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
