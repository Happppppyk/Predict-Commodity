#!/usr/bin/env python3
"""seed.sql.gz → news_v2.db 복원 (멱등).

기존 news_v2.db가 있으면 아무 것도 하지 않는다 (운영 데이터 보호).
"""
from __future__ import annotations

import argparse
import gzip
import sqlite3
import sys
from pathlib import Path


def seed(seed_gz: Path, news_db: Path) -> bool:
    """복원했으면 True, DB가 이미 있어 skip했으면 False."""
    if news_db.exists():
        print(f"[seed_news_db] {news_db} 이미 존재 — skip (운영 DB 보호)")
        return False
    if not seed_gz.exists():
        print(f"[seed_news_db] seed 없음: {seed_gz}", file=sys.stderr)
        sys.exit(1)

    news_db.parent.mkdir(parents=True, exist_ok=True)

    with gzip.open(seed_gz, "rt", encoding="utf-8") as gz:
        sql = gz.read()
    with sqlite3.connect(str(news_db)) as conn:
        conn.executescript(sql)

    return True


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=Path,
                   default=project_root / "data" / "db" / "news_v2_seed.sql.gz")
    p.add_argument("--news-db", type=Path,
                   default=project_root / "data" / "db" / "news_v2.db")
    args = p.parse_args()

    restored = seed(args.seed, args.news_db)
    if restored:
        print(f"[seed_news_db] restored {args.news_db}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
