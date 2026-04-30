#!/usr/bin/env python3
"""news_v2.db → body 제외 SQL dump (gzip).

LLM 점수와 메타는 보존, articles.body 컬럼은 NULL로 비움.
"""
from __future__ import annotations

import argparse
import gzip
import sqlite3
import sys
from pathlib import Path


def dump(news_db: Path, out_path: Path) -> int:
    """news_db에서 body 제외 dump를 gzip으로 저장. 출력 파일 바이트 수 반환."""
    if not news_db.exists():
        print(f"[dump_news_db] news_db 없음: {news_db}", file=sys.stderr)
        return -1

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # SQLite의 iterdump()를 쓰되, articles 테이블 INSERT만 가로채서 body를 NULL로 치환.
    # 가장 단순한 방법: dump 전에 임시 attached DB로 복사 → body 컬럼 UPDATE NULL → iterdump.
    with gzip.open(out_path, "wt", encoding="utf-8") as gz:
        with sqlite3.connect(":memory:") as mem:
            mem.execute("ATTACH DATABASE ? AS src", (str(news_db),))
            # 스키마와 데이터를 메모리로 복사
            for row in mem.execute(
                "SELECT name, sql FROM src.sqlite_master "
                "WHERE type IN ('table','index','view') AND name NOT LIKE 'sqlite_%'"
                " ORDER BY CASE type WHEN 'table' THEN 1 WHEN 'index' THEN 2 ELSE 3 END"
            ).fetchall():
                if row[1]:
                    mem.execute(row[1])
            for tbl in mem.execute(
                "SELECT name FROM src.sqlite_master WHERE type='table'"
                " AND name NOT LIKE 'sqlite_%'"
            ).fetchall():
                t = tbl[0]
                mem.execute(f"INSERT INTO {t} SELECT * FROM src.{t}")
            mem.commit()
            mem.execute("DETACH DATABASE src")
            # body 컬럼을 NULL로
            mem.execute("UPDATE articles SET body = NULL")
            mem.commit()
            for line in mem.iterdump():
                gz.write(line + "\n")

    return out_path.stat().st_size


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--news-db", type=Path,
                   default=project_root / "data" / "db" / "news_v2.db")
    p.add_argument("--out", type=Path,
                   default=project_root / "data" / "db" / "news_v2_seed.sql.gz")
    args = p.parse_args()

    n = dump(args.news_db, args.out)
    if n < 0:
        return 1
    print(f"[dump_news_db] wrote {args.out} ({n:,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
