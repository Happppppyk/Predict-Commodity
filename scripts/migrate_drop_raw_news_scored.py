#!/usr/bin/env python3
"""1회용 마이그레이션 — soybean.db.raw_news_scored 테이블 DROP.

배경:
    뉴스 v2 파이프라인은 raw_news_scored_v2를 정본으로 사용한다.
    기존 단순 키워드 scorer가 채우던 raw_news_scored 테이블은 더 이상 갱신되지 않으므로 제거.

특성:
    멱등 — 테이블이 이미 없으면 아무 것도 하지 않고 성공으로 종료.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

LEGACY_TABLE = "raw_news_scored"


def drop_legacy(db_path: Path) -> bool:
    """DROP 했으면 True, skip 했으면 False."""
    if not db_path.exists():
        print(f"[migrate] DB 없음: {db_path} — skip")
        return False

    with sqlite3.connect(str(db_path)) as conn:
        exists = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
            (LEGACY_TABLE,),
        ).fetchone()
        if exists is None:
            print(f"[migrate] {LEGACY_TABLE} 테이블 없음 — skip (멱등)")
            return False
        n = conn.execute(f"SELECT COUNT(*) FROM {LEGACY_TABLE}").fetchone()[0]
        print(f"[migrate] {LEGACY_TABLE} 행 {n}개 — DROP 진행")
        conn.execute(f"DROP TABLE {LEGACY_TABLE}")
        conn.commit()
    print(f"[migrate] DROP 완료: {LEGACY_TABLE}")
    return True


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--db", type=Path,
                   default=project_root / "data" / "db" / "soybean.db")
    args = p.parse_args()
    drop_legacy(args.db)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
