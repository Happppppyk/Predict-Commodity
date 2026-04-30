"""Build data/news_v2.db (articles + v_articles_scored VIEW).

Usage:
    python scripts/04_build_db.py --append-new
    python scripts/04_build_db.py --enriched data/enriched_v1.parquet \
        --jsonl data/fetched_v1.jsonl --db data/news_v2.db
"""
from __future__ import annotations

import argparse
import datetime as dt
import html
import json
import sqlite3
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[4]


def load_status_index(jsonl_path: Path) -> dict[str, dict]:
    """Read fetched jsonl line-by-line; return dict[url, latest_record].

    Latest line wins on duplicate URLs (resume safety: if 03_fetch was rerun,
    the later attempt overwrites the earlier).

    Raises FileNotFoundError if path doesn't exist.
    """
    if not jsonl_path.exists():
        raise FileNotFoundError(f"jsonl not found: {jsonl_path}")
    idx: dict[str, dict] = {}
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            url = rec.get("url")
            if not url:
                continue
            idx[url] = rec
    return idx


def build_articles_records(
    enriched_df, status_index: dict[str, dict]
):
    """Yield dicts ready for INSERT into articles.

    For each enriched row:
    - All enriched cols → schema columns
    - Lookup status_index by url:
        - if hit: body_status = rec["status"]; body from rec when status=OK
        - else (non-canonical, never attempted): body/body_status = NULL
    """
    for _, row in enriched_df.iterrows():
        url = _none_if_nan(row.get("url"))
        rec = status_index.get(url)
        status = rec.get("status") if rec else None
        is_ok = status == "OK"
        yield {
            "date":              _none_if_nan(row.get("date")),
            "date_int":          _int_or_none(row.get("date_int")),
            "title":             _decode_html_entities(_none_if_nan(row.get("title"))),
            "url":               url,
            "source":            _none_if_nan(row.get("source")),
            "dedup_cluster_id":  _int_or_none(row.get("dedup_cluster_id")),
            "is_canonical":      1 if bool(row.get("is_canonical")) else 0,
            "lang":              _none_if_nan(row.get("lang")),
            "gdelt_themes":      _none_if_nan(row.get("gdelt_themes")),
            "gdelt_locations":   _none_if_nan(row.get("gdelt_locations")),
            "gdelt_organizations": _none_if_nan(row.get("gdelt_organizations")),
            "body":              rec.get("body") if is_ok else None,
            "body_status":       status,
        }


def _decode_html_entities(v):
    """Decode repeatedly to handle double-escaped GDELT page titles."""
    if v is None:
        return None
    text = str(v)
    for _ in range(3):
        decoded = html.unescape(text).strip()
        if decoded == text:
            break
        text = decoded
    return text


def _none_if_nan(v):
    """pd.NA / NaN / None → None; non-NaN values (including empty strings) pass through unchanged."""
    if v is None or v is pd.NA:
        return None
    if isinstance(v, float) and pd.isna(v):
        return None
    return v


def _int_or_none(v):
    if v is None:
        return None
    if pd.isna(v):
        return None
    return int(v)


ARTICLES_SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    date              TEXT NOT NULL,
    date_int          INTEGER NOT NULL,
    title             TEXT NOT NULL,
    url               TEXT NOT NULL,
    source            TEXT,
    dedup_cluster_id  INTEGER NOT NULL,
    is_canonical      INTEGER NOT NULL,
    lang              TEXT,
    gdelt_themes      TEXT,
    gdelt_locations   TEXT,
    gdelt_organizations TEXT,
    body              TEXT,
    body_status       TEXT
);
CREATE INDEX IF NOT EXISTS idx_articles_cluster ON articles(dedup_cluster_id);
CREATE INDEX IF NOT EXISTS idx_articles_url     ON articles(url);
CREATE INDEX IF NOT EXISTS idx_articles_date    ON articles(date);
CREATE INDEX IF NOT EXISTS idx_articles_status  ON articles(body_status);
"""

VIEW_SQL = """
CREATE VIEW IF NOT EXISTS v_articles_scored AS
SELECT
    a.id, a.date, a.date_int, a.title, a.url, a.source,
    a.dedup_cluster_id, a.is_canonical, a.lang,
    a.gdelt_themes, a.gdelt_locations, a.gdelt_organizations,
    a.body, a.body_status,
    s.category, s.is_tradeable, s.skip_reason,
    s.sentiment, s.impact, s.certainty,
    s.weight, s.final_signal,
    s.scoring_input_mode, s.original_content_status,
    s.processed_at
FROM articles a
LEFT JOIN articles canon
    ON canon.dedup_cluster_id = a.dedup_cluster_id AND canon.is_canonical = 1
LEFT JOIN raw_news_scored_v2 s
    ON s.id = canon.id;
"""


def init_db(con: sqlite3.Connection) -> None:
    """DROP + recreate articles + indexes. VIEW created separately by create_view_if_possible."""
    con.execute("DROP VIEW IF EXISTS v_articles_scored")
    con.execute("DROP TABLE IF EXISTS articles")
    con.executescript(ARTICLES_SCHEMA)
    con.commit()


def create_view_if_possible(con: sqlite3.Connection) -> bool:
    """Create v_articles_scored VIEW iff raw_news_scored_v2 exists. Returns True if created."""
    has_scored = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='raw_news_scored_v2'"
    ).fetchone()
    if not has_scored:
        return False
    con.execute("DROP VIEW IF EXISTS v_articles_scored")
    con.executescript(VIEW_SQL)
    con.commit()
    return True


INSERT_SQL = """
INSERT INTO articles
(date, date_int, title, url, source,
 dedup_cluster_id, is_canonical, lang,
 gdelt_themes, gdelt_locations, gdelt_organizations,
 body, body_status)
VALUES (:date, :date_int, :title, :url, :source,
        :dedup_cluster_id, :is_canonical, :lang,
        :gdelt_themes, :gdelt_locations, :gdelt_organizations,
        :body, :body_status)
"""


def insert_batch(con: sqlite3.Connection, records, batch_size: int = 5000) -> int:
    """INSERT records in batches; commit per batch. Returns total inserted."""
    n = 0
    batch: list[dict] = []
    for r in records:
        batch.append(r)
        if len(batch) >= batch_size:
            con.executemany(INSERT_SQL, batch)
            con.commit()
            n += len(batch)
            batch.clear()
    if batch:
        con.executemany(INSERT_SQL, batch)
        con.commit()
        n += len(batch)
    return n


INSERT_IF_NEW_SQL = """
INSERT INTO articles
(date, date_int, title, url, source,
 dedup_cluster_id, is_canonical, lang,
 gdelt_themes, gdelt_locations, gdelt_organizations,
 body, body_status)
SELECT :date, :date_int, :title, :url, :source,
       :dedup_cluster_id, :is_canonical, :lang,
       :gdelt_themes, :gdelt_locations, :gdelt_organizations,
       :body, :body_status
WHERE NOT EXISTS (
    SELECT 1 FROM articles WHERE url = :url
)
"""


def append_new_batch(con: sqlite3.Connection, records, batch_size: int = 5000) -> int:
    """Append records whose URL is not already present, preserving existing IDs."""
    n = 0
    batch: list[dict] = []
    for r in records:
        batch.append(r)
        if len(batch) >= batch_size:
            before = con.total_changes
            con.executemany(INSERT_IF_NEW_SQL, batch)
            con.commit()
            n += con.total_changes - before
            batch.clear()
    if batch:
        before = con.total_changes
        con.executemany(INSERT_IF_NEW_SQL, batch)
        con.commit()
        n += con.total_changes - before
    return n


def validate(con: sqlite3.Connection, log) -> None:
    """Print row counts by status / lang / canonical and date range."""
    log("▶ 검증 쿼리")
    total = con.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
    log(f"  total articles: {total:,}")
    by_status = con.execute(
        "SELECT COALESCE(body_status,'(NULL)') AS s, COUNT(*) "
        "FROM articles GROUP BY 1 ORDER BY 2 DESC"
    ).fetchall()
    log("  body_status 분포:")
    for s, n in by_status:
        log(f"    {s}: {n:,}")
    by_lang = con.execute(
        "SELECT lang, COUNT(*) FROM articles GROUP BY 1 ORDER BY 2 DESC"
    ).fetchall()
    log("  lang 분포:")
    for lng, n in by_lang:
        log(f"    {lng}: {n:,}")
    n_canon = con.execute(
        "SELECT COUNT(*) FROM articles WHERE is_canonical=1"
    ).fetchone()[0]
    log(f"  canonical: {n_canon:,}")
    n_canon_with_status = con.execute(
        "SELECT COUNT(*) FROM articles WHERE is_canonical=1 AND body_status IS NOT NULL"
    ).fetchone()[0]
    log(f"  canonical w/ fetch attempt: {n_canon_with_status:,}")
    dr = con.execute("SELECT MIN(date), MAX(date) FROM articles").fetchone()
    log(f"  date range: {dr}")


def run(
    enriched: Path,
    jsonl: Path,
    db_path: Path,
    rebuild: bool = False,
    append_new: bool = False,
) -> int:
    """Build news_v2.db.

    Default: if articles table already has rows, skip INSERT and only refresh VIEW
    (this is the safe path after scoring_v2 has populated raw_news_scored_v2 — re-INSERT
    would assign new auto-increment IDs and break the canonical → score join).

    Set append_new=True for production incremental runs. This preserves all existing
    article IDs and inserts only URLs not already present, so raw_news_scored_v2.id
    remains valid for historical rows while new rows can be scored with --resume.

    Set rebuild=True to force DROP + re-INSERT. After a rebuild, scoring_v2 must
    also be re-run since raw_news_scored_v2.id no longer matches articles.id.

    Returns 0 on success.
    """
    log_dir = PROJECT / "logs" / "news"
    log_dir.mkdir(exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"04_build_db_{ts}.log"

    if not enriched.exists():
        raise FileNotFoundError(f"enriched parquet not found: {enriched}")
    if not jsonl.exists():
        raise FileNotFoundError(f"jsonl not found: {jsonl}")

    with log_path.open("w") as fh:
        def log(msg: str) -> None:
            line = f"[{dt.datetime.now().isoformat(timespec='seconds')}] {msg}"
            print(line, flush=True)
            fh.write(line + "\n")
            fh.flush()

        log(f"enriched: {enriched}")
        log(f"jsonl:    {jsonl}")
        log(f"db:       {db_path}")
        t0 = time.time()

        log("▶ [1] enriched 로드")
        enr_df = pd.read_parquet(enriched)
        log(f"  enriched rows: {len(enr_df):,}")

        log("▶ [2] status index 로드")
        idx = load_status_index(jsonl)
        log(f"  status records: {len(idx):,}")

        db_path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(str(db_path))
        try:
            # Decide whether to rebuild articles or just refresh VIEW
            articles_exists = con.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='articles'"
            ).fetchone()
            existing_count = 0
            if articles_exists:
                existing_count = con.execute("SELECT COUNT(*) FROM articles").fetchone()[0]

            if existing_count > 0 and not rebuild and not append_new:
                log(f"▶ [skip] articles 이미 {existing_count:,}행 존재 — INSERT 생략 (--append-new로 신규 URL만 추가 가능)")
                inserted = 0
            elif existing_count > 0 and append_new:
                log(f"▶ [append] articles 기존 {existing_count:,}행 보존 — 신규 URL만 INSERT")
                records = build_articles_records(enr_df, idx)
                inserted = append_new_batch(con, records)
                log(f"  inserted_new: {inserted:,}")
            else:
                if existing_count > 0 and rebuild:
                    log(f"▶ [rebuild] articles {existing_count:,}행 DROP 후 재생성")
                    log("  ⚠️  raw_news_scored_v2가 존재하면 ID가 어긋날 수 있음 — 별도로 scoring_v2 재실행 필요")

                log("▶ [3] DB 초기화")
                init_db(con)

                log("▶ [4] articles INSERT (배치 5,000)")
                records = build_articles_records(enr_df, idx)
                inserted = insert_batch(con, records)
                log(f"  inserted: {inserted:,}")

            log("▶ [5] VIEW 생성 시도")
            if create_view_if_possible(con):
                log("  VIEW v_articles_scored 생성 완료")
            else:
                log("  VIEW skip (raw_news_scored_v2 부재 — scoring_v2 첫 실행 후 재실행 필요)")

            validate(con, log)
        finally:
            con.close()

        elapsed = time.time() - t0
        log(f"\n▶ 전체 소요: {elapsed:.1f}s")
    return 0


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build data/news_v2.db (articles + VIEW)")
    ap.add_argument("--enriched", type=Path,
                    default=PROJECT / "data" / "news" / "intermediate" / "enriched_v1.parquet")
    ap.add_argument("--jsonl", type=Path,
                    default=PROJECT / "data" / "news" / "intermediate" / "fetched_v1.jsonl")
    ap.add_argument("--db", type=Path,
                    default=PROJECT / "data" / "db" / "news_v2.db")
    ap.add_argument("--rebuild", action="store_true",
                    help="DROP+recreate articles even if already populated (default: skip INSERT, only refresh VIEW)")
    ap.add_argument("--append-new", action="store_true",
                    help="preserve existing article IDs and insert only URLs not already present")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    return run(
        enriched=args.enriched,
        jsonl=args.jsonl,
        db_path=args.db,
        rebuild=args.rebuild,
        append_new=args.append_new,
    )


if __name__ == "__main__":
    sys.exit(main())
