"""rescore_v2.py — SAP AI Core 기반 뉴스 scoring v2.

목적:
    articles 테이블의 canonical 기사를 읽어 raw_news_scored_v2 에 저장한다.
    본문이 있으면 body, 없으면 title fallback 으로 placeholder 를 채워
    Launchpad 에 등록된 orchestration template (`news-sentiment-classifier`) 을 호출한다.

호출 구조 (per-item):
    1. _news_to_placeholder_values(news) → {id, title, content}
    2. aicore_client.run(placeholder_values) → JSON 문자열
    3. domain.Category / Score / ScoringResult 로 검증 및 aggregate 빌드
    4. weight_map.get_weight(category) 로 weight 결정
    5. ScoringResult.to_db_row() → DB INSERT

운영:
    python scoring_v2/rescore_v2.py --db data/news_v2.db \
      --input-table articles --include-no-body --resume \
      --workers 20

    smoke:
    python scoring_v2/rescore_v2.py --db data/news_v2.db --limit 5 --workers 20
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from src.ingestion.news.scoring_v2.domain import Category, Score, ScoringResult
from src.ingestion.news.scoring_v2.weight_map import get_weight
from src.ingestion.news.scoring_v2.aicore_client import run as _aicore_run, AICoreConfigError

# .env 자동 로드 (있으면)
try:
    from dotenv import load_dotenv  # type: ignore[import-not-found]
    load_dotenv()
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("rescore_v2")


# ---------------------------------------------------------------------------
# DB schema
# ---------------------------------------------------------------------------
V2_SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_news_scored_v2 (
    id                   INTEGER PRIMARY KEY,
    category             TEXT,
    is_tradeable         INTEGER,         -- 0/1
    skip_reason          TEXT,
    sentiment            INTEGER,         -- -2..+2
    impact               INTEGER,         -- 0 if not tradeable; otherwise 1..5
    certainty            INTEGER,         -- 0 if not tradeable; otherwise 1..5
    weight               INTEGER,         -- 0..5 (from category lookup)
    final_signal         REAL,            -- sentiment * (impact/5) * (certainty/5) * (weight/5)
    processed_at         TEXT,
    scoring_input_mode   TEXT,            -- body | title
    original_content_status TEXT          -- articles.body_status at scoring time
);
CREATE INDEX IF NOT EXISTS idx_v2_category ON raw_news_scored_v2(category);
CREATE INDEX IF NOT EXISTS idx_v2_signal   ON raw_news_scored_v2(final_signal);
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


# ---------------------------------------------------------------------------
# Input → placeholder
# ---------------------------------------------------------------------------
def _truncate_content(
    body: str,
    head: int = 700,
    tail: int = 150,
    max_total: int = 900,
) -> str:
    if len(body) <= max_total:
        return body
    head_part = body[:head].rsplit(" ", 1)[0] if " " in body[:head] else body[:head]
    tail_part = body[-tail:].split(" ", 1)[-1] if " " in body[-tail:] else body[-tail:]
    return f"{head_part} ... {tail_part}"


def _uses_body_input(news: dict) -> bool:
    body = str(news.get("body") or "").strip()
    return bool(body) and news.get("body_status") == "OK"


def _scoring_content(news: dict) -> str:
    content = str(news.get("body") or "").strip()
    if _uses_body_input(news):
        return content
    return str(news.get("title") or "").strip()


def _scoring_meta(news: dict) -> dict[str, Any]:
    has_body = _uses_body_input(news)
    return {
        "scoring_input_mode": "body" if has_body else "title",
        "original_content_status": news.get("body_status"),
    }


def _news_to_placeholder_values(news: dict) -> dict[str, str]:
    """Launchpad template 의 {{?id}} {{?title}} {{?content}} 자리를 채운다.

    source 는 분류·점수에 부수 정보라 템플릿에서 제거 (publisher 도메인이라
    1차 자료 식별엔 부정확). certainty 는 본문 키워드(USDA/EPA/CONAB 등) 기준.
    """
    return {
        "id": str(news.get("id", "")),
        "title": str(news.get("title", "")),
        "content": _truncate_content(_scoring_content(news)),
    }


# ---------------------------------------------------------------------------
# Output parsing
# ---------------------------------------------------------------------------
_FENCE_RE = re.compile(r"^```(?:json)?\s*\n?|\n?```\s*$", re.IGNORECASE)


def _strip_fence(text: str) -> str:
    return _FENCE_RE.sub("", text.strip()).strip()


def _parse_json(content: str) -> dict | None:
    if not content:
        return None
    try:
        obj = json.loads(_strip_fence(content))
    except (TypeError, json.JSONDecodeError):
        log.warning("JSON parse failed: %r", content[:200])
        return None
    return obj if isinstance(obj, dict) else None


def _classify(obj: dict | None) -> tuple[Category, bool, str | None]:
    """LLM 응답 → (category, is_tradeable, skip_reason).

    실패 케이스는 모두 OTHER + not-tradeable + 진단용 skip_reason 으로 fallback:
      - obj is None         → skip_reason="no_response"        (retry 소진 / 4xx)
      - non-dict             → skip_reason="non_dict_response"  (모델 비정상)
      - 알 수 없는 category  → skip_reason="invalid_category=<x>"
    category=OTHER 면 LLM 이 is_tradeable=true 라 해도 false 로 강제.
    """
    if obj is None:
        return Category.OTHER, False, "no_response"
    if not isinstance(obj, dict):
        return Category.OTHER, False, "non_dict_response"
    raw_cat = obj.get("category")
    category = Category.parse(raw_cat)
    if category is None:
        return Category.OTHER, False, f"invalid_category={raw_cat}"
    is_tradeable = bool(obj.get("is_tradeable", False))
    if category == Category.OTHER:
        is_tradeable = False
    return category, is_tradeable, obj.get("skip_reason")


# ---------------------------------------------------------------------------
# LLM call
# ---------------------------------------------------------------------------
def _mock_response(news: dict) -> dict:
    """dry-run 용 — source 키워드로 적당한 카테고리 흉내."""
    src = str(news.get("source") or "").lower()
    if "usda" in src:
        category = "snd_us"
    elif "conab" in src:
        category = "snd_br"
    else:
        category = "market_general"
    return {
        "id": str(news.get("id", "")),
        "category": category,
        "is_tradeable": True,
        "skip_reason": None,
        "sentiment": 0,
        "impact": 2,
        "certainty": 3,
    }


def _call_aicore_one(news: dict) -> dict | None:
    """AI Core 1회 호출 + JSON 파싱. retry 다 소진/파싱 실패 → None."""
    pv = _news_to_placeholder_values(news)
    text = _aicore_run(pv)
    if text is None:
        return None
    return _parse_json(text)


# ---------------------------------------------------------------------------
# Per-row processing
# ---------------------------------------------------------------------------
def process_one(news: dict, dry_run: bool = False) -> dict:
    """Score one row via AI Core orchestration (or mock when dry_run=True).

    Returns DB row dict (ScoringResult.to_db_row()) for direct upsert.
    """
    item = _mock_response(news) if dry_run else _call_aicore_one(news)

    category, is_tradeable, skip_reason = _classify(item)
    weight = get_weight(category)
    meta = _scoring_meta(news)

    # not-tradeable / weight=0 → 점수 0 강제. tradeable 인데 점수 파싱 실패 → score=None (NULL).
    score: Score | None
    if not is_tradeable or weight == 0:
        score = Score.neutral()
    else:
        score = Score.from_llm(item)
        if score is None:
            log.warning("Score parse failed id=%s, scores remain NULL", news["id"])

    result = ScoringResult(
        article_id=news["id"],
        category=category,
        is_tradeable=is_tradeable,
        skip_reason=skip_reason,
        score=score,
        weight=weight,
        final_signal=score.to_signal(weight) if score else 0.0,
        scoring_input_mode=meta["scoring_input_mode"],
        original_content_status=meta["original_content_status"],
        processed_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
    )
    return result.to_db_row()


# ---------------------------------------------------------------------------
# DB I/O
# ---------------------------------------------------------------------------
def init_db(con: sqlite3.Connection) -> None:
    # 워커 20개가 동시에 INSERT OR REPLACE 하면 default rollback journal 모드는
    # writer-lock 경합으로 'database is locked' 가 드물게 뜬다. WAL 모드로 전환하면
    # writer 가 다른 reader/writer 를 블락하지 않고, fsync 비용도 NORMAL 로 낮춤.
    # busy_timeout 은 락 경합 시 30 초 대기 (운영 안정성).
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA busy_timeout=30000")
    con.executescript(V2_SCHEMA)
    _ensure_v2_columns(con)
    _refresh_articles_scored_view(con)
    con.commit()


def _ensure_v2_columns(con: sqlite3.Connection) -> None:
    """Migrate older raw_news_scored_v2 tables in-place."""
    cur = con.execute("PRAGMA table_info(raw_news_scored_v2)")
    existing = {row[1] for row in cur.fetchall()}
    columns = {
        "scoring_input_mode": "TEXT",
        "original_content_status": "TEXT",
    }
    for name, col_type in columns.items():
        if name not in existing:
            con.execute(f"ALTER TABLE raw_news_scored_v2 ADD COLUMN {name} {col_type}")


def _refresh_articles_scored_view(con: sqlite3.Connection) -> None:
    has_articles = con.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='articles'"
    ).fetchone()
    if not has_articles:
        return
    con.execute("DROP VIEW IF EXISTS v_articles_scored")
    con.executescript(VIEW_SQL)


_ALLOWED_TABLES = {"articles"}


def fetch_inputs(
    con: sqlite3.Connection,
    limit: int | None,
    table: str = "articles",
    include_no_body: bool = False,
) -> list[dict]:
    """Fetch scoring candidates from `table` (whitelisted).

    Default filter: body_status='OK' (only articles with extracted body).

    With include_no_body=True (articles table only): widens to all canonical rows
    (is_canonical=1), so PAYWALL/TIMEOUT/etc. rows are also scored. process_one()
    falls back to title when body is NULL/empty.
    """
    if table not in _ALLOWED_TABLES:
        raise ValueError(f"input table not allowed: {table!r} (allowed: {sorted(_ALLOWED_TABLES)})")
    if include_no_body and table != "articles":
        raise ValueError("--include-no-body requires --input-table articles")
    where = "WHERE is_canonical=1" if include_no_body else "WHERE body_status='OK'"
    sql = f"SELECT id, date, title, url, source, body, body_status FROM {table} {where}"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    cur = con.execute(sql)
    cols = [c[0] for c in cur.description]
    return [dict(zip(cols, r)) for r in cur.fetchall()]


def already_processed(con: sqlite3.Connection, repair_missing_scores: bool = False) -> set[int]:
    where = ""
    if repair_missing_scores:
        where = """
        WHERE NOT (
            is_tradeable = 1
            AND (sentiment IS NULL OR impact IS NULL OR certainty IS NULL)
        )
        """
    cur = con.execute(f"SELECT id FROM raw_news_scored_v2 {where}")
    return {r[0] for r in cur.fetchall()}


def missing_score_ids(con: sqlite3.Connection) -> set[int]:
    cur = con.execute("""
        SELECT id
        FROM raw_news_scored_v2
        WHERE is_tradeable = 1
          AND (sentiment IS NULL OR impact IS NULL OR certainty IS NULL)
    """)
    return {r[0] for r in cur.fetchall()}


def upsert_record(con: sqlite3.Connection, rec: dict, lock: threading.Lock) -> None:
    cols = list(rec.keys())
    placeholders = ",".join(["?"] * len(cols))
    sql = f"INSERT OR REPLACE INTO raw_news_scored_v2 ({','.join(cols)}) VALUES ({placeholders})"
    with lock:
        con.execute(sql, [rec[c] for c in cols])
        con.commit()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True, help="path to news_v2.db")
    ap.add_argument("--limit", type=int, default=None, help="limit rows (smoke test)")
    ap.add_argument("--workers", type=int, default=20,
                    help="parallel AI Core calls (default 20; sweet spot — 30 이상은 rate limit 으로 효과 감소)")
    ap.add_argument("--dry-run", action="store_true", help="mock LLM, no AI Core call")
    ap.add_argument("--resume", action="store_true", help="skip ids already in v2 table")
    ap.add_argument(
        "--progress-every",
        type=int,
        default=1000,
        help="log progress every N processed rows (default: 1000)",
    )
    ap.add_argument(
        "--input-table",
        choices=sorted(_ALLOWED_TABLES),
        default="articles",
        help="source table to read scoring inputs from (default: articles)",
    )
    ap.add_argument(
        "--include-no-body",
        action="store_true",
        help="(articles only) score all canonical rows including PAYWALL/TIMEOUT/etc. — falls back to title when body is NULL",
    )
    ap.add_argument(
        "--repair-missing-scores",
        action="store_true",
        help="with --resume, reprocess rows where is_tradeable=1 but sentiment/impact/certainty is NULL",
    )
    ap.add_argument(
        "--only-missing-scores",
        action="store_true",
        help="process only rows already present in v2 with missing tradeable score fields",
    )
    args = ap.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        log.error("DB not found: %s", db_path)
        return 1

    con = sqlite3.connect(str(db_path), check_same_thread=False)
    try:
        return _run(con, args)
    finally:
        con.close()


def _run(con: sqlite3.Connection, args: argparse.Namespace) -> int:
    """Main scoring loop — extracted from main() so con.close() can sit in a finally."""
    init_db(con)

    fetch_limit = None if args.resume else args.limit
    rows = fetch_inputs(
        con,
        fetch_limit,
        table=args.input_table,
        include_no_body=args.include_no_body,
    )
    log.info("Total candidates: %d", len(rows))

    if args.only_missing_scores:
        repair_ids = missing_score_ids(con)
        rows = [r for r in rows if r["id"] in repair_ids]
        log.info("Missing-score repair mode: %d rows", len(rows))

    if args.resume:
        done = already_processed(con, repair_missing_scores=args.repair_missing_scores)
        rows = [r for r in rows if r["id"] not in done]
        if args.limit is not None:
            rows = rows[:args.limit]
        log.info("Resume mode: %d remaining (skipped %d)", len(rows), len(done))

    lock = threading.Lock()
    stats_lock = threading.Lock()
    abort_event = threading.Event()
    n_ok = n_skip = n_fail = 0
    processed = 0
    next_progress = args.progress_every

    def _progress_log() -> None:
        nonlocal next_progress
        while args.progress_every > 0 and processed >= next_progress:
            log.info(
                "Progress: processed=%d/%d tradeable=%d skipped=%d failed=%d",
                processed, len(rows), n_ok, n_skip, n_fail,
            )
            next_progress += args.progress_every

    def _update_stats(rec: dict) -> None:
        nonlocal n_ok, n_skip, processed
        with stats_lock:
            if rec["is_tradeable"]:
                n_ok += 1
            else:
                n_skip += 1
            processed += 1
            _progress_log()

    def _record_fail() -> None:
        nonlocal n_fail, processed
        with stats_lock:
            n_fail += 1
            processed += 1
            _progress_log()

    def _worker(news: dict) -> None:
        if abort_event.is_set():
            return
        try:
            rec = process_one(news, dry_run=args.dry_run)
            upsert_record(con, rec, lock)
            _update_stats(rec)
        except AICoreConfigError as e:
            log.error("AI Core config/auth failure — aborting batch: %s", e)
            abort_event.set()
            _record_fail()
        except Exception as e:  # noqa: BLE001
            log.error("Row id=%s failed: %s", news.get("id"), e)
            _record_fail()

    if args.workers <= 1:
        for r in rows:
            if abort_event.is_set():
                break
            _worker(r)
    else:
        ex = ThreadPoolExecutor(max_workers=args.workers)
        try:
            futures = [ex.submit(_worker, r) for r in rows]
            for f in as_completed(futures):
                _ = f.result()
                # AICoreConfigError 발생 시 워커가 abort_event.set() 한 직후
                # 남은 future 들을 즉시 취소해서 대기 시간 절감.
                # (이미 실행 중인 워커는 인터럽트 못 하지만, 큐의 워커는 시작 단계의
                # abort_event 체크에서 빠르게 빠져나옴.)
                if abort_event.is_set():
                    for pending in futures:
                        pending.cancel()
                    break
        finally:
            ex.shutdown(wait=True, cancel_futures=True)

    log.info("Done. tradeable=%d skipped=%d failed=%d aborted=%s",
             n_ok, n_skip, n_fail, abort_event.is_set())
    return 1 if abort_event.is_set() else 0


if __name__ == "__main__":
    raise SystemExit(main())
