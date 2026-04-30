"""Operational v2 pipeline runner for incremental soybean news updates.

This script keeps the historical `articles.id` values stable and only scores
canonical articles that are not already present in `raw_news_scored_v2`.
"""
from __future__ import annotations

import argparse
import datetime as dt
import shlex
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[4]
PYTHON = sys.executable


def _ts() -> str:
    return dt.datetime.now().isoformat(timespec="seconds")


def _run(cmd: list[str], dry_run: bool = False, allow_partial_failure: bool = False) -> int:
    """Run subprocess. allow_partial_failure=True 면 exit != 0 도 WARN 으로만 처리 후 진행.

    00_ingest 가 일부 GDELT 슬롯에서 실패해도 (예: 네트워크 일시 장애로 1900건 중 6건 fail)
    나머지 1881건은 성공적으로 디스크에 저장된 상태라, 후속 stage 가 그 데이터로 진행 가능.
    그래서 00_ingest 한정 partial failure 를 허용한다.
    """
    printable = " ".join(shlex.quote(part) for part in cmd)
    print(f"[{_ts()}] $ {printable}", flush=True)
    if dry_run:
        return 0
    if allow_partial_failure:
        result = subprocess.run(cmd, cwd=PROJECT, check=False)
        if result.returncode != 0:
            print(f"[{_ts()}] WARN exit={result.returncode} from {cmd[1] if len(cmd) > 1 else cmd[0]} — 진행 계속",
                  flush=True)
        return result.returncode
    subprocess.run(cmd, cwd=PROJECT, check=True)
    return 0


def _cleanup_intermediates(dry_run: bool = False) -> None:
    """Delete all pipeline intermediates — articles 가 DB에 영속화된 후엔 보관 의미 없음.

    대상:
      - data/gkg_raw/dt=*/      (00_ingest)
      - data/filtered_v1.parquet (01_filter)
      - data/enriched_v1.parquet (02_language)
      - data/fetched_v1.parquet  (03_fetch — OK 만 추린 스냅샷)
      - data/fetched_v1.jsonl    (03_fetch — in-run resume 용; 04_build_db 후엔 dead)

    재처리 필요 시 00_ingest 로 재다운로드 가능. articles 테이블이 SOT.
    """
    data_dir = PROJECT / "data" / "news"
    targets: list[Path] = []
    freed = 0

    raw_dir = data_dir / "gkg_raw"
    if raw_dir.exists():
        for d in sorted(raw_dir.glob("dt=*")):
            targets.append(d)

    intermediate_dir = data_dir / "intermediate"
    for name in ("filtered_v1.parquet", "enriched_v1.parquet",
                 "fetched_v1.parquet", "fetched_v1.jsonl"):
        f = intermediate_dir / name
        if f.exists():
            targets.append(f)

    if not targets:
        return

    for t in targets:
        if t.is_file():
            freed += t.stat().st_size
        else:
            freed += sum(f.stat().st_size for f in t.rglob("*") if f.is_file())

    print(f"[{_ts()}] cleanup: {len(targets)} items, {freed / 1024**2:.1f} MB",
          flush=True)
    if dry_run:
        return
    for t in targets:
        if t.is_file():
            t.unlink()
        else:
            shutil.rmtree(t)


def _max_article_date(db_path: Path) -> dt.date | None:
    if not db_path.exists():
        return None
    con = sqlite3.connect(str(db_path))
    try:
        has_articles = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='articles'"
        ).fetchone()
        if not has_articles:
            return None
        value = con.execute("SELECT MAX(date) FROM articles").fetchone()[0]
    finally:
        con.close()
    if not value:
        return None
    return dt.date.fromisoformat(str(value)[:10])


def _next_unloaded_window(db_path: Path) -> tuple[str, str] | None:
    last_date = _max_article_date(db_path)
    if last_date is None:
        return None
    start_date = last_date + dt.timedelta(days=1)
    today = dt.datetime.now().date()
    if start_date > today:
        return None
    start = dt.datetime.combine(start_date, dt.time.min)
    end = dt.datetime.now().replace(second=0, microsecond=0)
    return start.isoformat(timespec="minutes"), end.isoformat(timespec="minutes")


def _db_summary(db_path: Path) -> None:
    if not db_path.exists():
        print(f"[{_ts()}] DB summary skipped; missing: {db_path}", flush=True)
        return
    con = sqlite3.connect(str(db_path))
    try:
        has_scored = con.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='raw_news_scored_v2'"
        ).fetchone()
        articles = con.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        canonical = con.execute(
            "SELECT COUNT(*) FROM articles WHERE is_canonical=1"
        ).fetchone()[0]
        if has_scored:
            scored = con.execute("SELECT COUNT(*) FROM raw_news_scored_v2").fetchone()[0]
            unscored = con.execute(
                """
                SELECT COUNT(*)
                FROM articles a
                WHERE a.is_canonical=1
                  AND NOT EXISTS (
                      SELECT 1 FROM raw_news_scored_v2 s WHERE s.id = a.id
                  )
                """
            ).fetchone()[0]
            missing_scores = con.execute(
                """
                SELECT COUNT(*)
                FROM raw_news_scored_v2
                WHERE is_tradeable=1
                  AND (sentiment IS NULL OR impact IS NULL OR certainty IS NULL)
                """
            ).fetchone()[0]
        else:
            scored = 0
            unscored = canonical
            missing_scores = 0
    finally:
        con.close()
    print(
        f"[{_ts()}] DB summary: articles={articles:,} canonical={canonical:,} "
        f"scored={scored:,} unscored_canonical={unscored:,} "
        f"missing_tradeable_scores={missing_scores:,}",
        flush=True,
    )


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the incremental news_v2 pipeline")
    ap.add_argument("--db", type=Path, default=PROJECT / "data" / "db" / "news_v2.db")
    ap.add_argument("--raw-window", default=None, help="optional window for 00_news_ingest.py --last")
    ap.add_argument("--from", dest="from_ts", default=None, help="raw ingest start timestamp")
    ap.add_argument("--to", dest="to_ts", default=None, help="raw ingest end timestamp")
    ap.add_argument("--raw-workers", type=int, default=6)
    ap.add_argument("--fetch-workers", type=int, default=20)
    ap.add_argument("--score-workers", type=int, default=20,
                    help="parallel AI Core calls (default 20; sweet spot — 30+ 는 rate limit 으로 효과 감소)")
    ap.add_argument("--score-progress-every", type=int, default=1000)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--skip-raw", action="store_true")
    ap.add_argument("--skip-filter", action="store_true")
    ap.add_argument("--skip-language", action="store_true")
    ap.add_argument("--skip-fetch", action="store_true")
    ap.add_argument("--skip-score", action="store_true")
    ap.add_argument("--skip-export", action="store_true",
                    help="news_v2.db → soybean.db.raw_news_scored_v2 풀 dump 단계 건너뛰기 (디버깅용)")
    ap.add_argument("--keep-intermediates", action="store_true",
                    help="파이프라인 종료 후 중간 파일 (gkg_raw/, *_v1.parquet, *_v1.jsonl) 자동 삭제 비활성화 (디버깅용)")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    t0 = time.time()

    print(f"[{_ts()}] news_v2 incremental pipeline start", flush=True)

    no_new_raw_window = False

    if not args.skip_raw:
        raw_cmd = [PYTHON, "src/ingestion/news/scripts/00_news_ingest.py", "--workers", str(args.raw_workers)]
        if args.from_ts or args.to_ts:
            if not args.from_ts or not args.to_ts:
                raise SystemExit("--from and --to must be provided together")
            raw_cmd += ["--from", args.from_ts, "--to", args.to_ts]
        elif args.raw_window:
            raw_cmd += ["--last", args.raw_window]
        else:
            window = _next_unloaded_window(args.db)
            if window is None:
                print(
                    f"[{_ts()}] raw ingest skipped: no unloaded calendar date; "
                    "continuing to scoring checks",
                    flush=True,
                )
                no_new_raw_window = True
            else:
                from_ts, to_ts = window
                raw_cmd += ["--from", from_ts, "--to", to_ts]
        if not no_new_raw_window:
            # 00_ingest 는 GDELT 일부 슬롯이 실패해도 나머지는 정상 저장되므로 partial failure 허용
            _run(raw_cmd, dry_run=args.dry_run, allow_partial_failure=True)

    if no_new_raw_window:
        print(
            f"[{_ts()}] data stages skipped: no new raw data to filter/fetch/build",
            flush=True,
        )
    else:
        if not args.skip_filter:
            _run([PYTHON, "src/ingestion/news/scripts/01_news_filter.py"], dry_run=args.dry_run)

        if not args.skip_language:
            _run([PYTHON, "src/ingestion/news/scripts/02_news_language.py"], dry_run=args.dry_run)

        if not args.skip_fetch:
            _run(
                [PYTHON, "src/ingestion/news/scripts/03_news_fetch.py", "--workers", str(args.fetch_workers)],
                dry_run=args.dry_run,
            )

        _run(
            [
                PYTHON,
                "src/ingestion/news/scripts/04_news_build_db.py",
                "--append-new",
                "--db",
                str(args.db),
            ],
            dry_run=args.dry_run,
        )

    if not args.skip_score:
        score_cmd = [
            PYTHON,
            "src/ingestion/news/scripts/05_news_score.py",
            "--db",
            str(args.db),
            "--workers",
            str(args.score_workers),
            "--progress-every",
            str(args.score_progress_every),
        ]
        _run(score_cmd, dry_run=args.dry_run)

        if not no_new_raw_window:
            _run(
                [
                    PYTHON,
                    "src/ingestion/news/scripts/04_news_build_db.py",
                    "--append-new",
                    "--db",
                    str(args.db),
                ],
                dry_run=args.dry_run,
            )

    # scoring 까지 끝난 뒤 — soybean.db.raw_news_scored_v2 로 풀 dump
    if not args.skip_export:
        _run(
            [
                PYTHON, "-m", "src.ingestion.news.export_to_main",
                "--news-db", str(args.db),
                "--main-db", str(PROJECT / "data" / "db" / "soybean.db"),
            ],
            dry_run=args.dry_run,
        )

    # 모든 stage 완료 후 — 중간 파일 일괄 정리 (articles=SOT, 나머지는 dead weight)
    if not args.keep_intermediates:
        _cleanup_intermediates(dry_run=args.dry_run)

    if not args.dry_run:
        _db_summary(args.db)

    elapsed = time.time() - t0
    print(f"[{_ts()}] news_v2 incremental pipeline done in {elapsed:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
