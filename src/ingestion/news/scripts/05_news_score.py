"""Score unprocessed canonical articles in news_v2.db with the SAP AI Core backend."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


PROJECT = Path(__file__).resolve().parents[4]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run AI Core scoring for news_v2.db")
    ap.add_argument("--db", type=Path, default=PROJECT / "data" / "db" / "news_v2.db")
    ap.add_argument("--workers", type=int, default=20,
                    help="parallel AI Core calls (default 20; sweet spot, 30 이상은 rate limit 으로 효과 감소)")
    ap.add_argument("--progress-every", type=int, default=1000)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only-missing-scores", action="store_true")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    cmd = [
        sys.executable,
        str(PROJECT / "src" / "ingestion" / "news" / "scoring_v2" / "rescore_v2.py"),
        "--db",
        str(args.db),
        "--input-table",
        "articles",
        "--include-no-body",
        "--resume",
        "--workers",
        str(args.workers),
        "--progress-every",
        str(args.progress_every),
    ]
    if args.limit is not None:
        cmd += ["--limit", str(args.limit)]
    if args.dry_run:
        cmd += ["--dry-run"]
    if args.only_missing_scores:
        cmd += ["--only-missing-scores"]
    return subprocess.run(cmd, cwd=PROJECT).returncode


if __name__ == "__main__":
    raise SystemExit(main())
