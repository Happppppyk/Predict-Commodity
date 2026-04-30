"""Body fetch + drop gates (paywall/404/short/lang) for canonical clusters.

httpx async + trafilatura. Output: data/fetched_v1.parquet (status=OK only).
JSONL fsync resume pattern (data/fetched_v1.jsonl) for crash safety.
"""
import argparse
import asyncio
import datetime as dt
import json
import os
import sys
import threading
import time
from pathlib import Path

import httpx
import trafilatura
from langdetect import DetectorFactory, LangDetectException, detect

DetectorFactory.seed = 0  # 결정론

ALLOWED_LANGS = {"en", "pt", "es"}
MIN_BODY_CHARS = 400
SCRIPT_ARTIFACT_MARKERS = (
    "createElement",
    "querySelector",
    "document.",
    "window.",
    "setAttribute",
)


def _extract_body(html: str | None) -> str | None:
    """Extract main article body from HTML. Returns None if no body found."""
    if not html:
        return None
    return trafilatura.extract(html)


def _detect_lang(body: str) -> str | None:
    """Return ISO 639-1 lang code or None on failure."""
    try:
        return detect(body)
    except LangDetectException:
        return None


def _looks_like_script_artifact(body: str) -> bool:
    """Return True when extraction captured JavaScript instead of article text."""
    prefix = body[:300]
    if prefix.startswith('";var ') or prefix.startswith("';var "):
        return True
    marker_hits = sum(1 for marker in SCRIPT_ARTIFACT_MARKERS if marker in prefix)
    return marker_hits >= 2


def _validate(body: str, enriched_lang: str) -> tuple[bool, str]:
    """Return (ok, status) where status is OK or a body quality status.

    Script artifact and length gates fire first (cheap), then langdetect strict
    comparison.
    """
    if _looks_like_script_artifact(body):
        return False, "SCRIPT_ARTIFACT"
    if len(body) < MIN_BODY_CHARS:
        return False, "SHORT"
    detected = _detect_lang(body)
    if detected not in ALLOWED_LANGS or detected != enriched_lang:
        return False, "LANG_MISMATCH"
    return True, "OK"


HTTP_TIMEOUT_SEC = 15.0
TIMEOUT_RETRY_ONCE = 1


async def _fetch_one(url: str, client: httpx.AsyncClient) -> tuple[str, str | None]:
    """GET URL with status mapping policy.

    Returns:
        (status, html_or_None)
        - 200 → ("OK", html)
        - 401/403 → ("PAYWALL", None)
        - 404/410 → ("NOT_FOUND", None)
        - 그 외 4xx/5xx → ("HTTP_ERROR", None)
        - Timeout → 1회 retry 후 ("TIMEOUT", None)
        - 네트워크 오류 → ("FETCH_FAILED", None)
    """
    last_exc: Exception | None = None
    for attempt in range(TIMEOUT_RETRY_ONCE + 1):
        try:
            resp = await client.get(url, timeout=HTTP_TIMEOUT_SEC, follow_redirects=True)
            code = resp.status_code
            if code == 200:
                return "OK", resp.text
            if code in (401, 403):
                return "PAYWALL", None
            if code in (404, 410):
                return "NOT_FOUND", None
            return "HTTP_ERROR", None
        except httpx.TimeoutException as e:
            last_exc = e
            continue  # retry once
        except (httpx.NetworkError, httpx.RequestError) as e:
            last_exc = e
            return "FETCH_FAILED", None
    return "TIMEOUT", None


async def _call_one(
    cluster_id: int,
    url: str,
    enriched_lang: str,
    client: httpx.AsyncClient,
) -> dict:
    """Fetch URL, extract body, validate. Return record dict (never raises).

    Record always has: cluster_id, url, enriched_lang, status.
    On status=OK additionally: body, body_len, lang_recheck.
    Drop reasons (PAYWALL/NOT_FOUND/HTTP_ERROR/TIMEOUT/FETCH_FAILED/SHORT/LANG_MISMATCH)
    are reflected in `status`. trafilatura가 본문 추출 못한 경우 SHORT로 분류
    (length gate가 None을 SHORT로 본다는 의미).
    """
    fetch_status, html = await _fetch_one(url, client)
    if fetch_status != "OK":
        return {
            "cluster_id": cluster_id,
            "url": url,
            "enriched_lang": enriched_lang,
            "status": fetch_status,
        }
    body = _extract_body(html)
    if not body:
        return {
            "cluster_id": cluster_id,
            "url": url,
            "enriched_lang": enriched_lang,
            "status": "SHORT",
        }
    ok, val_status = _validate(body, enriched_lang)
    if not ok:
        return {
            "cluster_id": cluster_id,
            "url": url,
            "enriched_lang": enriched_lang,
            "status": val_status,
        }
    return {
        "cluster_id": cluster_id,
        "url": url,
        "enriched_lang": enriched_lang,
        "status": "OK",
        "body": body,
        "body_len": len(body),
        "lang_recheck": _detect_lang(body),
    }


def _load_processed_clusters(jsonl_path: Path) -> set[int]:
    """Load processed cluster_ids from JSONL (any status counts as processed)."""
    if not jsonl_path.exists():
        return set()
    ids: set[int] = set()
    with jsonl_path.open("r", encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"WARN {jsonl_path.name}:{lineno} 깨진 라인 skip: {e}",
                      file=sys.stderr)
                continue
            cid = rec.get("cluster_id")
            if isinstance(cid, int):
                ids.add(cid)
    return ids


def _append_jsonl(record: dict, path: Path, lock: threading.Lock) -> None:
    """Append one JSON line with flush + fsync (crash-safe)."""
    line = json.dumps(record, ensure_ascii=False)
    with lock:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()
            os.fsync(f.fileno())


def _jsonl_to_parquet(jsonl_path: Path, parquet_path: Path) -> None:
    """Convert JSONL to parquet, keeping only status=OK rows.

    Schema: [dedup_cluster_id (int64), url (str), body (str),
             body_len (int64), lang_recheck (str)]
    """
    import pandas as pd

    cluster_ids: list[int] = []
    urls: list[str] = []
    bodies: list[str] = []
    body_lens: list[int] = []
    langs: list[str] = []

    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") != "OK":
                continue
            cid = rec.get("cluster_id")
            if not isinstance(cid, int):
                continue
            cluster_ids.append(cid)
            urls.append(str(rec.get("url", "")))
            bodies.append(str(rec.get("body", "")))
            body_lens.append(int(rec.get("body_len", 0)))
            langs.append(str(rec.get("lang_recheck", "")))

    df = pd.DataFrame({
        "dedup_cluster_id": cluster_ids,
        "url": urls,
        "body": bodies,
        "body_len": body_lens,
        "lang_recheck": langs,
    })
    if df.empty:
        df = df.astype({
            "dedup_cluster_id": "int64",
            "url": "str",
            "body": "str",
            "body_len": "int64",
            "lang_recheck": "str",
        })
    parquet_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = parquet_path.with_suffix(parquet_path.suffix + ".tmp")
    try:
        df.to_parquet(tmp, index=False, compression="snappy")
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, parquet_path)


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ENRICHED = PROJECT_ROOT / "data" / "news" / "intermediate" / "enriched_v1.parquet"
DEFAULT_JSONL = PROJECT_ROOT / "data" / "news" / "intermediate" / "fetched_v1.jsonl"
DEFAULT_PARQUET = PROJECT_ROOT / "data" / "news" / "intermediate" / "fetched_v1.parquet"
DEFAULT_WORKERS = 20

USER_AGENT = "soybean-news/0.1 (+https://example.com/bot)"


def _ts_now() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _load_pending(
    enriched_path: Path, processed_ids: set[int], limit: int | None,
) -> list[tuple[int, str, str]]:
    """Filter canonical rows; exclude already-processed; apply limit.

    Returns list of (cluster_id, url, enriched_lang).
    """
    import pandas as pd

    df = pd.read_parquet(enriched_path)
    df = df[df["is_canonical"]]
    pending: list[tuple[int, str, str]] = []
    for _, row in df.iterrows():
        cid = int(row["dedup_cluster_id"])
        if cid in processed_ids:
            continue
        url = str(row["url"])
        lang = str(row.get("lang") or "")
        pending.append((cid, url, lang))
        if limit is not None and len(pending) >= limit:
            break
    return pending


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="03_fetch.py",
        description="Body fetch + drop gates for canonical clusters.",
    )
    p.add_argument("--enriched", type=Path, default=DEFAULT_ENRICHED)
    p.add_argument("--out-jsonl", type=Path, default=DEFAULT_JSONL)
    p.add_argument("--out-parquet", type=Path, default=DEFAULT_PARQUET)
    p.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args(argv)
    if args.workers < 1 or args.workers > 50:
        p.error("--workers must be 1..50")
    if args.limit is not None and args.limit <= 0:
        p.error("--limit must be positive")
    if not args.enriched.exists():
        p.error(f"enriched parquet missing: {args.enriched}")
    return args


async def main_async(argv: list[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    args.out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.out_jsonl.touch(exist_ok=True)

    logs_dir = PROJECT_ROOT / "logs" / "news"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = _ts_now()
    log_path = logs_dir / f"03_fetch_{ts}.log"
    json_path = logs_dir / f"03_fetch_{ts}.json"

    processed = set() if args.no_resume else _load_processed_clusters(args.out_jsonl)
    pending = _load_pending(args.enriched, processed, args.limit)

    counts = {
        "OK": 0, "PAYWALL": 0, "NOT_FOUND": 0, "TIMEOUT": 0,
        "HTTP_ERROR": 0, "SHORT": 0, "LANG_MISMATCH": 0, "FETCH_FAILED": 0,
    }
    t0 = time.time()
    lock = threading.Lock()
    sem = asyncio.Semaphore(args.workers)

    with log_path.open("w") as fh:
        def log(msg: str) -> None:
            line = f"[{dt.datetime.now().isoformat(timespec='seconds')}] {msg}"
            print(line)
            fh.write(line + "\n")
            fh.flush()

        log(
            f"enriched={args.enriched} pending={len(pending)} "
            f"workers={args.workers} resume={'no' if args.no_resume else 'yes'} "
            f"already_processed={len(processed)}"
        )

        async with httpx.AsyncClient(
            timeout=HTTP_TIMEOUT_SEC,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        ) as client:
            async def worker(cid: int, url: str, lang: str) -> None:
                async with sem:
                    rec = await _call_one(cid, url, lang, client)
                    _append_jsonl(rec, args.out_jsonl, lock)
                    counts[rec["status"]] += 1
                    done = sum(counts.values())
                    if done % 100 == 0:
                        log(f"  진행: {done}/{len(pending)}  counts={counts}")

            await asyncio.gather(*[worker(cid, url, lang) for cid, url, lang in pending])

        elapsed = round(time.time() - t0, 1)
        log(f"DONE pending={len(pending)} {counts} elapsed={elapsed}s")

        if args.out_jsonl.exists():
            log(f"  JSONL → parquet: {args.out_parquet}")
            _jsonl_to_parquet(args.out_jsonl, args.out_parquet)

    summary = {
        "timestamp": ts,
        "args": {
            "enriched": str(args.enriched),
            "out_jsonl": str(args.out_jsonl),
            "out_parquet": str(args.out_parquet),
            "workers": args.workers,
            "limit": args.limit,
            "no_resume": args.no_resume,
        },
        "already_processed": len(processed),
        "called": len(pending),
        "ok": counts["OK"],
        "paywall": counts["PAYWALL"],
        "not_found": counts["NOT_FOUND"],
        "timeout": counts["TIMEOUT"],
        "http_error": counts["HTTP_ERROR"],
        "short": counts["SHORT"],
        "lang_mismatch": counts["LANG_MISMATCH"],
        "fetch_failed": counts["FETCH_FAILED"],
        "elapsed_sec": elapsed,
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    return asyncio.run(main_async(argv))


if __name__ == "__main__":
    sys.exit(main())
