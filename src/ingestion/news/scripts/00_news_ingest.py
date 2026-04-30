"""GDELT GKG 2.0 15-minute batch file ingester."""
import argparse
import datetime as dt
import html
import io
import json
import os
import re
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT_ROOT / "src" / "ingestion" / "news"))
from config import GDELT_BASE_URL, GDELT_LASTUPDATE_URL  # noqa: E402

DEFAULT_OUT_ROOT_RAW = PROJECT_ROOT / "data" / "news" / "gkg_raw"
DEFAULT_OUT_ROOT_FILTERED = PROJECT_ROOT / "data" / "news" / "intermediate" / "filtered_v1"

SLOT_MINUTES = 15


def _floor_to_grid(t: dt.datetime) -> dt.datetime:
    """Floor a datetime down to the previous 15-minute boundary."""
    minute = (t.minute // SLOT_MINUTES) * SLOT_MINUTES
    return t.replace(minute=minute, second=0, microsecond=0)


def _ceil_to_grid(t: dt.datetime) -> dt.datetime:
    """Ceil a datetime up to the next 15-minute boundary.

    A datetime is considered "on grid" iff `minute % 15 == 0` and
    `second == 0` and `microsecond == 0`. Anything else is bumped up
    to the next 15-minute boundary (with seconds/microseconds zeroed).
    """
    if t.minute % SLOT_MINUTES == 0 and t.second == 0 and t.microsecond == 0:
        return t.replace(second=0, microsecond=0)
    return _floor_to_grid(t) + dt.timedelta(minutes=SLOT_MINUTES)


def slot_iter(start: dt.datetime, end: dt.datetime) -> Iterator[dt.datetime]:
    """Yield 15-minute grid slots covering [start, end].

    `start` is floored to the previous grid point and `end` is ceiled to
    the next, so the returned slot list is the smallest set of GKG batch
    slots that fully covers the requested time range. A range that lies
    entirely between two grid points yields BOTH endpoints (e.g.,
    00:01–00:14 → [00:00, 00:15]).

    Yields nothing if `start > end`.
    """
    if start > end:
        return
    cur = _floor_to_grid(start)
    last = _ceil_to_grid(end)
    while cur <= last:
        yield cur
        cur += dt.timedelta(minutes=SLOT_MINUTES)


def slot_url(slot: dt.datetime) -> str:
    """Return the GDELT GKG download URL for a 15-min slot."""
    return f"{GDELT_BASE_URL}/{slot.strftime('%Y%m%d%H%M%S')}.gkg.csv.zip"


def slot_path(slot: dt.datetime, root: Path) -> Path:
    """Return the output parquet path under a Hive-style date partition."""
    return root / f"dt={slot.strftime('%Y-%m-%d')}" / f"{slot.strftime('%H%M%S')}.parquet"


_PAGE_TITLE_RE = re.compile(r"<PAGE_TITLE>(.*?)</PAGE_TITLE>", re.DOTALL)


def extract_page_title(extras: str | None) -> str:
    """Pull <PAGE_TITLE>...</PAGE_TITLE> from the GKG Extras XML blob.

    Returns empty string if the tag is absent or the input is falsy.
    HTML entities are decoded repeatedly because GDELT titles are sometimes
    double-escaped (e.g. "&amp;#xE1;" -> "&#xE1;" -> "á").
    """
    if not extras:
        return ""
    m = _PAGE_TITLE_RE.search(extras)
    if not m:
        return ""
    title = m.group(1).strip()
    for _ in range(3):
        decoded = html.unescape(title).strip()
        if decoded == title:
            break
        title = decoded
    return title


GKG_NUM_COLUMNS = 27
_IDX_DATE = 1
_IDX_SOURCE = 3
_IDX_DOCID = 4
_IDX_V2THEMES = 8
_IDX_V2LOCS = 10
_IDX_V2ORGS = 14
_IDX_V2TONE = 15
_IDX_EXTRAS = 26

_OUTPUT_COLUMNS = [
    "DATE", "DocumentIdentifier", "SourceCommonName",
    "V2Themes", "V2Tone", "V2Organizations", "V2Locations", "title",
]


def parse_gkg_csv(zip_bytes: bytes) -> tuple[pd.DataFrame, dict[str, int]]:
    """Decode a GKG csv.zip payload into the 8-column slim DataFrame.

    Returns (df, drops) where drops counts malformed rows by reason:
      - short_row: tab-split fields < 27 columns
      - date_parse_fail: DATE column not parseable as int
    """
    drops = {"short_row": 0, "date_parse_fail": 0}
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        members = [n for n in zf.namelist() if not n.endswith("/")]
        if not members:
            return pd.DataFrame(columns=_OUTPUT_COLUMNS).astype({"DATE": "int64"}), drops
        with zf.open(members[0]) as fp:
            text = io.TextIOWrapper(fp, encoding="utf-8", errors="replace")
            dates: list[int] = []
            doc_ids: list[str] = []
            sources: list[str] = []
            v2themes: list[str] = []
            v2tones: list[str] = []
            v2orgs: list[str] = []
            v2locs: list[str] = []
            titles: list[str] = []
            for line in text:
                line = line.rstrip("\n").rstrip("\r")
                if not line:
                    continue
                fields = line.split("\t")
                if len(fields) < GKG_NUM_COLUMNS:
                    drops["short_row"] += 1
                    continue
                try:
                    dates.append(int(fields[_IDX_DATE]))
                except ValueError:
                    drops["date_parse_fail"] += 1
                    continue
                doc_ids.append(fields[_IDX_DOCID])
                sources.append(fields[_IDX_SOURCE])
                v2themes.append(fields[_IDX_V2THEMES])
                v2tones.append(fields[_IDX_V2TONE])
                v2orgs.append(fields[_IDX_V2ORGS])
                v2locs.append(fields[_IDX_V2LOCS])
                titles.append(extract_page_title(fields[_IDX_EXTRAS]))
    df = pd.DataFrame({
        "DATE": dates,
        "DocumentIdentifier": doc_ids,
        "SourceCommonName": sources,
        "V2Themes": v2themes,
        "V2Tone": v2tones,
        "V2Organizations": v2orgs,
        "V2Locations": v2locs,
        "title": titles,
    })
    return df, drops


HTTP_TIMEOUT_SEC = 30
MAX_RETRIES = 3
BACKOFF_BASE = 1.0  # 1s, 2s, 4s


def download_zip(url: str) -> bytes | None:
    """GET the URL with bounded retries.

    Returns the response body on success. Returns None on a 404 (treated
    as an expected "missing slot" condition for GKG batches). Raises the
    last URLError/HTTPError if retries are exhausted on transient failures.
    """
    last_err: Exception | None = None
    for attempt in range(MAX_RETRIES):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "soybean-news/0.1"})
            with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SEC) as resp:
                return resp.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            last_err = e
        except urllib.error.URLError as e:
            last_err = e
        if attempt < MAX_RETRIES - 1:
            time.sleep(BACKOFF_BASE * (2 ** attempt))
    assert last_err is not None
    raise last_err


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    """Write a DataFrame to parquet atomically.

    Writes to `<path>.tmp` first, then `os.replace` to the target path.
    The replace is atomic on the same filesystem, so a crash mid-write
    cannot leave a corrupt file at the target. If the write itself
    fails, the orphan `.tmp` is removed before re-raising.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    try:
        df.to_parquet(tmp, index=False, compression="snappy")
    except Exception:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)


@dataclass(frozen=True)
class IngestOpts:
    """Run-wide options threaded through process_slot / process_day."""
    out_root: Path
    force: bool = False
    filter_on_parse: bool = False
    min_disk_pct: float = 1.0
    min_disk_gb: float = 2.0
    workers: int = 6


@dataclass(frozen=True)
class SlotResult:
    """Per-slot outcome for logging and aggregation."""
    slot: dt.datetime
    status: str  # OK | SKIPPED | MISSING | FAILED
    rows: int = 0
    elapsed_ms: int = 0
    error: str | None = None
    drops_short_row: int = 0
    drops_date_fail: int = 0


def process_slot(slot: dt.datetime, opts: IngestOpts) -> SlotResult:
    """Download → parse → write a single slot. Never raises; returns a SlotResult."""
    t0 = time.time()
    out = slot_path(slot, opts.out_root)
    if out.exists() and not opts.force:
        return SlotResult(slot, "SKIPPED", 0, int((time.time() - t0) * 1000))
    url = slot_url(slot)
    try:
        payload = download_zip(url)
    except Exception as e:  # noqa: BLE001 — by design: never raise
        return SlotResult(slot, "FAILED", 0, int((time.time() - t0) * 1000), str(e))
    if payload is None:
        return SlotResult(slot, "MISSING", 0, int((time.time() - t0) * 1000))
    try:
        df, drops = parse_gkg_csv(payload)
        write_parquet(df, out)
    except Exception as e:  # noqa: BLE001
        return SlotResult(slot, "FAILED", 0, int((time.time() - t0) * 1000), str(e))
    return SlotResult(
        slot, "OK", len(df), int((time.time() - t0) * 1000),
        drops_short_row=drops["short_row"],
        drops_date_fail=drops["date_parse_fail"],
    )


# ---------------------------------------------------------------------------
# Filter-at-download mode: per-day batched filtering using soybean_filter_sql
# ---------------------------------------------------------------------------

_FILTER_SQL_CACHE: str | None = None


def _get_filter_sql() -> str:
    """Lazy-load soybean_filter_sql() once (cached for reuse across slots)."""
    global _FILTER_SQL_CACHE
    if _FILTER_SQL_CACHE is None:
        from config import soybean_filter_sql  # local import to defer config load
        _FILTER_SQL_CACHE = soybean_filter_sql()
    return _FILTER_SQL_CACHE


def apply_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Apply soybean_filter_sql to an in-memory DataFrame using DuckDB.

    Empty DataFrame returns unchanged. Single in-memory DuckDB connection
    per call; thread-safe by virtue of fresh connection per invocation.
    """
    if df.empty:
        return df
    import duckdb  # local import — only needed in filter mode
    con = duckdb.connect(":memory:")
    try:
        con.register("df_view", df)
        sql = _get_filter_sql()
        return con.execute(f"SELECT * FROM df_view WHERE {sql}").df()
    finally:
        con.close()


def free_disk_pct(path: Path) -> float:
    """Return free disk space % at `path` (or its nearest existing ancestor)."""
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    usage = shutil.disk_usage(p)
    return (usage.free / usage.total) * 100


def free_disk_gb(path: Path) -> float:
    """Return free disk space in GB at `path` (or its nearest existing ancestor)."""
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    usage = shutil.disk_usage(p)
    return usage.free / (1024 ** 3)


def disk_guard_check(path: Path, min_pct: float, min_gb: float) -> tuple[bool, str]:
    """Return (ok, message). ok=False if either pct or gb is under threshold."""
    pct = free_disk_pct(path)
    gb = free_disk_gb(path)
    if pct < min_pct:
        return False, f"disk free {pct:.2f}% < {min_pct}% (gb={gb:.2f})"
    if gb < min_gb:
        return False, f"disk free {gb:.2f} GB < {min_gb} GB (pct={pct:.2f})"
    return True, f"disk free {gb:.2f} GB ({pct:.2f}%)"


def day_path(day: str, root: Path) -> Path:
    """Per-day output path: <root>/dt=YYYY-MM-DD/day.parquet"""
    return root / f"dt={day}" / "day.parquet"


def _group_slots_by_day(slots: list[dt.datetime]) -> dict[str, list[dt.datetime]]:
    """Group slots by their YYYY-MM-DD date string."""
    by_day: dict[str, list[dt.datetime]] = defaultdict(list)
    for s in slots:
        by_day[s.strftime("%Y-%m-%d")].append(s)
    return dict(sorted(by_day.items()))


@dataclass(frozen=True)
class SlotResultMem:
    """Per-slot outcome WITHOUT writing — df held in memory for day batch."""
    slot: dt.datetime
    status: str  # OK | MISSING | FAILED
    df: pd.DataFrame | None = None
    elapsed_ms: int = 0
    error: str | None = None
    drops_short_row: int = 0
    drops_date_fail: int = 0


def process_slot_inmemory(slot: dt.datetime) -> SlotResultMem:
    """Download + parse only. No write. Returns df in result for batch use."""
    t0 = time.time()
    url = slot_url(slot)
    try:
        payload = download_zip(url)
    except Exception as e:  # noqa: BLE001
        return SlotResultMem(
            slot, "FAILED",
            elapsed_ms=int((time.time() - t0) * 1000),
            error=str(e),
        )
    if payload is None:
        return SlotResultMem(slot, "MISSING", elapsed_ms=int((time.time() - t0) * 1000))
    try:
        df, drops = parse_gkg_csv(payload)
    except Exception as e:  # noqa: BLE001
        return SlotResultMem(
            slot, "FAILED",
            elapsed_ms=int((time.time() - t0) * 1000),
            error=str(e),
        )
    return SlotResultMem(
        slot, "OK", df=df,
        elapsed_ms=int((time.time() - t0) * 1000),
        drops_short_row=drops["short_row"],
        drops_date_fail=drops["date_parse_fail"],
    )


@dataclass(frozen=True)
class DayResult:
    """Per-day batch outcome."""
    day: str
    status: str  # OK | SKIPPED | DISK_LOW | FAILED
    slots_total: int = 0
    slots_ok: int = 0
    slots_missing: int = 0
    slots_failed: int = 0
    raw_rows: int = 0
    filtered_rows: int = 0
    elapsed_sec: float = 0.0
    drops_short_row: int = 0
    drops_date_fail: int = 0
    error: str | None = None


def process_day(day: str, slots: list[dt.datetime], opts: IngestOpts) -> DayResult:
    """Download all slots of a day in parallel, optionally filter, write 1 day.parquet."""
    t0 = time.time()
    out = day_path(day, opts.out_root)
    if out.exists() and not opts.force:
        return DayResult(day=day, status="SKIPPED", slots_total=len(slots))

    raw_dfs: list[pd.DataFrame] = []
    counts = {"OK": 0, "MISSING": 0, "FAILED": 0}
    drops = {"short_row": 0, "date_parse_fail": 0}

    with ThreadPoolExecutor(max_workers=opts.workers) as pool:
        futures = {pool.submit(process_slot_inmemory, s): s for s in slots}
        for fut in as_completed(futures):
            res = fut.result()
            counts[res.status] = counts.get(res.status, 0) + 1
            drops["short_row"] += res.drops_short_row
            drops["date_parse_fail"] += res.drops_date_fail
            if res.status == "OK" and res.df is not None and not res.df.empty:
                raw_dfs.append(res.df)

    raw_count = sum(len(df) for df in raw_dfs)

    if not raw_dfs:
        empty_df = pd.DataFrame(columns=_OUTPUT_COLUMNS).astype({"DATE": "int64"})
        write_parquet(empty_df, out)
        return DayResult(
            day=day, status="OK", slots_total=len(slots),
            slots_ok=counts["OK"], slots_missing=counts["MISSING"],
            slots_failed=counts["FAILED"], raw_rows=0, filtered_rows=0,
            elapsed_sec=round(time.time() - t0, 1),
            drops_short_row=drops["short_row"],
            drops_date_fail=drops["date_parse_fail"],
        )

    combined = pd.concat(raw_dfs, ignore_index=True)
    if opts.filter_on_parse:
        try:
            filtered = apply_filter(combined)
        except Exception as e:  # noqa: BLE001
            return DayResult(
                day=day, status="FAILED", slots_total=len(slots),
                slots_ok=counts["OK"], slots_missing=counts["MISSING"],
                slots_failed=counts["FAILED"], raw_rows=raw_count,
                elapsed_sec=round(time.time() - t0, 1),
                error=f"filter error: {e}",
            )
    else:
        filtered = combined

    write_parquet(filtered, out)

    return DayResult(
        day=day, status="OK", slots_total=len(slots),
        slots_ok=counts["OK"], slots_missing=counts["MISSING"],
        slots_failed=counts["FAILED"], raw_rows=raw_count,
        filtered_rows=len(filtered),
        elapsed_sec=round(time.time() - t0, 1),
        drops_short_row=drops["short_row"],
        drops_date_fail=drops["date_parse_fail"],
    )


_LASTUPDATE_GKG_RE = re.compile(r"(\d{14})\.gkg\.csv\.zip\r?$", re.MULTILINE)


def resolve_latest() -> dt.datetime:
    """Hit GDELT's lastupdate.txt and return the latest GKG slot timestamp."""
    req = urllib.request.Request(
        GDELT_LASTUPDATE_URL, headers={"User-Agent": "soybean-news/0.1"}
    )
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_SEC) as resp:
        body = resp.read().decode("utf-8", errors="replace")
    m = _LASTUPDATE_GKG_RE.search(body)
    if not m:
        raise RuntimeError(f"No gkg line found in lastupdate.txt:\n{body}")
    s = m.group(1)
    return dt.datetime(
        int(s[0:4]), int(s[4:6]), int(s[6:8]),
        int(s[8:10]), int(s[10:12]), int(s[12:14]),
    )


MIN_DATE = dt.datetime(2020, 1, 1, 0, 0)


@dataclass
class CliArgs:
    """Parsed and validated CLI arguments for 00_ingest.py."""
    start: dt.datetime | None
    end: dt.datetime | None
    last_seconds: int | None
    latest: bool
    workers: int
    force: bool
    out_root: Path | None
    filter_on_parse: bool = False
    min_disk_pct: float = 1.0
    min_disk_gb: float = 2.0


_LAST_RE = re.compile(r"^(\d+)([mhd])$")


def _parse_last(s: str) -> int:
    """Convert a duration like '30m', '2h', '1d' into seconds."""
    m = _LAST_RE.match(s)
    if not m:
        raise argparse.ArgumentTypeError(
            f"--last must look like '30m', '2h', or '1d'; got {s!r}"
        )
    n, unit = int(m.group(1)), m.group(2)
    return n * {"m": 60, "h": 3600, "d": 86400}[unit]


def parse_args(argv: list[str] | None = None) -> CliArgs:
    """Parse CLI arguments into a validated CliArgs.

    Exits via argparse on unknown args, mutual-exclusion violations,
    pre-2020 dates, missing --to with --from, or malformed --last.
    """
    p = argparse.ArgumentParser(
        prog="00_ingest.py",
        description="Download GDELT GKG 2.0 15-minute batch files into Hive-partitioned parquet.",
    )
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--from", dest="start", type=dt.datetime.fromisoformat,
        help="ISO start datetime (e.g., 2026-04-27T00:00). Floored to 15-min grid.",
    )
    mode.add_argument(
        "--last", type=_parse_last,
        help="Recent window like '30m', '2h', '1d'. Anchored to now.",
    )
    mode.add_argument(
        "--latest", action="store_true",
        help="Download only the latest slot from lastupdate.txt.",
    )
    p.add_argument(
        "--to", dest="end", type=dt.datetime.fromisoformat,
        help="ISO end datetime; required with --from. Ceiled to 15-min grid.",
    )
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--force", action="store_true")
    p.add_argument("--out", dest="out_root", type=Path, default=None,
                   help="Output root. Default: data/gkg_raw/ in raw mode, "
                        "or data/filtered_v1/ with --filter-on-parse.")
    p.add_argument("--filter-on-parse", dest="filter_on_parse",
                   action="store_true", default=False,
                   help="Download slots, apply soybean filter immediately, and persist "
                        "per-day filtered parquet under data/filtered_v1/. The main "
                        "pipeline does not consume this mode directly.")
    p.add_argument("--no-filter-on-parse", dest="filter_on_parse",
                   action="store_false",
                   help="Compatibility no-op: raw per-slot parquet mode is the default.")
    p.add_argument("--min-disk-pct", type=float, default=1.0,
                   help="Abort if free disk %% drops below this (default 1.0).")
    p.add_argument("--min-disk-gb", type=float, default=2.0,
                   help="Abort if free disk GB drops below this (default 2.0). "
                        "Disk guard fires when EITHER %% or GB goes under threshold.")
    ns = p.parse_args(argv)

    if ns.start is not None and ns.end is None:
        p.error("--from requires --to")
    if ns.start is None and ns.end is not None:
        p.error("--to is only valid together with --from")
    if ns.start is not None and ns.end is not None and ns.end < ns.start:
        p.error("--to must not be earlier than --from")
    if ns.start is not None and ns.start < MIN_DATE:
        p.error(f"--from must be >= {MIN_DATE.isoformat()} (GKG 2020-01-01 cutoff)")

    return CliArgs(
        start=ns.start,
        end=ns.end,
        last_seconds=ns.last,
        latest=ns.latest,
        workers=ns.workers,
        force=ns.force,
        out_root=ns.out_root,
        filter_on_parse=ns.filter_on_parse,
        min_disk_pct=ns.min_disk_pct,
        min_disk_gb=ns.min_disk_gb,
    )


def _resolve_slots(args: CliArgs) -> list[dt.datetime]:
    """Resolve a CliArgs into a concrete list of 15-min slot datetimes."""
    if args.latest:
        return [resolve_latest()]
    if args.last_seconds is not None:
        end = dt.datetime.now(dt.timezone.utc).replace(tzinfo=None, second=0, microsecond=0)
        start = end - dt.timedelta(seconds=args.last_seconds)
        return list(slot_iter(start, end))
    assert args.start is not None and args.end is not None
    return list(slot_iter(args.start, args.end))


def _ts_now() -> str:
    """Filesystem-safe local timestamp for log filenames."""
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _run_raw_per_slot(args: CliArgs, opts: IngestOpts, slots: list[dt.datetime],
                      log, ts: str) -> dict:
    """Default raw mode: per-slot parquet writes consumed by 01_filter.py."""
    counts = {"OK": 0, "SKIPPED": 0, "MISSING": 0, "FAILED": 0}
    rows_total = 0
    drops_total = {"short_row": 0, "date_parse_fail": 0}
    failed_slots: list[str] = []
    drop_slots: list[dict] = []
    t0 = time.time()

    log(f"slots_total={len(slots)} workers={args.workers} force={args.force} "
        f"out={opts.out_root} mode=raw-per-slot")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_slot, s, opts): s for s in slots}
        for fut in as_completed(futures):
            res = fut.result()
            counts[res.status] += 1
            rows_total += res.rows
            if res.status == "FAILED":
                failed_slots.append(res.slot.strftime("%Y%m%d%H%M%S"))
            slot_drops = res.drops_short_row + res.drops_date_fail
            if slot_drops > 0:
                drops_total["short_row"] += res.drops_short_row
                drops_total["date_parse_fail"] += res.drops_date_fail
                drop_slots.append({
                    "slot": res.slot.strftime("%Y%m%d%H%M%S"),
                    "short_row": res.drops_short_row,
                    "date_parse_fail": res.drops_date_fail,
                })
            drops_suffix = (
                f" WARN drops short_row={res.drops_short_row} date_fail={res.drops_date_fail}"
                if slot_drops > 0 else ""
            )
            log(
                f"  {res.slot.strftime('%Y%m%d%H%M%S')} {res.status} "
                f"rows={res.rows} {res.elapsed_ms}ms"
                + (f" err={res.error}" if res.error else "")
                + drops_suffix
            )

    elapsed = round(time.time() - t0, 1)
    log(
        f"DONE mode=raw-per-slot total={len(slots)} ok={counts['OK']} skipped={counts['SKIPPED']} "
        f"missing={counts['MISSING']} failed={counts['FAILED']} rows={rows_total} elapsed={elapsed}s"
    )
    if drops_total["short_row"] + drops_total["date_parse_fail"] > 0:
        log(
            f"WARN malformed rows dropped: short_row={drops_total['short_row']} "
            f"date_parse_fail={drops_total['date_parse_fail']} "
            f"affected_slots={len(drop_slots)}"
        )

    return {
        "mode": "raw-per-slot",
        "slots_total": len(slots),
        "slots_ok": counts["OK"],
        "slots_skipped": counts["SKIPPED"],
        "slots_missing": counts["MISSING"],
        "slots_failed": counts["FAILED"],
        "rows_total": rows_total,
        "elapsed_sec": elapsed,
        "failed_slots": failed_slots,
        "drops_total": drops_total,
        "drop_slots": drop_slots,
        "exit_code": 1 if counts["FAILED"] > 0 else 0,
    }


def _run_filter_per_day(args: CliArgs, opts: IngestOpts, slots: list[dt.datetime],
                        log, ts: str) -> dict:
    """Filter-at-download mode: per-day batch with DuckDB filter, then 1 day.parquet."""
    days_map = _group_slots_by_day(slots)
    progress_path = PROJECT_ROOT / "logs" / "news" / "_ingest_filter_progress.json"

    day_results: list[DayResult] = []
    counts = {"OK": 0, "SKIPPED": 0, "FAILED": 0, "DISK_LOW": 0}
    raw_rows_total = 0
    filtered_rows_total = 0
    drops_total = {"short_row": 0, "date_parse_fail": 0}
    t0 = time.time()
    aborted_at: str | None = None

    log(f"days_total={len(days_map)} workers={args.workers} force={args.force} "
        f"out={opts.out_root} mode=filter-on-parse "
        f"min_disk_pct={opts.min_disk_pct} min_disk_gb={opts.min_disk_gb}")

    # Pre-check
    ok, msg = disk_guard_check(opts.out_root, opts.min_disk_pct, opts.min_disk_gb)
    log(msg)
    if not ok:
        log(f"ABORT: {msg}")
        return {"mode": "filter-on-parse", "days_total": len(days_map),
                "exit_code": 2, "abort_reason": msg}

    for i, (day, day_slots) in enumerate(days_map.items()):
        # Periodic disk re-check (every 10 days)
        if i > 0 and i % 10 == 0:
            ok, msg = disk_guard_check(opts.out_root, opts.min_disk_pct, opts.min_disk_gb)
            if not ok:
                log(f"ABORT at {day}: {msg}")
                aborted_at = day
                counts["DISK_LOW"] += len(days_map) - i
                break

        res = process_day(day, day_slots, opts)
        day_results.append(res)
        counts[res.status] = counts.get(res.status, 0) + 1
        raw_rows_total += res.raw_rows
        filtered_rows_total += res.filtered_rows
        drops_total["short_row"] += res.drops_short_row
        drops_total["date_parse_fail"] += res.drops_date_fail

        suffix = ""
        if res.drops_short_row + res.drops_date_fail > 0:
            suffix = f" WARN drops short_row={res.drops_short_row} date_fail={res.drops_date_fail}"
        log(f"  {day} {res.status} slots={res.slots_ok}/{res.slots_total} "
            f"raw={res.raw_rows} filtered={res.filtered_rows} "
            f"elapsed={res.elapsed_sec}s"
            + (f" err={res.error}" if res.error else "")
            + suffix)

        # Progress checkpoint
        progress_path.parent.mkdir(parents=True, exist_ok=True)
        progress_path.write_text(json.dumps({
            "ts": ts,
            "completed_days": i + 1,
            "total_days": len(days_map),
            "last_day": day,
            "raw_rows_total": raw_rows_total,
            "filtered_rows_total": filtered_rows_total,
            "counts": counts,
        }, indent=2))

    elapsed = round(time.time() - t0, 1)
    log(
        f"DONE mode=filter-on-parse days_total={len(days_map)} ok={counts['OK']} "
        f"skipped={counts['SKIPPED']} failed={counts['FAILED']} disk_low={counts['DISK_LOW']} "
        f"raw_rows={raw_rows_total} filtered_rows={filtered_rows_total} elapsed={elapsed}s"
    )
    if drops_total["short_row"] + drops_total["date_parse_fail"] > 0:
        log(f"WARN malformed rows dropped: short_row={drops_total['short_row']} "
            f"date_parse_fail={drops_total['date_parse_fail']}")

    failed_days = [r.day for r in day_results if r.status == "FAILED"]
    return {
        "mode": "filter-on-parse",
        "days_total": len(days_map),
        "days_ok": counts["OK"],
        "days_skipped": counts["SKIPPED"],
        "days_failed": counts["FAILED"],
        "days_disk_low": counts["DISK_LOW"],
        "raw_rows_total": raw_rows_total,
        "filtered_rows_total": filtered_rows_total,
        "elapsed_sec": elapsed,
        "failed_days": failed_days,
        "drops_total": drops_total,
        "aborted_at": aborted_at,
        "exit_code": 2 if aborted_at else (1 if counts["FAILED"] > 0 else 0),
    }


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns 0 on success, 1 if FAILED, 2 if disk-aborted."""
    args = parse_args(argv if argv is not None else sys.argv[1:])
    if args.out_root is not None:
        out_root = Path(args.out_root)
    else:
        out_root = DEFAULT_OUT_ROOT_FILTERED if args.filter_on_parse else DEFAULT_OUT_ROOT_RAW
    out_root.mkdir(parents=True, exist_ok=True)

    logs_dir = PROJECT_ROOT / "logs" / "news"
    logs_dir.mkdir(parents=True, exist_ok=True)
    ts = _ts_now()
    log_path = logs_dir / f"00_ingest_{ts}.log"
    json_path = logs_dir / f"00_ingest_{ts}.json"

    slots = _resolve_slots(args)
    opts = IngestOpts(
        out_root=out_root, force=args.force,
        filter_on_parse=args.filter_on_parse,
        min_disk_pct=args.min_disk_pct,
        min_disk_gb=args.min_disk_gb,
        workers=args.workers,
    )

    with log_path.open("w") as fh:
        def log(msg: str) -> None:
            line = f"[{dt.datetime.now().isoformat(timespec='seconds')}] {msg}"
            print(line)
            fh.write(line + "\n")
            fh.flush()

        if args.filter_on_parse:
            result = _run_filter_per_day(args, opts, slots, log, ts)
        else:
            result = _run_raw_per_slot(args, opts, slots, log, ts)

    summary = {
        "timestamp": ts,
        "args": {
            "from": args.start.isoformat() if args.start else None,
            "to": args.end.isoformat() if args.end else None,
            "last_seconds": args.last_seconds,
            "latest": args.latest,
            "workers": args.workers,
            "force": args.force,
            "out_root": str(out_root),
            "filter_on_parse": args.filter_on_parse,
            "min_disk_pct": args.min_disk_pct,
            "min_disk_gb": args.min_disk_gb,
        },
        **result,
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    return result.get("exit_code", 0)


if __name__ == "__main__":
    sys.exit(main())
