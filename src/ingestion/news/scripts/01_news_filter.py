"""
Step 07: Soybean 필터 통과 row를 필터링·정규화·dedup 한 결과를 data/filtered_v1.parquet 으로 저장.

단계:
  1. DuckDB로 soybean_filter_sql() 통과 row 추출 (+정규화 URL, +source/title/GKG metadata)
  2. URL 정규화 dedup: 같은 normalized URL → earliest DATE row 채택
  3. MinHash (char 5-gram, 128 perm) + LSH (Jaccard 0.8) + 24h 윈도우 클러스터링
  4. Canonical 선택: earliest DATE → TRUSTED_SOURCES → shortest URL
  5. Parquet 저장 (dedup_cluster_id, is_canonical 포함)

실행:
    python scripts/01_filter.py

산출:
    data/filtered_v1.parquet
    logs/01_filter_{ts}.log + .json
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import duckdb
import pandas as pd
from datasketch import MinHash, MinHashLSH

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT / "src" / "ingestion" / "news"))
from config import (  # noqa: E402
    GKG_RAW_DIR,
    NORMALIZED_URL_SQL,
    TRUSTED_SOURCES,
    soybean_filter_sql,
)

GKG_GLOB = f"{GKG_RAW_DIR}/dt=*/*.parquet"
LOG_DIR = PROJECT / "logs" / "news"
DATA_DIR = PROJECT / "data" / "news"
OUTPUT_PARQUET = DATA_DIR / "intermediate" / "filtered_v1.parquet"

# MinHash/LSH 파라미터
SHINGLE_K = 5
NUM_PERM = 128
JACCARD_THRESHOLD = 0.80
WINDOW_SECONDS = 24 * 3600

_WS_RE = re.compile(r"\s+")
# audit 2026-04-28 §3.7: AP wire 변형 — " – Winnipeg Free Press", " | News, Sports, Jobs"
# 같은 publisher-suffix가 Jaccard를 0.80 아래로 떨어뜨려 동일 기사가 여러 cluster로 잔존.
# en-dash / em-dash / pipe 모두 양쪽 공백 필수 → "US–Iran" 같은 본문 합성어는 보호.
_PUBLISHER_SUFFIX_RE = re.compile(r"\s+[–—|]\s+.{1,80}$")


def log(msg: str, fh) -> None:
    """Mirror timestamped progress lines to stdout and the step log file."""
    stamped = f"[{dt.datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(stamped)
    fh.write(stamped + "\n")
    fh.flush()


def shingles(text: str, k: int = SHINGLE_K) -> set[str]:
    """Convert a title into normalized character shingles for near-duplicate search."""
    raw = _PUBLISHER_SUFFIX_RE.sub("", (text or "").strip())
    norm = _WS_RE.sub(" ", raw.lower().strip())
    if len(norm) < k:
        return {norm} if norm else set()
    return {norm[i : i + k] for i in range(len(norm) - k + 1)}


def make_minhash(text: str) -> MinHash:
    """Build a MinHash signature so similar titles can be queried via LSH."""
    mh = MinHash(num_perm=NUM_PERM)
    for s in shingles(text):
        mh.update(s.encode("utf-8"))
    return mh


def date_int_to_iso(d: int) -> str:
    """Convert GKG integer datetime to YYYY-MM-DD for downstream parquet readability."""
    s = str(d).zfill(14)
    return f"{s[0:4]}-{s[4:6]}-{s[6:8]}"


def date_int_to_ts(d: int) -> float:
    """Convert GKG integer datetime to a Unix timestamp for window comparisons."""
    s = str(d).zfill(14)
    return dt.datetime(
        int(s[0:4]), int(s[4:6]), int(s[6:8]),
        int(s[8:10]), int(s[10:12]), int(s[12:14]),
    ).timestamp()


def pct(part: int, whole: int) -> float:
    """Return a stable percentage for reports, including empty inputs."""
    if whole == 0:
        return 0.0
    return part / whole * 100


def main() -> int:
    LOG_DIR.mkdir(exist_ok=True)
    DATA_DIR.mkdir(exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"01_filter_{ts}.log"

    with log_path.open("w") as fh:
        log(f"GKG glob: {GKG_GLOB}", fh)
        log(f"output: {OUTPUT_PARQUET}", fh)
        log(f"params: shingle=char{SHINGLE_K}, num_perm={NUM_PERM}, jaccard={JACCARD_THRESHOLD}, window={WINDOW_SECONDS}s", fh)

        con = duckdb.connect()
        base = f"read_parquet('{GKG_GLOB}')"
        filter_clause = soybean_filter_sql()

        t0 = time.time()

        # ---------------------------------------------------------------
        # 1) 필터 + URL 정규화 + 1차 dedup (같은 URL은 earliest DATE 채택)
        # ---------------------------------------------------------------
        log("▶ [1] 필터 통과 row 추출 + URL 정규화 + URL dedup (earliest DATE)", fh)
        sql = f"""
        WITH filtered AS (
            SELECT
                DATE                AS date_int,
                {NORMALIZED_URL_SQL} AS nurl,
                title,
                lower(SourceCommonName) AS source,
                V2Themes            AS gdelt_themes,
                V2Organizations     AS gdelt_organizations,
                V2Locations         AS gdelt_locations
            FROM {base}
            WHERE {filter_clause}
        ),
        ranked AS (
            SELECT *,
                   ROW_NUMBER() OVER (PARTITION BY nurl ORDER BY date_int ASC) AS rn
            FROM filtered
        )
        SELECT date_int, nurl AS url, title, source, gdelt_themes, gdelt_organizations, gdelt_locations
        FROM ranked
        WHERE rn = 1
        ORDER BY date_int
        """
        df = con.execute(sql).df()
        load_elapsed = time.time() - t0
        log(f"  rows after URL dedup: {len(df):,}  (소요 {load_elapsed:.1f}s)", fh)

        # ---------------------------------------------------------------
        # 2) MinHash 생성
        # ---------------------------------------------------------------
        # 여기서는 기사 본문이 아니라 title만 사용한다.
        # 목적은 "같은 와이어 기사가 다른 매체로 재게시된 경우"를 싸게 묶는 것이다.
        log("▶ [2] MinHash 생성 (char 5-gram, 128 perm)", fh)
        t1 = time.time()
        minhashes: list[MinHash] = []
        for i, title in enumerate(df["title"].tolist()):
            mh = make_minhash(title)
            minhashes.append(mh)
            if (i + 1) % 20000 == 0:
                log(f"  진행: {i+1:,} / {len(df):,}", fh)
        log(f"  MinHash 생성 완료: {time.time() - t1:.1f}s", fh)

        # ---------------------------------------------------------------
        # 3) LSH 인덱스 구축 (글로벌)
        # ---------------------------------------------------------------
        # 모든 타이틀의 signature를 인덱싱해 두면,
        # 이후 각 기사에서 유사 후보만 빠르게 가져와 실제 Jaccard를 재검증할 수 있다.
        log("▶ [3] LSH 인덱스 구축", fh)
        t2 = time.time()
        lsh = MinHashLSH(threshold=JACCARD_THRESHOLD, num_perm=NUM_PERM)
        for i, mh in enumerate(minhashes):
            lsh.insert(str(i), mh)
            if (i + 1) % 20000 == 0:
                log(f"  진행: {i+1:,} / {len(minhashes):,}", fh)
        log(f"  LSH 인덱스 완료: {time.time() - t2:.1f}s", fh)

        # ---------------------------------------------------------------
        # 4) Union-Find 클러스터링 (24h 윈도우 내 후보만 병합)
        # ---------------------------------------------------------------
        # 같은 제목이라도 시차가 너무 크면 다른 이벤트일 수 있으므로 24시간 윈도우를 둔다.
        # LSH는 false positive가 가능하므로 최종 병합 전 실제 jaccard를 다시 본다.
        log("▶ [4] 클러스터링 (24h 윈도우, Union-Find)", fh)
        t3 = time.time()
        n = len(df)
        parent = list(range(n))

        def find(x: int) -> int:
            """Path-compressed find for cluster root lookup."""
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x: int, y: int) -> None:
            """Merge two candidate duplicate groups."""
            rx, ry = find(x), find(y)
            if rx != ry:
                parent[ry] = rx

        date_ints = df["date_int"].tolist()
        timestamps = [date_int_to_ts(d) for d in date_ints]

        lsh_query_edges = 0
        confirmed_merges = 0
        for i, mh in enumerate(minhashes):
            candidates = lsh.query(mh)
            ts_i = timestamps[i]
            for j_str in candidates:
                j = int(j_str)
                if j <= i:  # symmetric — 한 방향만
                    continue
                if abs(timestamps[j] - ts_i) > WINDOW_SECONDS:
                    continue
                lsh_query_edges += 1
                # LSH는 false positive 가능 → 실제 jaccard 확인
                if mh.jaccard(minhashes[j]) >= JACCARD_THRESHOLD:
                    union(i, j)
                    confirmed_merges += 1
            if (i + 1) % 20000 == 0:
                log(f"  진행: {i+1:,} / {n:,}  edges_checked={lsh_query_edges:,} merges={confirmed_merges:,}", fh)

        log(f"  클러스터링 완료: {time.time() - t3:.1f}s", fh)
        log(f"  LSH edges checked: {lsh_query_edges:,}  confirmed merges: {confirmed_merges:,}", fh)

        # ---------------------------------------------------------------
        # 5) 클러스터 수집 + canonical 선택
        # ---------------------------------------------------------------
        # canonical은 "가장 이른 기사 -> trusted source 우선 -> URL 짧은 것" 순으로 고른다.
        # 이후 스코어링은 canonical 기준으로만 수행하고, 점수는 클러스터 전체에 전파된다.
        log("▶ [5] canonical 선택 (earliest DATE → TRUSTED → shortest URL)", fh)
        t4 = time.time()

        roots = [find(i) for i in range(n)]
        clusters: dict[int, list[int]] = defaultdict(list)
        for i, r in enumerate(roots):
            clusters[r].append(i)

        urls = df["url"].tolist()
        sources = df["source"].tolist()

        is_canonical = [False] * n
        cluster_id_compact = [0] * n

        sorted_roots = sorted(clusters.keys())
        for cid, root in enumerate(sorted_roots):
            members = clusters[root]
            # 우선순위 정렬
            members_sorted = sorted(
                members,
                key=lambda i: (
                    date_ints[i],
                    0 if (sources[i] or "") in TRUSTED_SOURCES else 1,
                    len(urls[i] or ""),
                ),
            )
            canonical_idx = members_sorted[0]
            is_canonical[canonical_idx] = True
            for i in members:
                cluster_id_compact[i] = cid

        log(f"  클러스터 {len(clusters):,}개  (소요 {time.time() - t4:.1f}s)", fh)

        # ---------------------------------------------------------------
        # 6) 결과 DataFrame 조립 + parquet 저장
        # ---------------------------------------------------------------
        log("▶ [6] parquet 저장", fh)
        t5 = time.time()
        df_out = pd.DataFrame({
            "date": [date_int_to_iso(d) for d in date_ints],
            "date_int": date_ints,
            "title": df["title"].tolist(),
            "url": urls,
            "source": sources,
            "gdelt_themes": df["gdelt_themes"].tolist(),
            "gdelt_organizations": df["gdelt_organizations"].tolist(),
            "gdelt_locations": df["gdelt_locations"].tolist(),
            "dedup_cluster_id": cluster_id_compact,
            "is_canonical": is_canonical,
        })
        df_out.to_parquet(OUTPUT_PARQUET, index=False, compression="snappy")
        log(f"  저장 완료: {OUTPUT_PARQUET}  (소요 {time.time() - t5:.1f}s)", fh)
        log(f"  파일 크기: {OUTPUT_PARQUET.stat().st_size / 1024 / 1024:.1f} MB", fh)

        # ---------------------------------------------------------------
        # 7) 리포트
        # ---------------------------------------------------------------
        log("▶ [7] 리포트", fh)
        n_canonical = int(sum(is_canonical))
        log(f"  전체 row: {n:,}", fh)
        log(f"  canonical row: {n_canonical:,}  ({pct(n_canonical, n):.1f}%)", fh)
        log(f"  클러스터 수: {len(clusters):,}", fh)

        # 클러스터 크기 분포
        size_dist = Counter(len(m) for m in clusters.values())
        log("  클러스터 크기 분포:", fh)
        for k in sorted(size_dist.keys()):
            log(f"    size={k:>4}: {size_dist[k]:>6}개 클러스터", fh)

        # Top 10 큰 클러스터 샘플
        log("  상위 10 큰 클러스터 대표 title:", fh)
        top_clusters = sorted(clusters.items(), key=lambda kv: len(kv[1]), reverse=True)[:10]
        big_cluster_samples = []
        for root, members in top_clusters:
            canonical_i = next(i for i in members if is_canonical[i])
            title = (df_out.iloc[canonical_i]["title"] or "")[:90]
            src = df_out.iloc[canonical_i]["source"]
            log(f"    [size={len(members):>4}] ({src}) {title}", fh)
            big_cluster_samples.append({
                "size": len(members),
                "canonical_source": src,
                "canonical_title": title,
            })

        elapsed = time.time() - t0
        log(f"▶ 총 소요: {elapsed:.1f}s", fh)

        summary = {
            "timestamp": ts,
            "params": {
                "shingle_k": SHINGLE_K,
                "num_perm": NUM_PERM,
                "jaccard_threshold": JACCARD_THRESHOLD,
                "window_seconds": WINDOW_SECONDS,
            },
            "input_rows_after_url_dedup": n,
            "clusters": len(clusters),
            "canonical_rows": n_canonical,
            "canonical_ratio_pct": round(pct(n_canonical, n), 2),
            "lsh_edges_checked": lsh_query_edges,
            "confirmed_merges": confirmed_merges,
            "cluster_member_count_distribution": {
                str(k): v for k, v in sorted(size_dist.items())
            },
            "top_clusters_sample": big_cluster_samples,
            "output_parquet": str(OUTPUT_PARQUET),
            "output_parquet_mb": round(OUTPUT_PARQUET.stat().st_size / 1024 / 1024, 1),
            "elapsed_sec": round(elapsed, 1),
        }
        (LOG_DIR / f"01_filter_{ts}.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        log(f"summary json: 01_filter_{ts}.json", fh)

    return 0


if __name__ == "__main__":
    sys.exit(main())
