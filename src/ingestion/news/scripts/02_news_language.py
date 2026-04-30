"""
Step 08: canonical 기사에 언어 감지 + domain fallback 적용.

파이프라인:
  1. filtered_v1.parquet 로드
  2. is_canonical=True 97K row에 대해 langdetect.detect_langs()
  3. top confidence >= LANGDETECT_CONFIDENCE_THRESHOLD(0.65) 이면 그 언어 채택
     그렇지 않으면 config.fallback_lang_for_source() 로 domain 기반 추정
     그것도 실패하면 'und'
  4. 클러스터 동일 → lang 전파 (비-canonical row에 복사)
  5. enriched_v1.parquet 저장 (모든 원본 컬럼 + lang / lang_conf / lang_method)
  6. 최종 언어 분포 + scoring 언어 볼륨 리포트

실행:
    python scripts/02_language.py

산출:
    data/enriched_v1.parquet
    logs/02_language_{ts}.log + .json
"""
from __future__ import annotations

import datetime as dt
import json
import sys
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from langdetect import DetectorFactory, detect_langs
from langdetect.lang_detect_exception import LangDetectException

DetectorFactory.seed = 42

PROJECT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PROJECT / "src" / "ingestion" / "news"))
from config import (  # noqa: E402
    ALLOWED_LANGS,
    LANGDETECT_CONFIDENCE_THRESHOLD,
    fallback_lang_for_source,
    strong_lang_for_source,
)

LOG_DIR = PROJECT / "logs" / "news"
DATA_DIR = PROJECT / "data" / "news"
INPUT_PARQUET = DATA_DIR / "intermediate" / "filtered_v1.parquet"
OUTPUT_PARQUET = DATA_DIR / "intermediate" / "enriched_v1.parquet"


def log(msg: str, fh) -> None:
    """Persist the same timestamped progress message to stdout and log file."""
    stamped = f"[{dt.datetime.now().isoformat(timespec='seconds')}] {msg}"
    print(stamped)
    fh.write(stamped + "\n")
    fh.flush()


def classify_lang(title: str, source: str) -> tuple[str, float | None, str]:
    """Return (lang_code, confidence, method).
    method ∈ {'strong_override', 'langdetect', 'domain_fallback', 'und'}

    순서:
      1) STRONG_DOMAIN_OVERRIDE — 고신뢰 TLD는 langdetect 오감지를 덮어씀
      2) langdetect (conf >= threshold)
      3) DOMAIN_LANG_FALLBACK (약한 보조)
      4) 'und'
    """
    strong = strong_lang_for_source(source)
    if strong:
        return strong, None, "strong_override"

    if not title or len(title.strip()) < 3:
        fb = fallback_lang_for_source(source)
        if fb:
            return fb, None, "domain_fallback"
        return "und", None, "und"

    try:
        langs = detect_langs(title)
        if langs:
            top = langs[0]
            if top.prob >= LANGDETECT_CONFIDENCE_THRESHOLD:
                return top.lang, float(top.prob), "langdetect"
    except LangDetectException:
        pass

    fb = fallback_lang_for_source(source)
    if fb:
        return fb, None, "domain_fallback"
    return "und", None, "und"


def scoring_lang_route(lang: str) -> str:
    """Return scoring language label: 'en' | 'pt' | 'es' | 'dropped'."""
    if lang in ALLOWED_LANGS:
        return lang
    return "dropped"


def percent(part: int, whole: int) -> float:
    """Return a stable percentage for reports, including empty inputs."""
    if whole == 0:
        return 0.0
    return part / whole * 100


def main() -> int:
    LOG_DIR.mkdir(exist_ok=True)
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = LOG_DIR / f"02_language_{ts}.log"

    with log_path.open("w") as fh:
        log(f"input: {INPUT_PARQUET}", fh)
        log(f"output: {OUTPUT_PARQUET}", fh)
        log(f"confidence threshold: {LANGDETECT_CONFIDENCE_THRESHOLD}", fh)

        t0 = time.time()

        log("▶ [1] filtered_v1.parquet 로드", fh)
        df = pd.read_parquet(INPUT_PARQUET)
        log(f"  total rows = {len(df):,}", fh)
        log(f"  canonical rows = {int(df['is_canonical'].sum()):,}", fh)

        # canonical row만 실제 감지 비용을 들인다.
        # 중복 기사들은 같은 클러스터라는 전제 아래 이후 전파로 처리한다.
        canonical_mask = df["is_canonical"].astype(bool).values
        canon_indices = df.index[canonical_mask].tolist()
        log(f"▶ [2] canonical {len(canon_indices):,}건 langdetect 실행", fh)

        t1 = time.time()
        lang_list: list[str] = []
        conf_list: list[float | None] = []
        method_list: list[str] = []
        method_counter: Counter = Counter()

        for i, idx in enumerate(canon_indices):
            title = df.at[idx, "title"]
            source = df.at[idx, "source"]
            lang, conf, method = classify_lang(title, source)
            lang_list.append(lang)
            conf_list.append(conf)
            method_list.append(method)
            method_counter[method] += 1
            if (i + 1) % 20000 == 0:
                log(f"  진행: {i+1:,} / {len(canon_indices):,}", fh)

        log(f"  감지 완료: {time.time() - t1:.1f}s", fh)
        log(f"  방법 breakdown: {dict(method_counter)}", fh)

        # 빈 컬럼을 먼저 만들고 canonical 결과를 채운다.
        # non-canonical row는 다음 단계에서 cluster id 기준으로 복사된다.
        df["lang"] = pd.Series(["und"] * len(df), index=df.index, dtype="string")
        df["lang_conf"] = pd.Series([None] * len(df), index=df.index, dtype="Float64")
        df["lang_method"] = pd.Series(["und"] * len(df), index=df.index, dtype="string")
        for idx, lang, conf, method in zip(canon_indices, lang_list, conf_list, method_list):
            df.at[idx, "lang"] = lang
            if conf is not None:
                df.at[idx, "lang_conf"] = conf
            df.at[idx, "lang_method"] = method

        # ---------------------------------------------------------------
        # 3) 클러스터 대표의 lang을 비-canonical 멤버에 전파
        # ---------------------------------------------------------------
        log("▶ [3] 클러스터 내 lang 전파 (canonical → non-canonical)", fh)
        t2 = time.time()
        # canonical cluster 대표의 결과를 맵으로 만든 뒤,
        # 동일 cluster id를 가진 나머지 row에 그대로 전파한다.
        canon_df = df.loc[canon_indices, ["dedup_cluster_id", "lang", "lang_conf", "lang_method"]]
        cluster_to_info = canon_df.set_index("dedup_cluster_id").to_dict(orient="index")

        non_canon_mask = ~canonical_mask
        propagated = 0
        for idx in df.index[non_canon_mask]:
            cid = df.at[idx, "dedup_cluster_id"]
            info = cluster_to_info.get(cid)
            if info:
                df.at[idx, "lang"] = info["lang"]
                if pd.notna(info["lang_conf"]):
                    df.at[idx, "lang_conf"] = info["lang_conf"]
                df.at[idx, "lang_method"] = str(info["lang_method"]) + "_propagated"
                propagated += 1
        log(f"  전파 완료: {propagated:,}건  (소요 {time.time() - t2:.1f}s)", fh)

        # ---------------------------------------------------------------
        # 4) ALLOWED_LANGS (en/pt/es) 필터링
        # ---------------------------------------------------------------
        # 현재 downstream 모델 체인은 en/pt/es만 직접 지원하므로
        # 나머지 언어는 이 단계에서 제거해 계산 비용을 통제한다.
        log(f"▶ [4] ALLOWED_LANGS 필터링: {sorted(ALLOWED_LANGS)}", fh)
        t_filter = time.time()
        before = len(df)
        before_canon = int(df["is_canonical"].sum())
        df = df[df["lang"].isin(ALLOWED_LANGS)].reset_index(drop=True)
        after = len(df)
        after_canon = int(df["is_canonical"].sum())
        log(f"  전체: {before:,} → {after:,} ({after - before:+,}, -{percent(before - after, before):.1f}%)", fh)
        log(f"  canonical: {before_canon:,} → {after_canon:,} ({after_canon - before_canon:+,})", fh)
        log(f"  소요 {time.time() - t_filter:.1f}s", fh)

        # ---------------------------------------------------------------
        # 4a) Very short title drop (P1): RSS placeholder·station ID 등 노이즈 제거
        # ---------------------------------------------------------------
        # 너무 짧은 제목은 기사 의미보다 방송국 식별자, RSS placeholder일 가능성이 높다.
        # canonical이 제거 대상이면 같은 cluster 전체를 제거해 일관성을 유지한다.
        log("▶ [4a] Very short title drop (len < 20)", fh)
        t_p1 = time.time()
        short_mask = df["title"].str.len() < 20
        short_cids = set(
            df.loc[short_mask & df["is_canonical"].astype(bool), "dedup_cluster_id"].tolist()
        )
        p1_cluster_drop = df["dedup_cluster_id"].isin(short_cids)
        before_p1 = len(df)
        before_canon_p1 = int(df["is_canonical"].sum())
        df = df[~p1_cluster_drop].reset_index(drop=True)
        log(f"  영향받은 클러스터: {len(short_cids):,}개", fh)
        log(f"  전체: {before_p1:,} → {len(df):,} ({len(df) - before_p1:+,})", fh)
        log(f"  canonical: {before_canon_p1:,} → {int(df['is_canonical'].sum()):,}", fh)
        log(f"  소요 {time.time() - t_p1:.1f}s", fh)

        # ---------------------------------------------------------------
        # 4b) ES "soy-verb" false positive 제거
        #     스페인어에서 'soy'는 동사 'I am'. title에 soja/soybean/soya 없이
        #     'soy'만 있으면 거의 확실히 대두와 무관.
        # ---------------------------------------------------------------
        log("▶ [4b] ES soy-verb false positive post-filter", fh)
        t_es = time.time()
        title_series = df["title"].fillna("")
        has_soy = title_series.str.contains(r"\bsoy\b", case=False, regex=True, na=False)
        has_definitive = title_series.str.contains(
            r"\b(?:soja|soybean|soybeans|soya)\b", case=False, regex=True, na=False
        )
        is_es = df["lang"] == "es"
        drop_mask = is_es & has_soy & ~has_definitive

        # 클러스터 전체 drop 여부: 캐노니컬 행이 drop 대상이면 동일 클러스터 전부 drop
        drop_cluster_ids = set(
            df.loc[drop_mask & df["is_canonical"].astype(bool), "dedup_cluster_id"].tolist()
        )
        cluster_drop_mask = df["dedup_cluster_id"].isin(drop_cluster_ids)

        before_es = len(df)
        before_canon_es = int(df["is_canonical"].sum())
        df = df[~cluster_drop_mask].reset_index(drop=True)
        after_es = len(df)
        after_canon_es = int(df["is_canonical"].sum())
        log(f"  직접 drop 대상 row: {int(drop_mask.sum()):,}", fh)
        log(f"  영향받은 클러스터: {len(drop_cluster_ids):,}개", fh)
        log(f"  전체: {before_es:,} → {after_es:,} ({after_es - before_es:+,})", fh)
        log(f"  canonical: {before_canon_es:,} → {after_canon_es:,} ({after_canon_es - before_canon_es:+,})", fh)
        log(f"  소요 {time.time() - t_es:.1f}s", fh)

        # ---------------------------------------------------------------
        # 5) 저장
        # ---------------------------------------------------------------
        log("▶ [5] enriched_v1.parquet 저장", fh)
        t3 = time.time()
        df.to_parquet(OUTPUT_PARQUET, index=False, compression="snappy")
        log(f"  저장 완료: {OUTPUT_PARQUET}", fh)
        log(f"  파일 크기: {OUTPUT_PARQUET.stat().st_size / 1024 / 1024:.1f} MB  (소요 {time.time() - t3:.1f}s)", fh)

        # ---------------------------------------------------------------
        # 6) 리포트
        # ---------------------------------------------------------------
        log("▶ [6] 리포트 (ALLOWED_LANGS 필터 후 canonical 기준)", fh)
        canonical_mask = df["is_canonical"].astype(bool).values
        canon_final = df.loc[canonical_mask]

        # 5-1) 언어 분포
        lang_dist = Counter(canon_final["lang"].tolist())
        total_canon = len(canon_final)
        log(f"  전체 canonical: {total_canon:,}", fh)
        log("  언어 분포 (Top 15):", fh)
        lang_rows = []
        for lang, n in lang_dist.most_common(15):
            share = percent(n, total_canon)
            log(f"    {lang}: {n:,} ({share:.1f}%)", fh)
            lang_rows.append({"lang": lang, "count": n, "pct": round(share, 2)})

        # 5-2) scoring language volume (dropped 제외)
        route_counter: Counter = Counter()
        for lang in canon_final["lang"].tolist():
            route_counter[scoring_lang_route(lang)] += 1
        log("  scoring 언어 볼륨 (en/pt/es):", fh)
        total_route = sum(route_counter.values())
        route_rows = {}
        for route in ["en", "pt", "es"]:
            n = route_counter.get(route, 0)
            route_share = percent(n, total_route)
            log(f"    {route}: {n:,} ({route_share:.1f}%)", fh)
            route_rows[route] = {"count": n, "pct": round(route_share, 2)}

        # 5-3) Method breakdown (감지 출처 비율)
        log(f"  langdetect vs domain fallback vs und:", fh)
        method_breakdown = {}
        for method, n in method_counter.most_common():
            share = percent(n, total_canon)
            log(f"    {method}: {n:,} ({share:.1f}%)", fh)
            method_breakdown[method] = {"count": n, "pct": round(share, 2)}

        # 5-4) 각 scoring language 별 샘플 5건
        log("  라우팅별 샘플 title:", fh)
        samples_by_route: dict[str, list[dict]] = {}
        for route in ["en", "pt", "es"]:
            subset = canon_final[canon_final["lang"].apply(lambda x: scoring_lang_route(x) == route)]
            samples = []
            if len(subset):
                for _, row in subset.sample(min(5, len(subset)), random_state=42).iterrows():
                    title_preview = (row["title"] or "")[:80]
                    samples.append({"lang": row["lang"], "source": row["source"], "title": title_preview})
                    log(f"    [{route}/{row['lang']}] ({row['source']}) {title_preview}", fh)
            samples_by_route[route] = samples

        elapsed = time.time() - t0
        log(f"▶ 총 소요: {elapsed:.1f}s", fh)

        summary = {
            "timestamp": ts,
            "threshold": LANGDETECT_CONFIDENCE_THRESHOLD,
            "total_canonical": total_canon,
            "lang_distribution_top15": lang_rows,
            "scoring_routing": route_rows,
            "detection_method_breakdown": method_breakdown,
            "samples_by_route": samples_by_route,
            "output_parquet": str(OUTPUT_PARQUET),
            "output_parquet_mb": round(OUTPUT_PARQUET.stat().st_size / 1024 / 1024, 1),
            "elapsed_sec": round(elapsed, 1),
        }
        (LOG_DIR / f"02_language_{ts}.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False)
        )
        log(f"summary json: 02_language_{ts}.json", fh)

    return 0


if __name__ == "__main__":
    sys.exit(main())
