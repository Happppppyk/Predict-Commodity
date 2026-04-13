"""
역할: GDELT v2 DOC API로 뉴스를 수집하고, LLM 없이 규칙 기반으로 카테고리·스코어를 매겨 `raw_news_scored`에 적재한다.
레이어: 데이터 수집 및 SQLite 적재 (1)
"""

from __future__ import annotations

import json
import sqlite3
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

# --- 감성(상승=가격·리스크 상방에 가깝게 해석) 키워드 → SS ---
# 상승 키워드: +0.6 ~ +1.0
BULLISH_KEYWORD_SCORES: dict[str, float] = {
    "drought": 0.95,
    "ban": 0.85,
    "export ban": 0.9,
    "cut": 0.65,
    "cuts": 0.65,
    "surge": 0.75,
    "shortage": 0.9,
    "shortages": 0.9,
    "soaring": 0.7,
    "spike": 0.7,
    "disruption": 0.65,
}

# 하락 키워드: -0.6 ~ -1.0
BEARISH_PHRASE_SCORES: dict[str, float] = {
    "record harvest": -1.0,
    "weak demand": -0.85,
    "bumper crop": -1.0,
}

BEARISH_KEYWORD_SCORES: dict[str, float] = {
    "surplus": -0.85,
    "bumper": -0.9,
}

GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
GDELT_QUERY = (
    "soybean oil OR soybean OR palm oil Brazil drought RVO biodiesel"
)

RAW_NEWS_TABLE = "raw_news_scored"

CREATE_RAW_NEWS_SQL = f"""
CREATE TABLE IF NOT EXISTS {RAW_NEWS_TABLE} (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date TEXT,
    title TEXT,
    url TEXT NOT NULL UNIQUE,
    source TEXT,
    category TEXT,
    sentiment_score REAL,
    impact_score REAL,
    certainty_factor REAL,
    weight REAL,
    final_score REAL
);
"""

INSERT_NEWS_IGNORE_SQL = f"""
INSERT OR IGNORE INTO {RAW_NEWS_TABLE}
    (date, title, url, source, category,
     sentiment_score, impact_score, certainty_factor, weight, final_score)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
"""

USER_AGENT = "Mozilla/5.0 (compatible; soybean-oil-poc/1.0; research)"


def _gdelt_ts(dt: datetime) -> str:
    """GDELT startdatetime/enddatetime: UTC YYYYMMDDHHMMSS."""
    return dt.strftime("%Y%m%d%H%M%S")


def _seendate_to_iso_date(seendate: str | None) -> str:
    """seendate 앞 8자리 YYYYMMDD → YYYY-MM-DD."""
    if not seendate or len(seendate) < 8:
        return ""
    y, m, d = seendate[:4], seendate[4:6], seendate[6:8]
    try:
        datetime(int(y), int(m), int(d))
    except ValueError:
        return ""
    return f"{y}-{m}-{d}"


def fetch_gdelt_news(days_back: int = 30) -> list[dict[str, str]]:
    """
    GDELT v2 DOC API (mode=artlist, format=json)로 기사 목록을 가져온다.

    반환: [{date, title, url, source, language}] — language == \"English\" 만.
    """
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=max(1, days_back))
    params = {
        "query": GDELT_QUERY,
        "mode": "artlist",
        "maxrecords": "250",
        "format": "json",
        "startdatetime": _gdelt_ts(start),
        "enddatetime": _gdelt_ts(end),
    }
    url = f"{GDELT_DOC_URL}?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urlopen(req, timeout=90) as resp:
            raw = resp.read().decode("utf-8", errors="replace")
    except (HTTPError, URLError, OSError, TimeoutError):
        return []

    if not raw.strip().startswith("{"):
        return []

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return []

    articles = payload.get("articles")
    if not isinstance(articles, list):
        return []

    out: list[dict[str, str]] = []
    for a in articles:
        if not isinstance(a, dict):
            continue
        lang = str(a.get("language") or "").strip()
        if lang != "English":
            continue
        title = str(a.get("title") or "").strip()
        link = str(a.get("url") or "").strip()
        if not title or not link:
            continue
        domain = str(a.get("domain") or "").strip()
        source = domain or str(a.get("sourcecountry") or "").strip()
        date_s = _seendate_to_iso_date(str(a.get("seendate") or ""))
        out.append(
            {
                "date": date_s,
                "title": title,
                "url": link,
                "source": source,
                "language": lang,
            }
        )
    return out


def classify_category(title: str) -> str:
    """
    제목 키워드 매칭으로 카테고리 (LLM 없음).
    우선순위: weather → policy → crop → export → other.
    """
    t = title.lower()

    weather_kw = (
        "drought",
        "flood",
        "rain",
        "la nina",
        "la niña",
        "el nino",
        "el niño",
        "frost",
        "clima",
        "chuva",
    )
    if any(k in t for k in weather_kw):
        return "weather"

    policy_kw = (
        "rvo",
        "biodiesel",
        "mandate",
        "epa",
        "export ban",
        "tariff",
        "subsidy",
        "subsidies",
    )
    if any(k in t for k in policy_kw):
        return "policy"

    crop_kw = (
        "harvest",
        "yield",
        "planting",
        "usda",
        "wasde",
        "production",
        "crop",
    )
    if any(k in t for k in crop_kw):
        return "crop"

    export_kw = ("export", "shipment", "port", "logistics", "cargo")
    if any(k in t for k in export_kw):
        return "export"

    return "other"


def _sentiment_score_from_title(title: str) -> float:
    """규칙 기반 SS ∈ [-1, 1]."""
    t = title.lower()
    bull = 0.0
    for phrase, sc in sorted(BULLISH_KEYWORD_SCORES.items(), key=lambda x: -len(x[0])):
        if phrase in t:
            bull = max(bull, sc)
    bear = 0.0
    for phrase, sc in sorted(BEARISH_PHRASE_SCORES.items(), key=lambda x: -len(x[0])):
        if phrase in t:
            bear = min(bear, sc)
    for word, sc in sorted(BEARISH_KEYWORD_SCORES.items(), key=lambda x: -len(x[0])):
        if word in t:
            bear = min(bear, sc)
    if bull != 0.0 and bear != 0.0:
        ss = bull + bear
    elif bull != 0.0:
        ss = bull
    else:
        ss = bear
    return max(-1.0, min(1.0, ss))


def _impact_score_from_title(title: str) -> float:
    """IS: 1 ~ 5."""
    t = title.lower()
    s = 1.0
    if "brazil" in t or "argentina" in t:
        s += 2.0
    if "indonesia" in t or "malaysia" in t:
        s += 2.0
    if "usda" in t or "epa" in t or "government" in t:
        s += 1.0
    return max(1.0, min(5.0, s))


def _certainty_factor_from_source(source: str) -> float:
    s = source.lower()
    if any(x in s for x in ("reuters", "bloomberg", "usda")):
        return 0.8
    if "ft.com" in s or "financial times" in s:
        return 0.7
    if "wsj" in s or "wall street journal" in s:
        return 0.7
    return 0.4


def _weight_from_category_and_title(category: str, title: str) -> float:
    t = title.lower()
    if category == "weather" and "brazil" in t:
        return 1.0
    if category == "policy":
        return 0.9
    return 0.5


def score_news_item(title: str, category: str, source: str) -> dict[str, float]:
    """
    규칙 기반 스코어.

    - sentiment_score (SS)
    - impact_score (IS) 1~5
    - certainty_factor (CF)
    - weight (W)
    - final_score = SS × IS × CF × W
    """
    ss = _sentiment_score_from_title(title)
    is_ = _impact_score_from_title(title)
    cf = _certainty_factor_from_source(source)
    w = _weight_from_category_and_title(category, title)
    final = ss * is_ * cf * w
    return {
        "sentiment_score": ss,
        "impact_score": is_,
        "certainty_factor": cf,
        "weight": w,
        "final_score": final,
    }


def load_news_to_db(scored_list: list[dict[str, Any]], conn: sqlite3.Connection) -> int:
    """
    `raw_news_scored`에 INSERT OR IGNORE (url 중복 무시).

    scored_list 원소는 date, title, url, source, category,
    sentiment_score, impact_score, certainty_factor, weight, final_score 포함.
    """
    conn.execute(CREATE_RAW_NEWS_SQL)
    conn.commit()
    if not scored_list:
        return 0
    rows = []
    for r in scored_list:
        rows.append(
            (
                str(r.get("date") or ""),
                str(r.get("title") or ""),
                str(r.get("url") or ""),
                str(r.get("source") or ""),
                str(r.get("category") or ""),
                float(r["sentiment_score"]),
                float(r["impact_score"]),
                float(r["certainty_factor"]),
                float(r["weight"]),
                float(r["final_score"]),
            )
        )
    conn.executemany(INSERT_NEWS_IGNORE_SQL, rows)
    conn.commit()
    return len(rows)


def run_news_pipeline(conn: sqlite3.Connection, days_back: int = 30) -> list[dict[str, Any]]:
    """
    fetch → classify → score → load 순 실행.

    완료 후 출력:
    - 카테고리별 수집 건수
    - 평균 final_score
    - final_score 상위 5개 제목
    """
    raw_items = fetch_gdelt_news(days_back=days_back)
    scored: list[dict[str, Any]] = []
    for it in raw_items:
        cat = classify_category(it["title"])
        sc = score_news_item(it["title"], cat, it["source"])
        scored.append(
            {
                "date": it["date"],
                "title": it["title"],
                "url": it["url"],
                "source": it["source"],
                "category": cat,
                **sc,
            }
        )

    load_news_to_db(scored, conn)

    by_cat = Counter(r["category"] for r in scored)
    print("[raw_news_scored] 카테고리별 건수:")
    for c in sorted(by_cat.keys()):
        print(f"  {c}: {by_cat[c]}")

    if scored:
        mean_f = sum(r["final_score"] for r in scored) / len(scored)
        print(f"[raw_news_scored] 평균 final_score: {mean_f:.4f} (n={len(scored)})")
        top = sorted(scored, key=lambda x: x["final_score"], reverse=True)[:5]
        print("[raw_news_scored] final_score 상위 5개 제목:")
        for i, r in enumerate(top, 1):
            print(f"  {i}. [{r['final_score']:.4f}] {r['title'][:120]}")
    else:
        print("[raw_news_scored] 수집 건수 0 (GDELT 응답 없음·레이트리밋·필터 등)")

    return scored


# --- 하위 호환(초기 스켈레톤 이름) ---


def ingest_news_batch(sources: list[str]) -> None:
    """미사용. GDELT 단일 소스로 대체됨."""
    raise NotImplementedError("GDELT 파이프라인은 run_news_pipeline(conn)을 사용하세요.")


def score_sentiment(text: str) -> float:
    """제목 기준 규칙 기반 감성 점수 (카테고리 없이 SS만)."""
    return _sentiment_score_from_title(text)


def persist_raw_news_scores(conn: sqlite3.Connection, records: list[dict]) -> None:
    """load_news_to_db 별칭."""
    load_news_to_db(records, conn)


if __name__ == "__main__":
    from pathlib import Path

    db = Path(__file__).resolve().parents[2] / "data" / "db" / "soybean.db"
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as c:
        run_news_pipeline(c, days_back=30)
