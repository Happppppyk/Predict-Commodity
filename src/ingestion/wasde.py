"""
역할: USDA WASDE/PSD 계열 수급 지표를 `raw_wasde`에 적재한다.

- `load_wasde_from_usda_api`: FAS Open Data PSD API (키 필요, https://apps.fas.usda.gov/opendataweb/home).
- `load_wasde_from_csv`: 플랫 CSV 또는 `data/raw/wasde_psd.csv` 형식의 PSD Oilseeds 벌크 CSV.

API 키: 프로젝트 루트 `.env` 파일의 `USDA_OPEN_DATA_API_KEY` 또는 동일 이름의 환경변수.
(`python-dotenv` 설치 시 모듈 로드 시 `.env`를 자동 로드)

⚠️ master_daily 조인·보간 (시계열 누수 방지):
`raw_wasde`는 PK가 `(release_date, marketing_year)` 복합일 수 있음 — 조인 시 마케팅연도·공표일 정의를 맞출 것.
release_date 기준으로만 master_daily에 조인할 것.
월중 날짜에 forward-fill 적용 시 release 이후 날짜만 채울 것.
"""

from __future__ import annotations

import json
import os
import sqlite3
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _load_dotenv_if_available() -> None:
    """프로젝트 루트 `.env`를 환경변수로 로드한다. `python-dotenv` 없으면 무시."""
    env_file = _PROJECT_ROOT / ".env"
    if not env_file.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_file)
    except ImportError:
        pass


_load_dotenv_if_available()

# FAS PSD Open Data (스웨거: /OpenData/swagger/docs/v1)
PSD_OPEN_DATA_BASE = "https://apps.fas.usda.gov/OpenData/api/psd"
PSD_REGISTRATION_URL = "https://apps.fas.usda.gov/opendataweb/home"
PSD_DOWNLOADS_URL = "https://apps.fas.usda.gov/psdonline/app/index.html#/app/downloads"

ENV_USDA_API_KEY = "USDA_OPEN_DATA_API_KEY"

# PSD Open Data 품목 코드(숫자). 문자열 코드(SOYBE 등)는 경로에서 500/빈 응답을 유발할 수 있음.
COMMODITY_SOYBEANS = "2222000"  # Soybeans
COMMODITY_SOYBEAN_OIL = "4235000"  # Soybean Oil

# PSD 벌크 CSV 품목명(정확 일치). "Oilseed, Soybean" 등은 Oil 부분문자열로 오인되기 쉬워 제외.
SOYBEAN_OIL_PSD_DESCRIPTIONS: frozenset[str] = frozenset(
    ("Oil, Soybean", "Oil, Soybean (Local)")
)
# 벌크 적재 시 포함할 최소 공표 연도(달력년)
PSD_BULK_MIN_CALENDAR_YEAR = 2010

RAW_WASDE_TABLE = "raw_wasde"

CREATE_RAW_WASDE_SQL = f"""
CREATE TABLE IF NOT EXISTS {RAW_WASDE_TABLE} (
    release_date TEXT NOT NULL,
    marketing_year TEXT NOT NULL,
    wasde_soyoil_stock_to_use REAL,
    wasde_soy_prod_brazil REAL,
    wasde_world_production REAL,
    is_wasde_release_day INTEGER,
    PRIMARY KEY (release_date, marketing_year)
);
"""

INSERT_WASDE_IGNORE_SQL = f"""
INSERT OR IGNORE INTO {RAW_WASDE_TABLE}
    (release_date, marketing_year, wasde_soyoil_stock_to_use,
     wasde_soy_prod_brazil, wasde_world_production, is_wasde_release_day)
VALUES (?, ?, ?, ?, ?, ?);
"""

USER_AGENT = "Mozilla/5.0 (compatible; soybean-oil-poc/1.0)"


def _raw_wasde_pk_column_names(conn: sqlite3.Connection) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({RAW_WASDE_TABLE})").fetchall()
    ordered = sorted((r[5], r[1]) for r in rows if r[5] > 0)
    return [name for _, name in ordered]


def _migrate_raw_wasde_single_pk_to_composite(conn: sqlite3.Connection) -> None:
    """과거 release_date 단일 PK 테이블 → (release_date, marketing_year) 복합 PK."""
    legacy = f"{RAW_WASDE_TABLE}_legacy_pk_mig"
    conn.execute(f"ALTER TABLE {RAW_WASDE_TABLE} RENAME TO {legacy}")
    conn.execute(CREATE_RAW_WASDE_SQL)
    conn.execute(
        f"""
        INSERT OR IGNORE INTO {RAW_WASDE_TABLE}
            (release_date, marketing_year, wasde_soyoil_stock_to_use,
             wasde_soy_prod_brazil, wasde_world_production, is_wasde_release_day)
        SELECT
            release_date,
            CASE
                WHEN marketing_year IS NULL OR TRIM(marketing_year) = '' THEN 'legacy'
                ELSE TRIM(marketing_year)
            END,
            wasde_soyoil_stock_to_use,
            wasde_soy_prod_brazil,
            wasde_world_production,
            is_wasde_release_day
        FROM {legacy}
        """
    )
    conn.execute(f"DROP TABLE {legacy}")
    conn.commit()


def ensure_raw_wasde_table(conn: sqlite3.Connection) -> None:
    """
    `raw_wasde` 스키마 보장.

    PK는 `(release_date, marketing_year)` 복합키(동일 공표월·다른 마케팅연도 행 보존).

    ⚠️ release_date 기준으로만 master_daily에 조인할 것.
    월중 날짜에 forward-fill 적용 시 release 이후 날짜만 채울 것.
    """
    cur = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
        (RAW_WASDE_TABLE,),
    )
    if cur.fetchone() is None:
        conn.execute(CREATE_RAW_WASDE_SQL)
        conn.commit()
        return
    pk_cols = _raw_wasde_pk_column_names(conn)
    if pk_cols == ["release_date", "marketing_year"]:
        return
    if pk_cols == ["release_date"]:
        _migrate_raw_wasde_single_pk_to_composite(conn)
        return


def _http_get_json(url: str, api_key: str | None, timeout: int = 90) -> Any:
    headers = {"User-Agent": USER_AGENT}
    if api_key:
        headers["API_KEY"] = api_key
    req = Request(url, headers=headers)
    with urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    if raw.strip().startswith('"') or raw.strip() == "Bad API Key":
        raise ValueError("PSD API: 잘못되었거나 누락된 API 키")
    return json.loads(raw)


def _normalize_psd_payload(data: Any) -> list[dict[str, Any]]:
    """응답이 list / {{data:[]}} 등 여러 형태일 때 레코드 list로 맞춘다."""
    if data is None:
        return []
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for k in ("data", "Data", "records", "Records", "results", "Results"):
            v = data.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
    return []


def _fetch_psd_path(api_key: str, path: str) -> list[dict[str, Any]]:
    """path는 '/api/psd/...' 형태."""
    url = f"https://apps.fas.usda.gov/OpenData{path}"
    data = _http_get_json(url, api_key)
    return _normalize_psd_payload(data)


def _load_commodity_attributes(api_key: str) -> dict[int, str]:
    rows = _fetch_psd_path(api_key, "/api/psd/commodityAttributes")
    out: dict[int, str] = {}
    for r in rows:
        aid = r.get("attributeId")
        name = r.get("attributeName") or r.get("attributeDescription")
        if aid is not None and name:
            try:
                out[int(aid)] = str(name)
            except (TypeError, ValueError):
                continue
    return out


def _resolve_brazil_country_code(api_key: str) -> str:
    rows = _fetch_psd_path(api_key, "/api/psd/countries")
    for r in rows:
        name = str(r.get("countryName") or r.get("name") or "").upper()
        if "BRAZIL" in name:
            c = r.get("countryCode") or r.get("code")
            if c is not None:
                return str(c).strip()
    return "BR"


def _record_key(r: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (r.get("marketYear"), r.get("calendarYear"), r.get("month"))


def _record_release_date_str(r: dict[str, Any]) -> str | None:
    for k, v in r.items():
        if v is None or not isinstance(k, str):
            continue
        lk = k.lower()
        if "release" in lk and "date" in lk:
            s = str(v).strip()
            if len(s) >= 10 and s[4] == "-" and s[7] == "-":
                return s[:10]
    return None


def _infer_release_date(
    calendar_year: int,
    month: int,
    fallback_dates: list[str],
) -> str:
    """PSD 행에 release 필드가 없을 때: 해당 월의 공표일 후보 중 첫 값, 없으면 월 12일."""
    prefix = f"{calendar_year:04d}-{int(month):02d}"
    for d in sorted(fallback_dates):
        if d.startswith(prefix):
            return d[:10]
    return f"{prefix}-12"


def _attr_name(attr_map: dict[int, str], attribute_id: Any) -> str:
    try:
        return attr_map.get(int(attribute_id), "").lower()
    except (TypeError, ValueError):
        return ""


def _pick_production_value(
    attr_name: str, value: float, current: float | None
) -> float | None:
    """국가별 레코드에서 대두 '생산' 계열 속성 선택."""
    if "production" not in attr_name and "prod" not in attr_name:
        return current
    if "oil" in attr_name or "meal" in attr_name:
        return current
    if current is None:
        return value
    return value


def _pick_stock_to_use(attr_name: str, value: float) -> bool:
    n = attr_name.replace(" ", "")
    if "stock" in attr_name and "use" in attr_name:
        return True
    if "stocks-to-use" in n or "stocktouse" in n:
        return True
    if "ratio" in attr_name and "stock" in attr_name:
        return True
    return False


def _aggregate_commodity_country_year(
    records: list[dict[str, Any]],
    attr_map: dict[int, str],
    *,
    mode: str,
) -> dict[tuple[Any, Any, Any], dict[str, Any]]:
    """
    mode: 'br_prod' | 'world_soy_prod' | 'world_oil_su'
    """
    buckets: dict[tuple[Any, Any, Any], dict[str, Any]] = defaultdict(dict)
    for r in records:
        k = _record_key(r)
        if k[1] is None or k[2] is None:
            continue
        aid = r.get("attributeId")
        an = _attr_name(attr_map, aid)
        try:
            val = float(r["value"])
        except (KeyError, TypeError, ValueError):
            continue
        rel = _record_release_date_str(r)
        if mode == "br_prod":
            new_v = _pick_production_value(an, val, buckets[k].get("wasde_soy_prod_brazil"))
            if new_v is not None:
                buckets[k]["wasde_soy_prod_brazil"] = new_v
        elif mode == "world_soy_prod":
            new_v = _pick_production_value(an, val, buckets[k].get("wasde_world_production"))
            if new_v is not None:
                buckets[k]["wasde_world_production"] = new_v
        elif mode == "world_oil_su":
            if _pick_stock_to_use(an, val):
                buckets[k]["wasde_soyoil_stock_to_use"] = val
        if rel:
            buckets[k]["release_date"] = rel
        buckets[k]["marketYear"] = r.get("marketYear")
        buckets[k]["calendarYear"] = r.get("calendarYear")
        buckets[k]["month"] = r.get("month")
    return buckets


def _merge_wasde_buckets(
    br: dict[tuple[Any, Any, Any], dict[str, Any]],
    wsoy: dict[tuple[Any, Any, Any], dict[str, Any]],
    woil: dict[tuple[Any, Any, Any], dict[str, Any]],
    release_fallback: list[str],
) -> list[tuple]:
    """INSERT용 튜플 리스트."""
    keys = set(br) | set(wsoy) | set(woil)
    rows_out: list[tuple] = []
    for k in sorted(keys, key=lambda x: (str(x[0] or ""), int(x[1] or 0), int(x[2] or 0))):
        merged: dict[str, Any] = {}
        for src in (br, wsoy, woil):
            if k in src:
                merged.update(src[k])
        cy = merged.get("calendarYear")
        mo = merged.get("month")
        my = merged.get("marketYear")
        if cy is None or mo is None:
            continue
        try:
            cy_i = int(cy)
            mo_i = int(mo)
        except (TypeError, ValueError):
            continue
        rel = merged.get("release_date")
        if not rel:
            rel = _infer_release_date(cy_i, mo_i, release_fallback)
        my_s = str(my) if my is not None else ""
        su = merged.get("wasde_soyoil_stock_to_use")
        brp = merged.get("wasde_soy_prod_brazil")
        wp = merged.get("wasde_world_production")
        rows_out.append(
            (
                rel,
                my_s,
                float(su) if su is not None else None,
                float(brp) if brp is not None else None,
                float(wp) if wp is not None else None,
                1,
            )
        )
    return rows_out


def load_wasde_from_usda_api(conn: sqlite3.Connection) -> int:
    """
    USDA FAS PSD Open Data API로 대두(commodity 2222000)·대두유(4235000) 데이터를 시도한다.

    `.env` 또는 환경변수 `USDA_OPEN_DATA_API_KEY` 필요 (발급: PSD Open Data Web).
    API 경로에는 PSD 숫자 품목 코드를 사용한다 — 월·마케팅연도별 레코드에서 다음을 추출해 병합한다.

    - marketing_year, month (마케팅연도·월은 DB의 marketing_year·release_date로 반영)
    - release_date (행에 있으면 사용, 없으면 dataReleaseDates 또는 YYYY-MM-12)
    - wasde_soy_prod_brazil: 2222000 + 브라질 + Production 계열
    - wasde_world_production: 2222000 + world + Production 계열
    - wasde_soyoil_stock_to_use: 4235000 + world + Stocks-to-use 계열 속성명 매칭

    API 실패·키 없음: `raw_wasde` 테이블만 생성하고 경고 후 0 반환.
    """
    ensure_raw_wasde_table(conn)
    api_key = (os.environ.get(ENV_USDA_API_KEY) or "").strip()
    if not api_key:
        warnings.warn(
            f"{ENV_USDA_API_KEY} 미설정 — PSD API 호출 생략. "
            f"키 발급: {PSD_REGISTRATION_URL} | 다운로드 UI: {PSD_DOWNLOADS_URL}",
            UserWarning,
            stacklevel=2,
        )
        return 0

    try:
        attr_map = _load_commodity_attributes(api_key)
        if not attr_map:
            raise RuntimeError("commodityAttributes 응답이 비었습니다.")

        brazil = _resolve_brazil_country_code(api_key)
        release_rows = _fetch_psd_path(
            api_key,
            f"/api/psd/commodity/{COMMODITY_SOYBEANS}/dataReleaseDates",
        )
        release_fallback: list[str] = []
        for x in release_rows:
            if isinstance(x, str) and len(x) >= 10:
                release_fallback.append(x[:10])
            elif isinstance(x, dict):
                for v in x.values():
                    if isinstance(v, str) and len(v) >= 10 and v[4] == "-":
                        release_fallback.append(v[:10])
        release_fallback = sorted(set(release_fallback))

        y_end = max(2026, __import__("datetime").date.today().year + 1)
        br_all: dict[tuple, dict[str, Any]] = {}
        wsoy_all: dict[tuple, dict[str, Any]] = {}
        woil_all: dict[tuple, dict[str, Any]] = {}

        for mky in range(2019, y_end + 1):
            path_br = (
                f"/api/psd/commodity/{COMMODITY_SOYBEANS}/country/{brazil}/year/{mky}"
            )
            path_ws = f"/api/psd/commodity/{COMMODITY_SOYBEANS}/world/year/{mky}"
            path_wo = f"/api/psd/commodity/{COMMODITY_SOYBEAN_OIL}/world/year/{mky}"
            rec_br = _fetch_psd_path(api_key, path_br)
            rec_ws = _fetch_psd_path(api_key, path_ws)
            rec_wo = _fetch_psd_path(api_key, path_wo)
            for k, v in _aggregate_commodity_country_year(
                rec_br, attr_map, mode="br_prod"
            ).items():
                br_all[k] = v
            for k, v in _aggregate_commodity_country_year(
                rec_ws, attr_map, mode="world_soy_prod"
            ).items():
                wsoy_all[k] = v
            for k, v in _aggregate_commodity_country_year(
                rec_wo, attr_map, mode="world_oil_su"
            ).items():
                woil_all[k] = v

        batch = _merge_wasde_buckets(br_all, wsoy_all, woil_all, release_fallback)
        batch = [b for b in batch if any(b[i] is not None for i in (2, 3, 4))]

        if not batch:
            warnings.warn(
                "PSD API는 성공했으나 대두(2222000)/대두유(4235000) 파싱 결과 행이 없습니다. "
                "commodity 코드·속성명 매핑을 확인하거나 CSV 적재를 사용하세요.",
                UserWarning,
                stacklevel=2,
            )
            return 0

        conn.executemany(INSERT_WASDE_IGNORE_SQL, batch)
        conn.commit()
        return len(batch)
    except (HTTPError, URLError, ValueError, json.JSONDecodeError, RuntimeError, OSError) as e:
        warnings.warn(
            f"USDA PSD API 실패 — 빈 테이블만 유지합니다: {type(e).__name__}: {e}",
            UserWarning,
            stacklevel=2,
        )
        return 0


def _psd_bulk_column_map(columns: list[str]) -> dict[str, str]:
    """소문자 키 → 실제 열 이름."""
    return {str(c).strip().lower(): str(c).strip() for c in columns}


def _is_psd_oilseeds_bulk_format(col_map: dict[str, str]) -> bool:
    """USDA PSD Oilseeds 벌크 CSV(속성별 롱 포맷) 여부."""
    need = ("commodity_description", "country_name", "market_year", "calendar_year", "month")
    return all(k in col_map for k in need) and (
        "attribute_description" in col_map or "attribute_id" in col_map
    ) and "value" in col_map


def _soybean_oil_commodity_mask(series: pd.Series) -> pd.Series:
    """PSD 벌크에서 대두유 품목만 (표기 정확 일치 — Meal/Oilseed/팜 등 혼입 방지)."""
    return series.astype(str).str.strip().isin(SOYBEAN_OIL_PSD_DESCRIPTIONS)


def _ingest_psd_oilseeds_bulk_csv(
    path: Path,
    col_map: dict[str, str],
    conn: sqlite3.Connection,
) -> int:
    """
    PSD 벌크 CSV에서 대두유·브라질 생산·세계 합산 지표를 추출해 적재.

    - 품목: `Oil, Soybean`, `Oil, Soybean (Local)` 만 (진단: Oilseed/Meal은 'Oil' 부분일치로 제외).
    - 세계 지표: 동일 (Market_Year, Calendar_Year, Month)에서 전국가 합산.
    - 행 단위: 각 (MY, CY, Month)마다 1행; `release_date` 단일 PK 시 월당 1MY만 남기던 로직은 폐기.
      테이블 PK는 `(release_date, marketing_year)`.
    - 달력연도 `Calendar_Year >= PSD_BULK_MIN_CALENDAR_YEAR` 만 적재.
    """
    c_commodity = col_map["commodity_description"]
    c_country = col_map["country_name"]
    c_my = col_map["market_year"]
    c_cy = col_map["calendar_year"]
    c_mo = col_map["month"]
    c_attr = col_map.get("attribute_description") or col_map.get("attribute_id")
    c_val = col_map["value"]

    raw_total_rows = 0
    kept_oil_rows = 0
    kept_oil_rows_after_year = 0

    world_prod: dict[tuple[int, int, int], float] = defaultdict(float)
    world_end: dict[tuple[int, int, int], float] = defaultdict(float)
    world_dom: dict[tuple[int, int, int], float] = defaultdict(float)
    brazil_prod: dict[tuple[int, int, int], float] = {}

    chunksize = 200_000
    for chunk in pd.read_csv(path, chunksize=chunksize):
        raw_total_rows += len(chunk)
        chunk[c_val] = pd.to_numeric(chunk[c_val], errors="coerce")
        oil = chunk.loc[_soybean_oil_commodity_mask(chunk[c_commodity])].copy()
        kept_oil_rows += len(oil)
        if oil.empty:
            continue
        oil = oil.loc[pd.to_numeric(oil[c_cy], errors="coerce") >= PSD_BULK_MIN_CALENDAR_YEAR].copy()
        kept_oil_rows_after_year += len(oil)
        if oil.empty:
            continue
        attr_l = oil[c_attr].astype(str).str.strip().str.lower()
        oil["_attr"] = attr_l

        def _add_sum(mask: pd.Series, target: dict[tuple[int, int, int], float]) -> None:
            sub = oil.loc[mask]
            if sub.empty:
                return
            g = sub.groupby([c_my, c_cy, c_mo], sort=False)[c_val].sum()
            for (my, cy, mo), val in g.items():
                if pd.isna(val):
                    continue
                k = (int(my), int(cy), int(mo))
                target[k] += float(val)

        _add_sum(oil["_attr"] == "production", world_prod)
        _add_sum(oil["_attr"] == "ending stocks", world_end)
        _add_sum(oil["_attr"] == "domestic consumption", world_dom)

        br_m = (
            oil[c_country]
            .astype(str)
            .str.contains("Brazil", case=False, na=False)
            & (oil["_attr"] == "production")
        )
        sub_br = oil.loc[br_m]
        if not sub_br.empty:
            gbr = sub_br.groupby([c_my, c_cy, c_mo], sort=False)[c_val].sum()
            for (my, cy, mo), val in gbr.items():
                if pd.isna(val):
                    continue
                k = (int(my), int(cy), int(mo))
                brazil_prod[k] = float(val)

    all_keys = set(world_prod) | set(world_end) | set(world_dom) | set(brazil_prod)
    keys_use = sorted(
        (k for k in all_keys if k[1] >= PSD_BULK_MIN_CALENDAR_YEAR),
        key=lambda t: (t[1], t[2], t[0]),
    )

    print(
        f"[{RAW_WASDE_TABLE}] PSD 벌크 필터 진단: 원본 {raw_total_rows:,}행 → "
        f"대두유 품목 {kept_oil_rows:,}행 → CY>={PSD_BULK_MIN_CALENDAR_YEAR} "
        f"{kept_oil_rows_after_year:,}행"
    )

    rows: list[tuple] = []
    for my, cy, mo in keys_use:
        k = (my, cy, mo)
        rd = f"{int(cy):04d}-{int(mo):02d}-10"
        my_s = str(my)
        br_v = brazil_prod.get(k)
        wp = world_prod.get(k)
        dom = world_dom.get(k, 0.0)
        end = world_end.get(k, 0.0)
        su: float | None
        if dom and dom != 0:
            su = end / dom
        else:
            su = None
        if not any(x is not None for x in (su, br_v, wp)):
            continue
        rows.append((rd, my_s, su, br_v, wp, 1))

    conn.executemany(INSERT_WASDE_IGNORE_SQL, rows)
    conn.commit()

    if rows:
        years = [int(r[0][:4]) for r in rows]
        y0, y1 = min(years), max(years)
        my_vals = [int(r[1]) for r in rows if str(r[1]).strip().isdigit()]
        my_rng = f"{min(my_vals)}–{max(my_vals)}" if my_vals else "—"
        n_br = sum(1 for r in rows if r[3] is not None)
        print(
            f"[{RAW_WASDE_TABLE}] PSD 벌크 적재: {len(rows)}행 "
            f"(Brazil Production 비NULL {n_br}행) | "
            f"Calendar_Year: {y0}–{y1} | Market_Year: {my_rng}"
        )
    else:
        print(
            f"[{RAW_WASDE_TABLE}] PSD 벌크 적재: 0행 "
            f"(대두유·CY>={PSD_BULK_MIN_CALENDAR_YEAR}·지표 없음)"
        )
    return len(rows)


def _ingest_simple_wasde_csv(path: Path, conn: sqlite3.Connection) -> int:
    """기존 플랫 CSV (열당 하나의 값이 이미 계산된 형태)."""
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c.lower(): c for c in df.columns}

    def col(*names: str) -> pd.Series | None:
        for n in names:
            if n.lower() in lower:
                return df[lower[n.lower()]]
        return None

    rel = col("release_date", "releasedate", "Release Date")
    if rel is None:
        raise ValueError("CSV에 release_date 열이 필요합니다 (또는 PSD 벌크 형식 열을 확인하세요).")
    my = col("marketing_year", "market_year", "Marketing Year")
    su = col("wasde_soyoil_stock_to_use", "stock_to_use", "soyoil_stock_to_use")
    br = col("wasde_soy_prod_brazil", "prod_brazil", "brazil_production")
    wp = col("wasde_world_production", "world_production")
    fl = col("is_wasde_release_day", "is_release_day")

    n = len(df)
    rows: list[tuple] = []
    for i in range(n):
        rd = str(rel.iloc[i]).strip()[:10]
        if not rd or rd.lower() == "nan":
            continue
        m_y = "" if my is None or pd.isna(my.iloc[i]) else str(my.iloc[i]).strip()

        def fnum(s: pd.Series | None) -> float | None:
            if s is None:
                return None
            v = s.iloc[i]
            if pd.isna(v):
                return None
            return float(v)

        f_long = fnum(su)
        f_br = fnum(br)
        f_wp = fnum(wp)
        is_day = 1
        if fl is not None and not pd.isna(fl.iloc[i]):
            is_day = int(float(fl.iloc[i]))
        rows.append((rd, m_y, f_long, f_br, f_wp, is_day))

    conn.executemany(INSERT_WASDE_IGNORE_SQL, rows)
    conn.commit()
    return len(rows)


def load_wasde_from_csv(filepath: str | Path, conn: sqlite3.Connection) -> int:
    """
    CSV를 `raw_wasde`에 INSERT OR IGNORE.

    - **PSD Oilseeds 벌크** (`Commodity_Description`, …):
      품목은 `Oil, Soybean` / `Oil, Soybean (Local)` 만;
      `Calendar_Year >= PSD_BULK_MIN_CALENDAR_YEAR`(기본 2019) 인 (MY, CY, Month)마다 1행.
      세계 합산·Brazil Production·ending/domestic 비율로 지표 채움.
      `release_date` = 해당 월 10일(공표일 근사); PK는 `(release_date, marketing_year)`.
    - **플랫 CSV** (기존): release_date, marketing_year, 지표 열이 직접 있는 형식.

    적재 직후 컬럼 목록과 필터 진단(원본→품목→연도)을 출력한다.

    ⚠️ release_date 기준으로만 master_daily에 조인할 것.

    Returns
    -------
    int
        적재 시도 행 수.
    """
    ensure_raw_wasde_table(conn)
    path = Path(filepath)
    if not path.is_file():
        raise FileNotFoundError(f"CSV 없음: {path.resolve()}")

    header = pd.read_csv(path, nrows=0)
    cols_list = [str(c).strip() for c in header.columns]
    print("CSV columns:", cols_list)

    col_map = _psd_bulk_column_map(cols_list)
    if _is_psd_oilseeds_bulk_format(col_map):
        if "attribute_description" not in col_map:
            raise ValueError(
                "PSD 벌크 형식인데 attribute_description 열이 없습니다. "
                "Ending Stocks / Domestic Consumption / Production 매핑에 필요합니다."
            )
        return _ingest_psd_oilseeds_bulk_csv(path, col_map, conn)
    return _ingest_simple_wasde_csv(path, conn)


def load_wasde_to_sqlite(
    db_path: str | Path | None = None,
    *,
    csv_path: str | Path | None = None,
) -> None:
    """편의: DB 열고 API 시도 후, csv_path 있으면 CSV도 적재."""
    root = Path(__file__).resolve().parents[2]
    path = Path(db_path) if db_path is not None else root / "data" / "db" / "soybean.db"
    path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(path) as conn:
        n_api = load_wasde_from_usda_api(conn)
        n_csv = 0
        if csv_path is not None:
            n_csv = load_wasde_from_csv(csv_path, conn)
        print(f"[{RAW_WASDE_TABLE}] API 행: {n_api}, CSV 행: {n_csv} | DB: {path.resolve()}")


if __name__ == "__main__":
    csv_default = _PROJECT_ROOT / "data" / "raw" / "wasde_psd.csv"
    db_default = _PROJECT_ROOT / "data" / "db" / "soybean.db"
    db_default.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_default) as conn:
        load_wasde_from_csv(csv_default, conn)
