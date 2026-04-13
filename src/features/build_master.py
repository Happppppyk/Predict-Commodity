from __future__ import annotations

import re
import sqlite3
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

try:
    from statsmodels.tsa.stattools import adfuller
except ImportError:  # pragma: no cover
    adfuller = None  # type: ignore[misc, assignment]

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "features.yaml"
DEFAULT_DB_PATH = Path(__file__).resolve().parents[2] / "data" / "db" / "soybean.db"


def load_feature_config(config_path: str | Path | None = None) -> dict:
    """features.yaml에서 전체 설정을 로드한다."""
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _read_sql(conn: sqlite3.Connection, sql: str) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn)


def _to_datetime_index(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col])
    return out.set_index(date_col).sort_index()


def _zl_futures_date_index(conn: sqlite3.Connection) -> pd.DatetimeIndex:
    """대두유(ZL=F) 거래일 목록 = master_daily 일별 뼈대."""
    sql = """
    SELECT date FROM raw_price_futures
    WHERE commodity = 'ZL=F'
    ORDER BY date
    """
    d = _read_sql(conn, sql)
    if d.empty:
        return pd.DatetimeIndex([])
    return pd.DatetimeIndex(pd.to_datetime(d["date"], errors="coerce").dropna())


def load_price_features(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    ZL=F 대두유 선물 종가 기반 가격·파생 피처.

    ADF 검정 결과는 p-value 출력 및 경고만 (변환 없음).
    """
    sql = """
    SELECT date, close AS price_close
    FROM raw_price_futures
    WHERE commodity = 'ZL=F'
    ORDER BY date
    """
    df = _read_sql(conn, sql)
    if df.empty:
        return pd.DataFrame()
    df = _to_datetime_index(df)
    c = df["price_close"].astype(float)

    out = pd.DataFrame(index=df.index)
    out["price_close"] = c
    for n in (1, 3, 7, 14, 30):
        out[f"price_lag_{n}"] = c.shift(n)
    for w in (7, 14, 30):
        out[f"price_ma_{w}"] = c.rolling(w, min_periods=1).mean()
    out["price_to_ma7_ratio"] = c / out["price_ma_7"].replace(0, np.nan)
    out["price_to_ma14_ratio"] = c / out["price_ma_14"].replace(0, np.nan)
    out["price_to_ma30_ratio"] = c / out["price_ma_30"].replace(0, np.nan)

    out["return_1d"] = c.pct_change(1)
    out["return_3d"] = c.pct_change(3)
    out["return_7d"] = c.pct_change(7)
    out["return_14d"] = c.pct_change(14)

    r1 = out["return_1d"]
    out["volatility_5d"] = r1.rolling(5, min_periods=2).std()
    out["volatility_10d"] = r1.rolling(10, min_periods=3).std()
    out["volatility_20d"] = r1.rolling(20, min_periods=5).std()

    s = c.dropna()
    if adfuller is not None and len(s) > 20:
        try:
            stat, pvalue, *_ = adfuller(s.values, autolag="AIC")
            print(
                f"[ADF] price_close: ADF={stat:.4f}, p-value={pvalue:.4g}. "
                f"p<0.05이면 단위근 H0 기각(정상성 증거). 변환은 적용하지 않음."
            )
            if pvalue > 0.05:
                warnings.warn(
                    "price_close ADF: p>=0.05 — 비정상 가능. 모델 단에서 차분 등 검토.",
                    UserWarning,
                    stacklevel=2,
                )
        except Exception as e:  # pragma: no cover
            warnings.warn(f"ADF 검정 실패: {e}", UserWarning, stacklevel=2)
    elif adfuller is None:
        warnings.warn(
            "statsmodels 미설치 — ADF 생략. pip install statsmodels 권장.",
            UserWarning,
            stacklevel=2,
        )

    return out


def load_exchange_features(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    BRL=X 환율. `is_interpolated=1`인 날도 값은 그대로 두고 플래그만 유지한다.
    """
    sql = """
    SELECT date, usd_brl_close, is_interpolated
    FROM raw_exchange_rate
    ORDER BY date
    """
    df = _read_sql(conn, sql)
    if df.empty:
        return pd.DataFrame()
    df = _to_datetime_index(df)
    x = df["usd_brl_close"].astype(float)
    out = pd.DataFrame(index=df.index)
    out["usd_brl_close"] = x
    out["is_interpolated"] = df["is_interpolated"].fillna(0).astype(int)
    out["usd_brl_lag_1"] = x.shift(1)
    out["usd_brl_return_7d"] = x.pct_change(7)
    rx = x.pct_change(1)
    out["usd_brl_volatility_14d"] = rx.rolling(14, min_periods=5).std()
    return out


def load_crude_features(conn: sqlite3.Connection) -> pd.DataFrame:
    """WTI(CL=F) 종가 및 파생."""
    sql = """
    SELECT date, close AS wti_close
    FROM raw_crude_oil
    WHERE commodity = 'CL=F'
    ORDER BY date
    """
    df = _read_sql(conn, sql)
    if df.empty:
        sql2 = """
        SELECT date, close AS wti_close
        FROM raw_crude_oil
        ORDER BY date
        """
        df = _read_sql(conn, sql2)
    if df.empty:
        return pd.DataFrame()
    df = _to_datetime_index(df)
    w = df["wti_close"].astype(float)
    out = pd.DataFrame(index=df.index)
    out["wti_close"] = w
    out["wti_lag_1"] = w.shift(1)
    out["wti_return_7d"] = w.pct_change(7)
    rw = w.pct_change(1)
    out["wti_volatility_14d"] = rw.rolling(14, min_periods=5).std()
    return out


def load_crush_spread_features(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    CBOT 단위를 달러/톤으로 맞춘 압착 마진(톤당).

    - ZL (대두유): cents/lb → $/톤 = × 22.0462
    - ZM (대두박): $/단톤 → $/톤 = × 1.10231
    - ZS (대두): cents/부셸 → $/톤 = × 36.744 / 100

    대두 1톤 압착 산출 비율:
    - 대두유 0.179톤
    - 대두박 0.765톤

    ``crush_spread`` = (ZL_per_ton×0.179) + (ZM_per_ton×0.765) − ZS_per_ton

    인덱스는 세 종목 **교집합** 날짜; master 조인 시 left join으로 뼈대 외 날짜는 NaN.
    """
    q_zl = """
    SELECT date, close AS zl_close
    FROM raw_price_futures
    WHERE commodity = 'ZL=F'
    ORDER BY date
    """
    q_zs = """
    SELECT date, close AS zs_close
    FROM raw_soybean_futures
    WHERE commodity = 'ZS=F'
    ORDER BY date
    """
    q_zm = """
    SELECT date, close AS zm_close
    FROM raw_soymeal_futures
    WHERE commodity = 'ZM=F'
    ORDER BY date
    """
    zl = _to_datetime_index(_read_sql(conn, q_zl))
    zs = _to_datetime_index(_read_sql(conn, q_zs))
    zm = _to_datetime_index(_read_sql(conn, q_zm))
    if zl.empty or zs.empty or zm.empty:
        return pd.DataFrame()

    m = zl.join(zs, how="inner").join(zm, how="inner")
    soyoil = m["zl_close"].astype(float)
    soymeal = m["zm_close"].astype(float)
    soybean = m["zs_close"].astype(float)
    soyoil_per_ton = soyoil * 22.0462
    soymeal_per_ton = soymeal * 1.10231
    soybean_per_ton = soybean * 36.744 / 100.0
    m["crush_spread"] = (
        soyoil_per_ton * 0.179
        + soymeal_per_ton * 0.765
        - soybean_per_ton
    )
    m["zl_close"] = soyoil
    m["zm_close"] = soymeal
    m["zs_close"] = soybean
    out = pd.DataFrame(index=m.index)
    out["crush_spread"] = m["crush_spread"]
    out["crush_spread_ma7"] = out["crush_spread"].rolling(7, min_periods=1).mean()
    out["crush_spread_lag_1"] = out["crush_spread"].shift(1)
    out["zl_close"] = m["zl_close"]
    out["zm_close"] = m["zm_close"]
    out["zs_close"] = m["zs_close"]
    return out


def load_palm_features(conn: sqlite3.Connection) -> pd.DataFrame:
    """팜유(CPOc1 등) 종가·파생. palm_soyoil_spread는 master 빌드 시 price_close와 계산."""
    sql = """
    SELECT date, close AS palm_close
    FROM raw_palm_oil
    WHERE commodity = 'CPOc1'
    ORDER BY date
    """
    df = _read_sql(conn, sql)
    if df.empty:
        sql2 = """
        SELECT date, close AS palm_close
        FROM raw_palm_oil
        ORDER BY date
        """
        df = _read_sql(conn, sql2)
    if df.empty:
        return pd.DataFrame()
    df = _to_datetime_index(df)
    p = df["palm_close"].astype(float)
    out = pd.DataFrame(index=df.index)
    out["palm_close"] = p
    out["palm_lag_1"] = p.shift(1)
    out["palm_return_7d"] = p.pct_change(7)
    return out


def load_cftc_features(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    CFTC 비상업 순포지션.

    ⚠️ 반드시 `release_date` 기준으로 정리한 뒤 ZL 거래일 인덱스에 reindex + forward-fill.
    ⚠️ `report_date` 기준 조인·정렬 금지 (Look-ahead bias).
    """
    daily_index = _zl_futures_date_index(conn)
    cols = [
        "cftc_noncomm_net",
        "cftc_noncomm_net_chg_1w",
        "cftc_long_short_ratio",
    ]
    if len(daily_index) == 0:
        return pd.DataFrame()

    sql = """
    SELECT release_date, noncomm_long, noncomm_short, noncomm_net
    FROM raw_cftc
    ORDER BY release_date
    """
    df = _read_sql(conn, sql)
    if df.empty:
        return pd.DataFrame(np.nan, index=daily_index, columns=cols)

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df = df.dropna(subset=["release_date"])
    df = df.sort_values("release_date").drop_duplicates(subset=["release_date"], keep="last")
    df = df.set_index("release_date")

    long_ = df["noncomm_long"].astype(float)
    short_ = df["noncomm_short"].astype(float)
    net = df["noncomm_net"].astype(float)

    weekly = pd.DataFrame(index=df.index)
    weekly["cftc_noncomm_net"] = net
    weekly["cftc_noncomm_net_chg_1w"] = net.diff(1)
    weekly["cftc_long_short_ratio"] = long_ / short_.replace(0, np.nan)

    daily_index = daily_index.sort_values()
    return weekly.reindex(daily_index).ffill()


def _marketing_year_sort_key(val: object) -> int:
    """raw_wasde.marketing_year 비교용 (큰 값 = 더 최신 작황연도)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return -(10**9)
    s = str(val).strip()
    if s.isdigit():
        return int(s)
    m = re.match(r"^(\d{4})", s)
    if m:
        return int(m.group(1))
    parts = s.split("/")
    if parts and parts[0].strip().isdigit():
        return int(parts[0].strip())
    return -(10**9)


def load_wasde_features(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    WASDE/PSD 수급 지표.

    - 동일 `release_date`에 여러 `marketing_year`가 있으면 **marketing_year 정렬 키 최댓값** 행만 사용.
    - ZL 거래일 인덱스에 맞춘 뒤 `release_date` 시점 기준 **forward-fill** (미래 공표 누수 없음).
    """
    daily_index = _zl_futures_date_index(conn)
    cols_out = [
        "wasde_soyoil_stock_to_use",
        "wasde_soy_prod_brazil",
        "wasde_world_production",
    ]
    if len(daily_index) == 0:
        return pd.DataFrame()

    sql = """
    SELECT release_date, marketing_year,
           wasde_soyoil_stock_to_use, wasde_soy_prod_brazil, wasde_world_production
    FROM raw_wasde
    ORDER BY release_date, marketing_year
    """
    df = _read_sql(conn, sql)
    daily_index = daily_index.sort_values()
    if df.empty:
        return pd.DataFrame(np.nan, index=daily_index, columns=cols_out)

    df["release_date"] = pd.to_datetime(df["release_date"], errors="coerce")
    df = df.dropna(subset=["release_date"])
    if df.empty:
        return pd.DataFrame(np.nan, index=daily_index, columns=cols_out)

    df["_my_sort"] = df["marketing_year"].map(_marketing_year_sort_key)
    df = df.sort_values(["release_date", "_my_sort"], kind="mergesort")
    pick_idx = df.groupby("release_date", sort=False)["_my_sort"].idxmax()
    df = df.loc[pick_idx].drop(columns=["_my_sort", "marketing_year"], errors="ignore")

    for c in cols_out:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.set_index("release_date")[cols_out].sort_index()
    return df.reindex(daily_index).ffill()


def load_dollar_index_features(conn: sqlite3.Connection) -> pd.DataFrame:
    """``raw_dollar_index`` DXY 종가 및 파생. ZL 거래일 인덱스로 reindex 후 forward-fill."""
    sql = """
    SELECT date, dxy_close
    FROM raw_dollar_index
    ORDER BY date
    """
    df = _read_sql(conn, sql)
    if df.empty:
        return pd.DataFrame()
    df = _to_datetime_index(df)
    x = df["dxy_close"].astype(float)
    out = pd.DataFrame(index=df.index)
    out["dxy_close"] = x
    out["dxy_lag_1"] = x.shift(1)
    out["dxy_return_7d"] = x.pct_change(7)
    rx = x.pct_change(1)
    out["dxy_volatility_14d"] = rx.rolling(14, min_periods=5).std()

    daily_index = _zl_futures_date_index(conn)
    if len(daily_index) == 0:
        return pd.DataFrame()
    return out.reindex(daily_index.sort_values()).ffill()


def load_canola_features(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    ``raw_canola_oil`` 종가·파생. 0행이면 빈 DataFrame(조인 스킵).
    ZL 거래일 인덱스로 reindex 후 forward-fill.
    """
    sql = """
    SELECT date, close AS canola_close
    FROM raw_canola_oil
    WHERE commodity = 'canola'
    ORDER BY date
    """
    df = _read_sql(conn, sql)
    if df.empty:
        sql2 = """
        SELECT date, close AS canola_close
        FROM raw_canola_oil
        ORDER BY date
        """
        df = _read_sql(conn, sql2)
    if df.empty:
        return pd.DataFrame()

    df = _to_datetime_index(df)
    c = df["canola_close"].astype(float)
    out = pd.DataFrame(index=df.index)
    out["canola_close"] = c
    out["canola_lag_1"] = c.shift(1)
    out["canola_return_7d"] = c.pct_change(7)

    daily_index = _zl_futures_date_index(conn)
    if len(daily_index) == 0:
        return out
    return out.reindex(daily_index.sort_values()).ffill()


def load_sunflower_features(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    ``raw_sunflower_oil`` 월별 종가. ZL 거래일로 reindex 후 forward-fill,
    ``sunflower_return_30d`` 는 일별 인덱스 기준 30스텝 수익률(월 간격 근사).
    """
    daily_index = _zl_futures_date_index(conn)
    if len(daily_index) == 0:
        return pd.DataFrame()

    sql = """
    SELECT date, sunflower_close
    FROM raw_sunflower_oil
    ORDER BY date
    """
    df = _read_sql(conn, sql)
    if df.empty:
        return pd.DataFrame()

    df = _to_datetime_index(df)
    s = df["sunflower_close"].astype(float)
    monthly = pd.DataFrame({"sunflower_close": s}).sort_index()
    daily = monthly.reindex(daily_index.sort_values()).ffill()
    out = pd.DataFrame(index=daily.index)
    out["sunflower_close"] = daily["sunflower_close"]
    out["sunflower_return_30d"] = daily["sunflower_close"].pct_change(30)
    return out


def load_eia_features(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    ``raw_eia_biodiesel`` 월간 대두유 바이오연료 투입량(Million Pounds).
    ZL 거래일로 reindex 후 forward-fill.

    ``soybean_oil_biofuel_chg_3m``: 월별 시계열 3개월 전 대비 변화율(pct_change(3)).
    """
    daily_index = _zl_futures_date_index(conn)
    if len(daily_index) == 0:
        return pd.DataFrame()

    sql = """
    SELECT date, soybean_oil_biofuel_mmlb
    FROM raw_eia_biodiesel
    ORDER BY date
    """
    try:
        df = _read_sql(conn, sql)
    except Exception as e:
        if "no such table" in str(e).lower():
            return pd.DataFrame()
        raise
    if df.empty:
        return pd.DataFrame()

    df = _to_datetime_index(df)
    v = df["soybean_oil_biofuel_mmlb"].astype(float)
    monthly = pd.DataFrame({"soybean_oil_biofuel_mmlb": v}).sort_index()
    monthly["soybean_oil_biofuel_chg_3m"] = monthly["soybean_oil_biofuel_mmlb"].pct_change(3)
    daily = monthly.reindex(daily_index.sort_values()).ffill()
    return daily


def load_fed_rate_features(conn: sqlite3.Connection) -> pd.DataFrame:
    """``raw_fed_rate`` 월별 금리를 ZL 거래일에 reindex+ffill 후 파생 생성."""
    daily_index = _zl_futures_date_index(conn)
    if len(daily_index) == 0:
        return pd.DataFrame()

    sql = """
    SELECT date, fed_rate
    FROM raw_fed_rate
    ORDER BY date
    """
    try:
        df = _read_sql(conn, sql)
    except Exception as e:
        if "no such table" in str(e).lower():
            return pd.DataFrame()
        raise
    if df.empty:
        return pd.DataFrame()

    df = _to_datetime_index(df)
    v = df["fed_rate"].astype(float)
    monthly = pd.DataFrame({"fed_rate": v}).sort_index()
    monthly["fed_rate_chg_1m"] = monthly["fed_rate"].diff(1)
    monthly["fed_rate_chg_3m"] = monthly["fed_rate"].diff(3)
    daily = monthly.reindex(daily_index.sort_values()).ffill()
    return daily


def load_vix_features(conn: sqlite3.Connection) -> pd.DataFrame:
    """``raw_vix`` 일별 종가/파생 피처."""
    sql = """
    SELECT date, vix_close
    FROM raw_vix
    ORDER BY date
    """
    try:
        df = _read_sql(conn, sql)
    except Exception as e:
        if "no such table" in str(e).lower():
            return pd.DataFrame()
        raise
    if df.empty:
        return pd.DataFrame()

    df = _to_datetime_index(df)
    v = df["vix_close"].astype(float)
    out = pd.DataFrame(index=df.index)
    out["vix_close"] = v
    out["vix_lag_1"] = v.shift(1)
    out["vix_return_7d"] = v.pct_change(7)
    out["vix_regime"] = (v > 25.0).astype(int)

    daily_index = _zl_futures_date_index(conn)
    if len(daily_index) == 0:
        return out
    return out.reindex(daily_index.sort_values()).ffill()


def load_spot_features(conn: sqlite3.Connection) -> pd.DataFrame:
    """
    World Bank 등 월별 `raw_price_spot`. ZL 거래일 인덱스로 reindex 후 forward-fill.
    `basis`는 master에서 `price_close - spot_price_usd`로 계산.
    """
    daily_index = _zl_futures_date_index(conn)
    if len(daily_index) == 0:
        return pd.DataFrame()

    sql = """
    SELECT date, spot_price_usd
    FROM raw_price_spot
    WHERE source = 'WorldBank'
    ORDER BY date
    """
    df = _read_sql(conn, sql)
    if df.empty:
        return pd.DataFrame(np.nan, index=daily_index, columns=["spot_price_usd"])

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df.sort_values("date").drop_duplicates(subset=["date"], keep="last")
    df = df.set_index("date")
    s = pd.to_numeric(df["spot_price_usd"], errors="coerce")
    monthly = pd.DataFrame({"spot_price_usd": s}).sort_index()
    daily_index = daily_index.sort_values()
    return monthly.reindex(daily_index).ffill()


def add_season_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    월·분기·주차, 브라질 재배 달력 근사(남반구).
    - is_planting_season: 9~11월
    - is_flowering_season: 1~2월
    - is_harvest_season: 3~5월
    - days_to_next_harvest: 다음 3월 1일까지 일수
    """
    out = df.copy()
    idx = pd.DatetimeIndex(pd.to_datetime(out.index))

    out["month"] = idx.month.astype(int)
    out["week_of_year"] = idx.isocalendar().week.astype(int)
    out["quarter"] = idx.quarter.astype(int)
    out["is_planting_season"] = idx.month.isin([9, 10, 11]).astype(int)
    out["is_flowering_season"] = idx.month.isin([1, 2]).astype(int)
    out["is_harvest_season"] = idx.month.isin([3, 4, 5]).astype(int)

    def _days_to_next_march_first(ts: pd.Timestamp) -> int:
        t = pd.Timestamp(ts).normalize()
        mar = pd.Timestamp(year=t.year, month=3, day=1)
        target = mar if t <= mar else pd.Timestamp(year=t.year + 1, month=3, day=1)
        return int((target - t).days)

    out["days_to_next_harvest"] = [_days_to_next_march_first(x) for x in idx]
    return out


def add_usda_event_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    USDA/CFTC 이벤트 피처를 날짜 인덱스만으로 생성한다.

    - WASDE: 매월 둘째 화요일
    - CFTC: 매주 금요일
    """
    out = df.copy()
    idx = pd.DatetimeIndex(pd.to_datetime(out.index)).sort_values()
    if len(idx) == 0:
        return out

    min_d = idx.min().normalize()
    max_d = idx.max().normalize()

    # WASDE 둘째 화요일 일정 생성 (범위 확장)
    month_starts = pd.date_range(
        start=(min_d - pd.offsets.MonthBegin(2)).normalize(),
        end=(max_d + pd.offsets.MonthBegin(2)).normalize(),
        freq="MS",
    )
    wasde_dates: list[pd.Timestamp] = []
    for ms in month_starts:
        # monthcalendar 기반으로 둘째 화요일 계산
        cal = pd.date_range(start=ms, end=(ms + pd.offsets.MonthEnd(1)), freq="D")
        tuesdays = [d for d in cal if d.weekday() == 1]
        if len(tuesdays) >= 2:
            wasde_dates.append(pd.Timestamp(tuesdays[1]).normalize())
    wasde_idx = pd.DatetimeIndex(sorted(set(wasde_dates)))

    # 주(월~일) 기준: 해당 주에 WASDE가 있으면 1
    week_start = idx - pd.to_timedelta(idx.weekday, unit="D")
    wasde_week_starts = set(wasde_idx - pd.to_timedelta(wasde_idx.weekday, unit="D"))
    out["is_wasde_week"] = pd.Series(
        [1 if ws in wasde_week_starts else 0 for ws in week_start],
        index=out.index,
        dtype="Int64",
    )
    out["wasde_release_day"] = pd.Series(
        [1 if d.normalize() in set(wasde_idx) else 0 for d in idx],
        index=out.index,
        dtype="Int64",
    )

    # 영업일(월~금) 기준 거리
    def _bday_diff(a: pd.Timestamp, b: pd.Timestamp) -> int:
        if b <= a:
            return 0
        return max(0, len(pd.bdate_range(a + pd.Timedelta(days=1), b)))

    days_to_w: list[int] = []
    days_since_w: list[int] = []
    for d in idx:
        next_w = wasde_idx[wasde_idx >= d.normalize()]
        prev_w = wasde_idx[wasde_idx <= d.normalize()]
        if len(next_w) == 0:
            dtw = 30
        else:
            dtw = min(30, _bday_diff(d.normalize(), next_w[0].normalize()))
        if len(prev_w) == 0:
            dsw = 30
        else:
            dsw = min(30, _bday_diff(prev_w[-1].normalize(), d.normalize()))
        days_to_w.append(dtw)
        days_since_w.append(dsw)
    out["days_to_wasde"] = pd.Series(days_to_w, index=out.index, dtype="Int64")
    out["days_since_wasde"] = pd.Series(days_since_w, index=out.index, dtype="Int64")

    # CFTC: 금요일 발표
    out["is_cftc_release_day"] = pd.Series(
        (idx.weekday == 4).astype(int),
        index=out.index,
        dtype="Int64",
    )
    out["days_to_cftc"] = pd.Series(((4 - idx.weekday) % 7).astype(int), index=out.index, dtype="Int64")
    return out


# build_master_daily 완료 후 결측 요약에 쓰는 대표 신규·외생 컬럼
_NEW_FEATURE_NULL_SUMMARY_COLS: tuple[str, ...] = (
    "fed_rate",
    "vix_close",
    "palm_soyoil_spread",
    "momentum_accel_7d",
    "cftc_percentile_52w",
    "price_above_200d_ma",
)


def _print_new_feature_null_rates(df: pd.DataFrame) -> None:
    """요청한 신규·대표 피처 결측 비율(지정 5열은 항상 출력)."""
    print("[master_daily] 신규 피처 결측 비율 (지정 컬럼):")
    for c in _NEW_FEATURE_NULL_SUMMARY_COLS:
        if c not in df.columns:
            print(f"    {c}: — (컬럼 없음 / 조인 스킵)")
        else:
            frac = float(df[c].isna().mean())
            print(f"    {c}: {100.0 * frac:.1f}%")


def _print_crush_spread_stats(series: pd.Series) -> None:
    """crush_spread 기술통계 및 음수 비율 경고."""
    s = series.dropna().astype(float)
    if s.empty:
        print("[master_daily] crush_spread: 유효 값 없음 — 기술통계 생략")
        return
    neg_ratio = float((s < 0).mean()) * 100.0
    print(
        "[master_daily] crush_spread 기술통계: "
        f"mean={s.mean():.4f}, std={s.std(ddof=1):.4f}, "
        f"min={s.min():.4f}, max={s.max():.4f}, 음수 비율={neg_ratio:.1f}%"
    )
    if neg_ratio >= 20.0:
        warnings.warn(
            f"crush_spread 음수 비율이 {neg_ratio:.1f}% (>=20%) — 단위·공식 재검토 권장.",
            UserWarning,
            stacklevel=2,
        )
    else:
        print("[master_daily] crush_spread 음수 비율 20% 미만 — 정상 범위로 판단")


def _print_crush_spread_samples(df: pd.DataFrame) -> None:
    """crush_spread 계산 검증용 샘플 5행 출력."""
    need = ["zl_close", "zm_close", "zs_close", "crush_spread"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        print(f"[master_daily] crush_spread 샘플 출력 생략 (누락 컬럼: {missing})")
        return
    sample = (
        df[need]
        .dropna(subset=["crush_spread"])
        .tail(5)
        .copy()
    )
    if sample.empty:
        print("[master_daily] crush_spread 샘플 출력 생략 (유효 행 없음)")
        return
    sample.index = pd.to_datetime(sample.index, errors="coerce").strftime("%Y-%m-%d")
    print("[master_daily] crush_spread 샘플 5행 (date, ZL, ZM, ZS, crush_spread):")
    for dt, row in sample.iterrows():
        print(
            f"    {dt} | ZL={float(row['zl_close']):.4f} | "
            f"ZM={float(row['zm_close']):.4f} | ZS={float(row['zs_close']):.4f} | "
            f"crush_spread={float(row['crush_spread']):.4f}"
        )


def _print_target_t7_t28_valid_counts(df: pd.DataFrame) -> None:
    """t7 vs t28 라벨 유효 행 수 비교."""
    n7 = int(df["target_return_t7"].notna().sum()) if "target_return_t7" in df.columns else 0
    n28 = int(df["target_return_t28"].notna().sum()) if "target_return_t28" in df.columns else 0
    print(f"[master_daily] 유효 행 수: target_return_t7 → {n7:,} | target_return_t28 → {n28:,}")


def _print_master_build_summary(df: pd.DataFrame) -> None:
    """행·열·기간·결측 상위 5·타깃 상승/하락 비율."""
    n, k = len(df), len(df.columns)
    print(f"[master_daily] 총 행 수: {n:,} | 총 컬럼 수: {k}")
    if n == 0:
        return
    idx = pd.DatetimeIndex(pd.to_datetime(df.index))
    print(f"[master_daily] 날짜 범위: {idx.min().date()} ~ {idx.max().date()}")

    null_frac = df.isna().mean().sort_values(ascending=False)
    top5 = null_frac.head(5)
    print("[master_daily] 결측 비율 상위 5개 컬럼:")
    for col, frac in top5.items():
        print(f"    {col}: {100.0 * float(frac):.1f}%")

    if "target_updown_t7" in df.columns:
        u = df["target_updown_t7"]
        valid = u.dropna()
        if len(valid) > 0:
            n_up = int((valid.astype(int) == 1).sum())
            n_dn = int((valid.astype(int) == 0).sum())
            tot = n_up + n_dn
            print(
                f"[master_daily] target_updown_t7 분포: 상승(1) {n_up} ({100*n_up/tot:.1f}%) | "
                f"하락(0) {n_dn} ({100*n_dn/tot:.1f}%) | 유효 {tot}행"
            )
        else:
            print("[master_daily] target_updown_t7: 유효 값 없음")


def _print_usda_event_summary(df: pd.DataFrame) -> None:
    """USDA 이벤트 피처 요약 출력."""
    if "wasde_release_day" in df.columns:
        cnt = int(pd.to_numeric(df["wasde_release_day"], errors="coerce").fillna(0).sum())
        print(f"[master_daily] wasde_release_day 발생 횟수: {cnt}")
    if "is_wasde_week" in df.columns:
        ratio = float(pd.to_numeric(df["is_wasde_week"], errors="coerce").fillna(0).mean()) * 100.0
        print(f"[master_daily] is_wasde_week 비율: {ratio:.1f}%")


def build_master_daily(
    conn: sqlite3.Connection,
    *,
    as_of_date: str | None = None,
    feature_version: str | None = None,
    config_path: str | Path | None = None,
    persist: bool = True,
    db_path_log: str | Path | None = None,
) -> pd.DataFrame:
    """
    ZL=F 거래일을 뼈대로 exchange → crude → crush → palm → cftc → wasde → spot
    → dollar index → canola → sunflower → EIA 바이오디젤 → season 순 left join.

    조인 후:
    - palm_soyoil_spread = price_close - palm_close
    - basis = price_close - spot_price_usd
    - market_avg_price_30d = price_close.shift(1).rolling(30).mean()  # 전일까지 30거래일 평균
    - price_vs_market_avg = price_close / market_avg_price_30d - 1

    타깃 (⚠️ 미래 정보 — **피처로 사용 금지 / 학습·평가 라벨 전용**):
    - target_price_t1: t+1 거래일 종가 (shift -1)
    - target_price_t7, target_return_t7, target_updown_t7
    - target_price_t28, target_return_t28, target_updown_t28 (t+28 거래일)
    - target_thresh_t7: t+7 수익률 > 2% 이면 1 (그 외 0, 미래 미정은 NA)
    - target_thresh_t28: t+28 수익률 > 3% 이면 1 (그 외 0, 미래 미정은 NA)

    결측치는 임의 보간하지 않는다(CFTC·WASDE·spot의 ffill은 공표일 as-of 규칙).

    persist=True이면 `master_daily` 테이블을 `if_exists='replace'`로 덮어쓴다.
    """
    cfg = load_feature_config(config_path)
    version = feature_version or cfg.get("active_version", "v1")

    price = load_price_features(conn)
    if price.empty:
        warnings.warn("raw_price_futures(ZL=F) 없음 — master_daily를 만들 수 없습니다.", UserWarning)
        return pd.DataFrame()

    base = price.copy()

    for loader in (
        load_exchange_features,
        load_crude_features,
        load_crush_spread_features,
        load_palm_features,
        load_cftc_features,
        load_wasde_features,
        load_spot_features,
        load_dollar_index_features,
        load_canola_features,
        load_sunflower_features,
        load_eia_features,
        load_fed_rate_features,
        load_vix_features,
    ):
        part = loader(conn)
        if not part.empty:
            base = base.join(part, how="left")

    base = add_season_features(base)
    base = add_usda_event_features(base)

    pc = base["price_close"].astype(float)
    # Look-ahead Bias 방지: 어제까지의 30일 평균 대비 오늘 가격
    base["market_avg_price_30d"] = pc.shift(1).rolling(30).mean()
    base["price_vs_market_avg"] = pc / base["market_avg_price_30d"].replace(0, np.nan) - 1.0

    if "palm_close" in base.columns:
        base["palm_soyoil_spread"] = pc - base["palm_close"].astype(float)
    else:
        base["palm_soyoil_spread"] = np.nan

    if "spot_price_usd" in base.columns:
        base["basis"] = pc - base["spot_price_usd"].astype(float)
    else:
        base["basis"] = np.nan

    base["momentum_accel_7d"] = base["return_7d"] - base["return_14d"]

    if "cftc_noncomm_net" in base.columns:
        cftc_net = base["cftc_noncomm_net"].astype(float)
        base["cftc_percentile_52w"] = cftc_net.rolling(252, min_periods=30).apply(
            lambda x: float(pd.Series(x).rank(method="average").iloc[-1]) / float(len(x)),
            raw=False,
        )
    else:
        base["cftc_percentile_52w"] = np.nan

    # 같은 주차의 과거 평균 return_7d 대비 현재 (현재 행 제외)
    hist_same_week_mean = (
        base.groupby("week_of_year")["return_7d"]
        .transform(lambda s: s.shift(1).expanding(min_periods=1).mean())
    )
    base["seasonal_adj_return_7d"] = base["return_7d"] - hist_same_week_mean
    base["price_above_200d_ma"] = (
        pc > pc.rolling(200, min_periods=200).mean()
    ).astype("Int64")

    # ⚠️ Look-ahead: 아래 타깃은 t 이후 가격을 사용한다. 피처 열에 포함하지 말 것.
    base["target_price_t1"] = pc.shift(-1)
    base["target_price_t7"] = pc.shift(-7)
    base["target_return_t7"] = base["target_price_t7"] / pc - 1.0
    tr = base["target_return_t7"]
    base["target_updown_t7"] = pd.Series(
        np.where(tr.isna(), pd.NA, (tr > 0).astype(int)),
        index=base.index,
        dtype="Int64",
    )

    base["target_price_t28"] = pc.shift(-28)
    base["target_return_t28"] = base["target_price_t28"] / pc - 1.0
    tr28 = base["target_return_t28"]
    base["target_updown_t28"] = pd.Series(
        np.where(tr28.isna(), pd.NA, (tr28 > 0).astype(int)),
        index=base.index,
        dtype="Int64",
    )

    base["target_thresh_t7"] = pd.Series(
        np.where(tr.isna(), pd.NA, (tr > 0.02).astype(int)),
        index=base.index,
        dtype="Int64",
    )
    base["target_thresh_t28"] = pd.Series(
        np.where(tr28.isna(), pd.NA, (tr28 > 0.03).astype(int)),
        index=base.index,
        dtype="Int64",
    )

    base["feature_version"] = version

    if as_of_date is not None:
        base = base.loc[base.index <= pd.Timestamp(as_of_date)]

    base = base.replace([np.inf, -np.inf], np.nan)

    _print_master_build_summary(base)
    _print_new_feature_null_rates(base)
    _print_target_t7_t28_valid_counts(base)
    _print_usda_event_summary(base)
    if "crush_spread" in base.columns:
        _print_crush_spread_stats(base["crush_spread"])
        _print_crush_spread_samples(base)

    if persist and not base.empty:
        out = base.reset_index()
        if out.columns[0] != "date":
            out.rename(columns={out.columns[0]: "date"}, inplace=True)
        out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")
        out.to_sql("master_daily", conn, if_exists="replace", index=False)
        conn.commit()
        logp = Path(db_path_log).resolve() if db_path_log else DEFAULT_DB_PATH.resolve()
        print(f"[master_daily] SQLite 저장 완료 (replace) | {logp}")

    return base


def write_master_daily_table(
    conn: sqlite3.Connection,
    df: pd.DataFrame,
    *,
    replace: bool = True,
) -> pd.DataFrame:
    """
    이미 계산된 DataFrame을 `master_daily`에 기록.
    replace=True면 테이블 전체 치환.
    """
    if df.empty:
        warnings.warn("저장할 master_daily 행이 없습니다.", UserWarning)
        return df

    out = df.reset_index()
    if out.columns[0] != "date":
        out.rename(columns={out.columns[0]: "date"}, inplace=True)
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    mode = "replace" if replace else "append"
    out.to_sql("master_daily", conn, if_exists=mode, index=False)
    conn.commit()
    return df


def run_pipeline(config_path: str | Path | None = None, db_path: str | Path | None = None) -> pd.DataFrame:
    """DB 연결 후 master_daily 빌드·저장."""
    db = Path(db_path) if db_path else DEFAULT_DB_PATH
    cfg_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    db.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db) as conn:
        df = build_master_daily(
            conn, config_path=cfg_path, persist=True, db_path_log=db
        )
    return df


if __name__ == "__main__":
    db = DEFAULT_DB_PATH
    db.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db) as conn:
        build_master_daily(conn, persist=True, db_path_log=db)
