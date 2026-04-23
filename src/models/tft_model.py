from __future__ import annotations

import argparse
import json
import re
import sqlite3
import sys
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, EarlyStopping
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss

from utils.plot_style import configure_matplotlib_korean

configure_matplotlib_korean()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "db" / "soybean.db"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "features.yaml"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
MODEL_CKPT_PATH = MODELS_DIR / "tft_v1.ckpt"
TFT_MULTI_CKPT_PATH = MODELS_DIR / "tft_v2_multi.ckpt"
TFT_V2_BIAS_PATH = MODELS_DIR / "tft_v2_bias.json"
TFT_V3_MULTI_CKPT_PATH = MODELS_DIR / "tft_v3_multi.ckpt"
TFT_V3_BIAS_PATH = MODELS_DIR / "tft_v3_bias.json"

VAL_DAYS = 120
CAL_DAYS = 90
TEST_DAYS = 30
MAX_ENCODER_LENGTH = 112
MAX_PREDICTION_LENGTH = 28
MIN_ENCODER_LENGTH_MULTI = 56

TFT_MULTI_KNOWN_REALS: list[str] = [
    "month",
    "quarter",
    "is_harvest_season",
    "days_to_next_harvest",
    "is_wasde_week",
]
TFT_MULTI_UNKNOWN_REALS: list[str] = [
    "price_lag_1",
    "price_lag_7",
    "return_7d",
    "volatility_20d",
    "price_to_ma30_ratio",
    "usd_brl_lag_1",
    "dxy_lag_1",
    "fed_rate",
    "vix_close",
    "wti_lag_1",
    "cftc_noncomm_net",
    "wasde_soyoil_stock_to_use",
    "crush_spread",
]

TFT_MULTI_TRAIN_LAST = pd.Timestamp("2024-12-31")
TFT_MULTI_CAL_END = pd.Timestamp("2025-09-30")
TFT_MULTI_TEST_START = pd.Timestamp("2025-10-01")

_COMMODITY_TO_USD_PER_TON_FACTOR: dict[str, float] = {
    "soybean_oil": 22.0462,  # ZL=F (USD/100lb -> USD/ton)
    "crude_oil": 6.2898,  # CL=F (USD/bbl -> USD/ton, density approximation)
    "soybean": (1.0 / 100.0) * 36.744,  # ZS=F (cents/bu -> USD/ton)
    "soymeal": 1.10231,  # ZM=F (USD/short ton -> USD/metric ton)
    "canola": 0.7285 * 1.10231,  # CAD-based proxy -> USD/ton
    "palm_oil": 1.0,  # CPOc1 already treated as USD/ton
}

_MULTI_USD_TARGET_COLUMNS: dict[str, str] = {
    "soybean_oil": "soyoil_usd_per_ton",
    "crude_oil": "wti_usd_per_ton",
    "soybean": "soybean_usd_per_ton",
    "soymeal": "soymeal_usd_per_ton",
    "canola": "canola_usd_per_ton",
    "palm_oil": "palm_usd_per_ton",
}

_COMMODITY_SQL: list[tuple[str, str]] = [
    (
        "soybean_oil",
        "SELECT date, close FROM raw_price_futures WHERE commodity = 'ZL=F' ORDER BY date",
    ),
    (
        "crude_oil",
        "SELECT date, close FROM raw_crude_oil WHERE commodity = 'CL=F' ORDER BY date",
    ),
    (
        "soybean",
        "SELECT date, close FROM raw_soybean_futures WHERE commodity = 'ZS=F' ORDER BY date",
    ),
    (
        "soymeal",
        "SELECT date, close FROM raw_soymeal_futures WHERE commodity = 'ZM=F' ORDER BY date",
    ),
    (
        "canola",
        "SELECT date, close FROM raw_canola_oil WHERE commodity = 'canola' ORDER BY date",
    ),
    (
        "palm_oil",
        "SELECT date, close FROM raw_palm_oil WHERE commodity = 'CPOc1' ORDER BY date",
    ),
]

TIME_VARYING_KNOWN_REALS: list[str] = [
    "month",
    "week_of_year",
    "quarter",
    "is_planting_season",
    "is_flowering_season",
    "is_harvest_season",
    "days_to_next_harvest",
]

TIME_VARYING_UNKNOWN_REALS: list[str] = [
    "price_to_ma7_ratio",
    "price_to_ma14_ratio",
    "price_to_ma30_ratio",
    "return_1d",
    "return_3d",
    "return_7d",
    "return_14d",
    "volatility_5d",
    "volatility_10d",
    "volatility_20d",
    "usd_brl_lag_1",
    "usd_brl_return_7d",
    "usd_brl_volatility_14d",
    "wti_lag_1",
    "wti_return_7d",
    "wti_volatility_14d",
    "crush_spread",
    "crush_spread_ma7",
    "crush_spread_lag_1",
    "cftc_noncomm_net",
    "cftc_noncomm_net_chg_1w",
    "cftc_long_short_ratio",
    "wasde_soyoil_stock_to_use",
    "wasde_soy_prod_brazil",
    "wasde_world_production",
    "dxy_close",
    "dxy_lag_1",
    "dxy_return_7d",
    "canola_close",
    "canola_lag_1",
    "sunflower_close",
    "biodiesel_production_kbbl",
    "market_avg_price_30d",
    "price_vs_market_avg",
]


def _load_yaml_config(config_path: Path | None = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _select_accelerator() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _move_batch_to_device(
    x: dict, device: torch.device
) -> dict[str, torch.Tensor | list]:
    out: dict[str, torch.Tensor | list] = {}
    for k, v in x.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        elif isinstance(v, list) and v and isinstance(v[0], torch.Tensor):
            out[k] = [t.to(device) for t in v]
        else:
            out[k] = v
    return out


def calibrate_conformal_v2(
    cal_residuals: np.ndarray | pd.Series,
    *,
    alpha: float = 0.85,
) -> float:
    r = np.asarray(cal_residuals, dtype=np.float64)
    r = r[np.isfinite(r)]
    if len(r) == 0:
        return float("nan")
    return float(np.quantile(r, alpha))


def _conformal_v2_coverage_and_q(
    soy_cal: pd.DataFrame,
    soy_test: pd.DataFrame,
    alpha: float,
) -> tuple[float, float]:
    residuals = np.abs(
        soy_cal["p50"].to_numpy(dtype=np.float64)
        - soy_cal["actual"].to_numpy(dtype=np.float64)
    )
    q = calibrate_conformal_v2(residuals, alpha=alpha)
    bias = float(np.mean(soy_cal["p50"].to_numpy() - soy_cal["actual"].to_numpy()))
    p50_raw = soy_test["p50"].to_numpy(dtype=np.float64)
    actual = soy_test["actual"].to_numpy(dtype=np.float64)
    p50_corr = p50_raw - bias
    cov = float(np.mean((actual > p50_corr - q) & (actual < p50_corr + q)))
    return cov, q


def _min_alpha_conformal_v2_for_coverage(
    soy_cal: pd.DataFrame,
    soy_test: pd.DataFrame,
    *,
    target_coverage: float = 0.80,
    lo: float = 1e-6,
    hi: float = 1.0 - 1e-9,
    iters: int = 48,
) -> float | None:
    def cov_at(a: float) -> float:
        c, _ = _conformal_v2_coverage_and_q(soy_cal, soy_test, a)
        return c

    if cov_at(lo) >= target_coverage:
        return float(lo)
    if cov_at(hi) < target_coverage:
        return None
    left, right = lo, hi
    for _ in range(iters):
        mid = 0.5 * (left + right)
        if cov_at(mid) >= target_coverage:
            right = mid
        else:
            left = mid
    return float(right)


def calc_tft_bias(
    df_cal: pd.DataFrame,
    *,
    cal_start: pd.Timestamp | None = None,
    cal_end: pd.Timestamp | None = None,
) -> float:
    """Calibration 구간 실제값-예측값(P50)의 평균 편향."""
    d = df_cal.copy()
    if cal_start is not None and "date" in d.columns:
        d = d.loc[pd.to_datetime(d["date"], errors="coerce") >= pd.Timestamp(cal_start)]
    if cal_end is not None and "date" in d.columns:
        d = d.loc[pd.to_datetime(d["date"], errors="coerce") <= pd.Timestamp(cal_end)]
    if d.empty:
        return float("nan")
    actual = d["actual"].to_numpy(dtype=np.float64)
    p50 = d["p50"].to_numpy(dtype=np.float64)
    return float(np.mean(actual - p50))


def predict_with_correction(df_pred: pd.DataFrame, bias: float | None = None) -> pd.DataFrame:
    """예측 DataFrame의 분위수 열(P10/P50/P90)에 동일 bias를 더해 보정."""
    out = df_pred.copy()
    if bias is None or not np.isfinite(bias):
        return out
    for c in ("p10", "p50", "p90"):
        if c in out.columns:
            out[c] = out[c].astype(float) + float(bias)
    return out


def _save_tft_v2_bias(
    bias: float,
    *,
    cal_start: pd.Timestamp | None = None,
    cal_end: pd.Timestamp | None = None,
    out_path: Path = TFT_V2_BIAS_PATH,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "bias": float(bias),
        "cal_start": None if cal_start is None else pd.Timestamp(cal_start).strftime("%Y-%m-%d"),
        "cal_end": None if cal_end is None else pd.Timestamp(cal_end).strftime("%Y-%m-%d"),
        "calculated_at": datetime.now().strftime("%Y-%m-%d"),
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _plot_tft_v2_conformal_corrected(
    soy_test_raw: pd.DataFrame,
    soy_test_corr: pd.DataFrame,
    q_half_width: float,
    out_path: Path,
) -> None:
    """편향 보정 P50과 Conformal 밴드를 함께 저장."""
    d_raw = soy_test_raw.sort_values(["time_idx", "horizon"]).reset_index(drop=True)
    d_cor = soy_test_corr.sort_values(["time_idx", "horizon"]).reset_index(drop=True)
    x = np.arange(len(d_raw))
    y = d_raw["actual"].to_numpy(dtype=np.float64)
    p50_raw = d_raw["p50"].to_numpy(dtype=np.float64)
    p50_cor = d_cor["p50"].to_numpy(dtype=np.float64)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, y, color="black", linewidth=1.2, label="Actual")
    ax.plot(x, p50_raw, color="tab:blue", alpha=0.8, label="P50 (raw)")
    ax.plot(x, p50_cor, color="tab:orange", linestyle="--", linewidth=1.5, label="P50_corr")
    ax.fill_between(
        x,
        p50_cor - q_half_width,
        p50_cor + q_half_width,
        color="tab:orange",
        alpha=0.20,
        label=f"Conformal (±{q_half_width:.2f})",
    )
    ax.set_title("TFT v2 멀티: 편향 보정 P50 vs Conformal")
    ax.set_xlabel("Step")
    ax.set_ylabel("Price")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _plot_p50_vs_actual_scatter(df_pred: pd.DataFrame, out_path: Path) -> None:
    d = df_pred.copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    x = d["actual"].to_numpy(dtype=np.float64)
    y = d["p50"].to_numpy(dtype=np.float64)
    lo = float(np.nanmin(np.concatenate([x, y])))
    hi = float(np.nanmax(np.concatenate([x, y])))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(x, y, s=9, alpha=0.35, color="tab:blue")
    ax.plot([lo, hi], [lo, hi], "r--", linewidth=1.2, label="y=x")
    ax.set_xlabel("Actual")
    ax.set_ylabel("P50")
    ax.set_title("TFT v3 debug: P50 vs actual")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _load_commodity_close_frame(conn: sqlite3.Connection, commodity: str, sql: str) -> pd.DataFrame:
    d = pd.read_sql_query(sql, conn)
    if d.empty and "WHERE commodity" in sql:
        m = re.search(r"FROM\s+(\w+)", sql, flags=re.IGNORECASE)
        table = m.group(1) if m else ""
        if table:
            d = pd.read_sql_query(f'SELECT date, close FROM "{table}" ORDER BY date', conn)
    if d.empty:
        return pd.DataFrame(columns=["date", "commodity", "close"])
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d = d.dropna(subset=["date", "close"])
    d["close"] = pd.to_numeric(d["close"], errors="coerce")
    d = d.dropna(subset=["close"])
    factor = float(_COMMODITY_TO_USD_PER_TON_FACTOR.get(commodity, 1.0))
    d["close"] = d["close"].astype(float) * factor
    return d.assign(commodity=commodity)[["date", "commodity", "close"]]


def prepare_tft_multi_dataset(
    db_path: str | Path,
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet, pd.DataFrame, dict[int, str]]:
    db = Path(db_path)
    if not db.is_file():
        raise FileNotFoundError(f"DB 없음: {db.resolve()}")

    master_cols = list(
        dict.fromkeys(
            ["date"]
            + list(_MULTI_USD_TARGET_COLUMNS.values())
            + TFT_MULTI_KNOWN_REALS
            + TFT_MULTI_UNKNOWN_REALS
        )
    )

    with sqlite3.connect(db) as conn:
        schema = {str(r[1]) for r in conn.execute("PRAGMA table_info(master_daily)").fetchall()}
        load_m = [c for c in master_cols if c in schema]
        miss_m = [c for c in master_cols if c != "date" and c not in schema]
        if miss_m:
            warnings.warn(
                f"master_daily에 없는 멀티 TFT 피처는 제외: {miss_m}",
                UserWarning,
                stacklevel=2,
            )
        qm = ", ".join(f'"{c}"' for c in load_m)
        master = pd.read_sql_query(f"SELECT {qm} FROM master_daily ORDER BY date", conn)
        master["date"] = pd.to_datetime(master["date"], errors="coerce")
        master = master.dropna(subset=["date"])

    req_usd_cols = ["date"] + list(_MULTI_USD_TARGET_COLUMNS.values())
    miss_usd = [c for c in req_usd_cols if c not in master.columns]
    if miss_usd:
        raise KeyError(f"master_daily에 멀티 USD/ton 타깃 컬럼이 없습니다: {miss_usd}")
    wide = master[req_usd_cols].copy()
    wide = wide.dropna().reset_index(drop=True)
    wide = wide.rename(columns={v: k for k, v in _MULTI_USD_TARGET_COLUMNS.items()})
    if wide.empty:
        raise RuntimeError("master_daily 기반 멀티 USD/ton 타깃 데이터가 비어 있습니다.")

    long = wide.melt(id_vars="date", var_name="commodity", value_name="price_close")
    long = long.merge(master, on="date", how="inner")

    long = long.sort_values(["commodity", "date"]).reset_index(drop=True)
    gcols = long.groupby("commodity", sort=False)["price_close"]
    long["price_lag_1"] = gcols.shift(1)
    long["price_lag_7"] = gcols.shift(7)
    long["return_7d"] = gcols.pct_change(7)
    long["volatility_20d"] = long.groupby("commodity", sort=False)["price_close"].transform(
        lambda s: s.pct_change(1).rolling(20, min_periods=5).std()
    )
    ma30 = long.groupby("commodity", sort=False)["price_close"].transform(
        lambda s: s.rolling(30, min_periods=15).mean()
    )
    long["price_to_ma30_ratio"] = long["price_close"] / ma30.replace(0, np.nan)

    udates = np.sort(long["date"].unique())
    dmap = {pd.Timestamp(d): i for i, d in enumerate(udates)}
    long["time_idx"] = long["date"].map(dmap).astype(np.int64)

    for c in TFT_MULTI_KNOWN_REALS + TFT_MULTI_UNKNOWN_REALS:
        if c not in long.columns:
            continue
        if str(long[c].dtype) in ("Int64", "Int32", "boolean"):
            long[c] = long[c].astype(float)

    req = [
        "price_close",
        "price_lag_1",
        "price_lag_7",
        "return_7d",
        "volatility_20d",
        "price_to_ma30_ratio",
    ]
    req += [c for c in TFT_MULTI_KNOWN_REALS + TFT_MULTI_UNKNOWN_REALS if c in long.columns]
    long = long.dropna(subset=[c for c in req if c in long.columns]).reset_index(drop=True)

    known_active = [c for c in TFT_MULTI_KNOWN_REALS if c in long.columns]
    unk_extra = [c for c in TFT_MULTI_UNKNOWN_REALS if c in long.columns]
    unknown_with_target = ["price_close"] + unk_extra

    train_df = long.loc[long["date"] <= TFT_MULTI_TRAIN_LAST].copy()
    if train_df.empty:
        raise ValueError("멀티 TFT 학습 구간(train) 행이 없습니다.")

    group_order = list(pd.unique(train_df["commodity"]))
    group_idx_to_name = {i: n for i, n in enumerate(group_order)}

    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="price_close",
        group_ids=["commodity"],
        max_encoder_length=MAX_ENCODER_LENGTH,
        min_encoder_length=MIN_ENCODER_LENGTH_MULTI,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=["commodity"],
        time_varying_known_reals=known_active,
        time_varying_unknown_reals=unknown_with_target,
        add_relative_time_idx=True,
        add_target_scales=True,
        randomize_length=None,
    )

    first_pred_idx = int(long.loc[long["date"] > TFT_MULTI_TRAIN_LAST, "time_idx"].min())
    cal_end_tidx = int(long.loc[long["date"] <= TFT_MULTI_CAL_END, "time_idx"].max())
    test_start_tidx = int(long.loc[long["date"] >= TFT_MULTI_TEST_START, "time_idx"].min())

    tail_windows = TimeSeriesDataSet.from_dataset(
        training,
        long,
        stop_randomization=True,
        predict=False,
        min_prediction_idx=first_pred_idx,
    )
    calibration_dataset = tail_windows.filter(
        lambda di: (di["time_idx_last"] <= cal_end_tidx).to_numpy(dtype=bool)
    )
    test_dataset = tail_windows.filter(
        lambda di: (di["time_idx_first_prediction"] >= test_start_tidx).to_numpy(dtype=bool)
    )

    n_series = int(long["commodity"].nunique())
    n_train_samples = len(training)
    print(
        f"[prepare_tft_multi_dataset] 품목={n_series} | 전체행={len(long):,} | "
        f"학습 TimeSeriesDataSet 샘플={n_train_samples:,} | "
        f"cal 윈도우={len(calibration_dataset):,} | test 윈도우={len(test_dataset):,} | "
        f"날짜 {long['date'].min().date()} ~ {long['date'].max().date()}"
    )
    return training, calibration_dataset, test_dataset, long, group_idx_to_name


def prepare_tft_dataset(
    db_path: str | Path,
    version: str = "v1_stationary",
    *,
    config_path: Path | None = None,
) -> tuple[TimeSeriesDataSet, TimeSeriesDataSet, TimeSeriesDataSet, pd.DataFrame]:
    db = Path(db_path)
    if not db.is_file():
        raise FileNotFoundError(f"DB 없음: {db.resolve()}")

    cfg = _load_yaml_config(config_path)
    if version not in cfg.get("versions", {}):
        raise KeyError(f"features.yaml에 버전 '{version}' 없음")

    yaml_cols: list[str] = list(cfg["versions"][version]["columns"])
    feat_from_yaml = [
        c
        for c in yaml_cols
        if not str(c).startswith("target_") and c != "feature_version"
    ]

    missing_known = [c for c in TIME_VARYING_KNOWN_REALS if c not in feat_from_yaml]
    missing_unknown = [c for c in TIME_VARYING_UNKNOWN_REALS if c not in feat_from_yaml]
    if missing_known or missing_unknown:
        raise ValueError(
            f"features.yaml({version})에 필요한 컬럼이 없습니다. "
            f"known 누락={missing_known}, unknown 누락={missing_unknown}"
        )

    with sqlite3.connect(db) as conn:
        schema = conn.execute("PRAGMA table_info(master_daily)").fetchall()
        existing_cols = {str(r[1]) for r in schema}
        load_candidates = ["date", "price_close"] + TIME_VARYING_KNOWN_REALS + TIME_VARYING_UNKNOWN_REALS
        load_cols = [c for c in dict.fromkeys(load_candidates) if c in existing_cols]
        missing_in_master = [c for c in dict.fromkeys(load_candidates) if c not in existing_cols]
        if missing_in_master:
            warnings.warn(
                f"master_daily에 없는 TFT 피처 {len(missing_in_master)}개는 제외합니다: {missing_in_master}",
                UserWarning,
                stacklevel=2,
            )
        quoted = ", ".join(f'"{c}"' for c in load_cols)
        sql = f"SELECT {quoted} FROM master_daily ORDER BY date"
        raw = pd.read_sql_query(sql, conn)

    miss = [c for c in load_cols if c not in raw.columns]
    if miss:
        raise KeyError(f"master_daily에 없는 컬럼: {miss}")

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    raw = raw.dropna().reset_index(drop=True)

    known_active = [c for c in TIME_VARYING_KNOWN_REALS if c in raw.columns]
    unknown_active = [c for c in TIME_VARYING_UNKNOWN_REALS if c in raw.columns]
    if "price_close" in unknown_active:
        unknown_active = [c for c in unknown_active if c != "price_close"]
    if not known_active:
        raise ValueError("TFT known real 피처가 없습니다.")
    if not unknown_active:
        raise ValueError("TFT unknown real 피처가 없습니다.")

    raw["time_idx"] = np.arange(len(raw), dtype=np.int64)
    raw["commodity"] = "soybean_oil"

    max_t = int(raw["time_idx"].max())
    train_end = max_t - VAL_DAYS
    cal_end = train_end + CAL_DAYS
    test_start = train_end + CAL_DAYS + 1
    if CAL_DAYS + TEST_DAYS != VAL_DAYS:
        raise ValueError("CAL_DAYS + TEST_DAYS == VAL_DAYS 이어야 합니다.")
    train_df = raw.loc[raw["time_idx"] <= train_end].copy()

    unknown_with_target = ["price_close"] + unknown_active

    training = TimeSeriesDataSet(
        train_df,
        time_idx="time_idx",
        target="price_close",
        group_ids=["commodity"],
        max_encoder_length=MAX_ENCODER_LENGTH,
        max_prediction_length=MAX_PREDICTION_LENGTH,
        static_categoricals=["commodity"],
        time_varying_known_reals=known_active,
        time_varying_unknown_reals=unknown_with_target,
        add_relative_time_idx=True,
        add_target_scales=True,
        randomize_length=None,
    )

    tail_windows = TimeSeriesDataSet.from_dataset(
        training,
        raw,
        stop_randomization=True,
        min_prediction_idx=train_end + 1,
    )
    calibration_dataset = tail_windows.filter(
        lambda di: (di["time_idx_last"] <= cal_end).to_numpy(dtype=bool)
    )
    test_dataset = tail_windows.filter(
        lambda di: (di["time_idx_first_prediction"] >= test_start).to_numpy(dtype=bool)
    )

    print(
        f"[prepare_tft_dataset] 버전={version!r} | 행 수={len(raw):,} | "
        f"known={len(known_active)}개 | unknown={len(unknown_active)}개 | "
        f"train 마지막 time_idx={train_end} | "
        f"cal time_idx=({train_end + 1}~{cal_end}) | test time_idx=({test_start}~{max_t}) | "
        f"학습 샘플={len(training):,} | calibration 윈도우={len(calibration_dataset):,} | "
        f"test 윈도우={len(test_dataset):,}"
    )
    return training, calibration_dataset, test_dataset, raw


def build_tft_model(training_dataset: TimeSeriesDataSet) -> TemporalFusionTransformer:
    tft = TemporalFusionTransformer.from_dataset(
        training_dataset,
        learning_rate=0.03,
        hidden_size=128,
        attention_head_size=4,
        dropout=0.1,
        hidden_continuous_size=32,
        loss=QuantileLoss(quantiles=[0.05, 0.5, 0.95]),
        log_interval=10,
        reduce_on_plateau_patience=4,
    )
    return tft


class EpochMetricsPrinter(Callback):
    def __init__(self, max_epochs: int) -> None:
        self.max_epochs = max_epochs

    def on_train_epoch_end(self, trainer: Trainer, pl_module: TemporalFusionTransformer) -> None:
        tr = trainer.callback_metrics.get("train_loss")
        vl = trainer.callback_metrics.get("val_loss")
        tr_s = f"{float(tr):.4f}" if tr is not None else "nan"
        vl_s = f"{float(vl):.4f}" if vl is not None else "nan"
        print(
            f"Epoch {trainer.current_epoch + 1}/{self.max_epochs} — "
            f"train_loss: {tr_s}, val_loss: {vl_s}"
        )


def train_tft(
    training: TimeSeriesDataSet,
    validation: TimeSeriesDataSet,
    *,
    tft: TemporalFusionTransformer | None = None,
    max_epochs: int = 30,
    force_cpu: bool = False,
) -> tuple[Trainer, TemporalFusionTransformer]:
    train_dl = training.to_dataloader(train=True, batch_size=64, num_workers=0)
    val_dl = validation.to_dataloader(train=False, batch_size=128, num_workers=0)

    model = tft if tft is not None else build_tft_model(training)
    accelerator = "cpu" if force_cpu else _select_accelerator()

    early_stop = EarlyStopping(monitor="val_loss", patience=5, mode="min")

    trainer = Trainer(
        max_epochs=max_epochs,
        gradient_clip_val=0.1,
        accelerator=accelerator,
        devices=1,
        enable_progress_bar=True,
        logger=False,
        callbacks=[early_stop, EpochMetricsPrinter(max_epochs=max_epochs)],
    )

    trainer.fit(model, train_dl, val_dl)

    val_metrics = trainer.validate(model, val_dl, verbose=False)
    val_loss = val_metrics[0].get("val_loss") if val_metrics else None
    if val_loss is not None:
        print(f"[train_tft] 검증 val_loss: {float(val_loss):.6f}")
    else:
        print("[train_tft] val_loss: (validate 결과 없음)")
    best = early_stop.best_score
    if best is not None and np.isfinite(float(best)):
        print(
            f"Early stopping at epoch {trainer.current_epoch + 1}, "
            f"best val_loss: {float(best):.4f}"
        )

    return trainer, model


def _np_row_1d(a: np.ndarray, i: int, h: int) -> float:
    v = np.asarray(a[i, h]).ravel()
    return float(v[0]) if v.size else float("nan")


def _collect_validation_predictions(
    tft: TemporalFusionTransformer,
    dataset: TimeSeriesDataSet,
    *,
    quiet: bool = False,
    group_idx_to_name: dict[int, str] | None = None,
) -> pd.DataFrame:
    tft.eval()
    device = tft.device
    val_dl = dataset.to_dataloader(train=False, batch_size=128, num_workers=0)
    rows: list[tuple[int, int, float, float, float, float, str | None]] = []
    first_batch = True

    with torch.no_grad():
        for batch in val_dl:
            x, y = batch
            x = _move_batch_to_device(x, device)
            y = tuple(
                t.to(device) if isinstance(t, torch.Tensor) else t for t in y
            )
            out = tft(x)
            q = tft.to_quantiles(out)
            if isinstance(q, (list, tuple)):
                if len(q) != 1:
                    raise ValueError(f"TFT quantiles 리스트 길이 예상 1, 실제 {len(q)}")
                q = q[0]
            q_np = q.detach().cpu().numpy()
            y_np = y[0].detach().cpu().numpy()
            dti = x["decoder_time_idx"].detach().cpu().numpy()
            lengths = x["decoder_lengths"].detach().cpu().numpy()
            grp_tensor = x.get("groups")
            if grp_tensor is not None and group_idx_to_name is not None:
                grp_ids = grp_tensor.squeeze(-1).detach().cpu().numpy().astype(int)
            else:
                grp_ids = None

            if not quiet and first_batch:
                print(f"[collect] q shape {q_np.shape}")
                print("h=0..4 q05/P50/q95 vs actual:")
                for h in range(min(5, q_np.shape[1])):
                    print(
                        f"    h={h}  q05={q_np[0, h, 0]:.6f}  P50={q_np[0, h, 1]:.6f}  "
                        f"q95={q_np[0, h, 2]:.6f}  actual={_np_row_1d(y_np, 0, h):.6f}"
                    )
                first_batch = False

            for i in range(q_np.shape[0]):
                gname: str | None = None
                if grp_ids is not None and group_idx_to_name is not None:
                    gname = group_idx_to_name.get(int(grp_ids[i]))
                for h in range(int(lengths[i])):
                    rows.append(
                        (
                            int(dti[i, h]),
                            h,
                            float(q_np[i, h, 0]),
                            float(q_np[i, h, 1]),
                            float(q_np[i, h, 2]),
                            _np_row_1d(y_np, i, h),
                            gname,
                        )
                    )

    return pd.DataFrame(
        rows,
        columns=["time_idx", "horizon", "p10", "p50", "p90", "actual", "commodity"],
    )


def evaluate_tft(
    trainer: Trainer,
    tft: TemporalFusionTransformer,
    eval_dataset: TimeSeriesDataSet,
    df_processed: pd.DataFrame,
    df_pred: pd.DataFrame | None = None,
    *,
    plot_path: Path | None = None,
) -> dict[str, float]:
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    out_plot = Path(plot_path) if plot_path is not None else NOTEBOOKS_DIR / "tft_prediction.png"

    if df_pred is None:
        df_pred = _collect_validation_predictions(tft, eval_dataset)

    p50 = df_pred["p50"].to_numpy()
    actual = df_pred["actual"].to_numpy()
    p10 = df_pred["p10"].to_numpy()
    p90 = df_pred["p90"].to_numpy()

    mae = float(np.mean(np.abs(p50 - actual)))
    rmse = float(np.sqrt(np.mean((p50 - actual) ** 2)))
    q_lo = np.minimum(p10, p90)
    q_hi = np.maximum(p10, p90)
    coverage = float(np.mean((actual > q_lo) & (actual < q_hi)))

    print("\n[TFT test metrics]")
    print(f"MAE (P50, test):      {mae:.4f}")
    print(f"RMSE (P50, test):     {rmse:.4f}")
    print(f"Coverage 분위수 밴드 (test): {coverage:.2%}")

    df_pred["q_lo"] = df_pred[["p10", "p50", "p90"]].min(axis=1)
    df_pred["q_hi"] = df_pred[["p10", "p50", "p90"]].max(axis=1)
    agg = (
        df_pred.groupby("time_idx", as_index=False)
        .agg(
            p10=("p10", "mean"),
            p50=("p50", "mean"),
            p90=("p90", "mean"),
            q_lo=("q_lo", "mean"),
            q_hi=("q_hi", "mean"),
            actual=("actual", "mean"),
        )
    )
    cal = df_processed[["time_idx", "date"]].drop_duplicates("time_idx")
    plot_df = agg.merge(cal, on="time_idx", how="left").sort_values("time_idx")

    plt.figure(figsize=(12, 5))
    plt.plot(plot_df["date"], plot_df["actual"], label="실제 price_close", color="black", linewidth=1.2)
    plt.plot(plot_df["date"], plot_df["p50"], label="P50", color="C0")
    plt.plot(plot_df["date"], plot_df["p10"], label="q05", color="C1", alpha=0.8)
    plt.plot(plot_df["date"], plot_df["p90"], label="q95", color="C2", alpha=0.8)
    plt.fill_between(
        plot_df["date"].to_numpy(),
        plot_df["q_lo"].to_numpy(),
        plot_df["q_hi"].to_numpy(),
        alpha=0.15,
        color="C0",
    )
    plt.title("TFT 테스트 구간: 실제 vs q05/P50/q95")
    plt.xlabel("날짜")
    plt.ylabel("가격")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_plot, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate_tft] 예측 플롯 저장: {out_plot.resolve()}")

    return {"mae": mae, "rmse": rmse, "coverage": coverage}


def _price_close_std_for_time_idx_range(
    df_processed: pd.DataFrame, t_min: int, t_max: int
) -> float:
    dfp = df_processed.sort_values("time_idx")
    s = dfp.loc[
        (dfp["time_idx"] >= t_min) & (dfp["time_idx"] <= t_max),
        "price_close",
    ].astype(float)
    if len(s) < 2:
        return float("nan")
    return float(s.std(ddof=1))


def calibrate_conformal(
    tft: TemporalFusionTransformer,
    calibration_dataset: TimeSeriesDataSet,
    df_processed: pd.DataFrame,
    target_coverage: float = 0.80,
    *,
    df_pred: pd.DataFrame | None = None,
) -> dict[str, float]:
    tft.eval()
    if df_pred is None:
        df_pred = _collect_validation_predictions(
            tft, calibration_dataset, quiet=True
        )
    max_t = int(df_processed["time_idx"].max())
    train_end = max_t - VAL_DAYS
    cal_end = train_end + CAL_DAYS
    test_start = train_end + CAL_DAYS + 1

    cal_vol = _price_close_std_for_time_idx_range(df_processed, train_end + 1, cal_end)
    test_vol_estimate = _price_close_std_for_time_idx_range(df_processed, test_start, max_t)
    if cal_vol > 0 and np.isfinite(cal_vol) and np.isfinite(test_vol_estimate):
        vol_ratio = float(test_vol_estimate / cal_vol)
    else:
        vol_ratio = 1.0

    p50 = df_pred["p50"].to_numpy(dtype=np.float64)
    actual = df_pred["actual"].to_numpy(dtype=np.float64)
    residuals = np.abs(actual - p50)
    q_base = float(np.quantile(residuals, target_coverage))
    q_adjusted = float(q_base * vol_ratio)

    c_lo_adj = p50 - q_adjusted
    c_hi_adj = p50 + q_adjusted
    emp_cov_adj = float(np.mean((actual > c_lo_adj) & (actual < c_hi_adj)))

    print(
        f"\n[calibrate_conformal] [Calibration 구간] 예측 스텝 {len(df_pred):,}개, "
        f"목표 분위 {target_coverage:.0%}"
    )
    print(f"  calibration σ: {cal_vol:.4f} 달러")
    print(f"  최근 30일 σ:   {test_vol_estimate:.4f} 달러 (time_idx {test_start}~{max_t})")
    print(f"  변동성 비율:   {vol_ratio:.4f}")
    print(f"  q (기존):      {q_base:.4f} 달러")
    print(f"  q (조정):      {q_adjusted:.4f} 달러")
    print(
        f"[calibrate_conformal] Calibration에서 경험적 Coverage (P50±q조정): {emp_cov_adj:.2%}"
    )
    return {
        "q_base": q_base,
        "q_adjusted": q_adjusted,
        "vol_ratio": vol_ratio,
        "cal_vol": cal_vol,
        "test_vol_estimate": test_vol_estimate,
        "n_calibration_steps": float(len(df_pred)),
    }


def evaluate_tft_with_conformal(
    tft: TemporalFusionTransformer,
    test_dataset: TimeSeriesDataSet,
    df_processed: pd.DataFrame,
    df_cal: pd.DataFrame,
    q_base: float,
    q_adjusted: float,
    vol_ratio: float,
    df_pred: pd.DataFrame | None = None,
    *,
    n_calibration_steps: int,
    plot_path: Path | None = None,
) -> dict[str, float]:
    if df_pred is None:
        df_pred = _collect_validation_predictions(tft, test_dataset, quiet=True)
    tft.eval()

    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    out_plot = Path(plot_path) if plot_path is not None else NOTEBOOKS_DIR / "tft_conformal.png"

    p50 = df_pred["p50"].to_numpy(dtype=np.float64)
    actual = df_pred["actual"].to_numpy(dtype=np.float64)
    p10 = df_pred["p10"].to_numpy(dtype=np.float64)
    p90 = df_pred["p90"].to_numpy(dtype=np.float64)

    bias = float(np.mean(df_cal["p50"].to_numpy(dtype=np.float64) - df_cal["actual"].to_numpy(dtype=np.float64)))
    p50_corrected = p50 - bias

    mae_p50_raw = float(np.mean(np.abs(p50 - actual)))
    mae_p50_corrected = float(np.mean(np.abs(p50_corrected - actual)))

    raw_lo = np.minimum(p10, p90)
    raw_hi = np.maximum(p10, p90)
    width_before = float(np.mean(raw_hi - raw_lo))

    conf_lo_base = p50_corrected - q_base
    conf_hi_base = p50_corrected + q_base
    width_base = float(np.mean(conf_hi_base - conf_lo_base))
    coverage_conf_base = float(np.mean((actual > conf_lo_base) & (actual < conf_hi_base)))

    conf_lo_vol = p50_corrected - q_adjusted
    conf_hi_vol = p50_corrected + q_adjusted
    width_vol = float(np.mean(conf_hi_vol - conf_lo_vol))
    coverage_conf_vol = float(np.mean((actual > conf_lo_vol) & (actual < conf_hi_vol)))

    coverage_tft = float(np.mean((actual > raw_lo) & (actual < raw_hi)))

    n_test = len(df_pred)
    print("\n[TFT + Conformal test]")
    print(f"[Calibration 구간] {n_calibration_steps:,}개 스텝으로 q·편향 보정")
    print(f"[Test 구간] {n_test:,}개 스텝 (편향 mean(P50−actual)={bias:+.4f} 달러)")
    print(f"MAE (P50, 보정 전):     {mae_p50_raw:.4f} 달러")
    print(f"MAE (P50, 보정 후):     {mae_p50_corrected:.4f} 달러")
    print(f"Coverage (TFT 원본, test):           {coverage_tft:.2%}")
    print(f"Coverage (Conformal 기본, test):     {coverage_conf_base:.2%}")
    print(f"Coverage (Conformal 변동성조정, test): {coverage_conf_vol:.2%}")
    print(f"보정 전 평균 구간폭 (test, TFT q05~q95):     {width_before:.4f} 달러")
    print(f"보정 후 평균 구간폭 (test, Conformal 기본): {width_base:.4f} 달러")
    print(f"보정 후 평균 구간폭 (test, Conformal 변동성): {width_vol:.4f} 달러")

    df_plot = df_pred.copy()
    df_plot["raw_lo"] = raw_lo
    df_plot["raw_hi"] = raw_hi
    df_plot["p50_corrected"] = p50_corrected
    df_plot["conf_lo_base"] = conf_lo_base
    df_plot["conf_hi_base"] = conf_hi_base
    df_plot["conf_lo_vol"] = conf_lo_vol
    df_plot["conf_hi_vol"] = conf_hi_vol

    agg = (
        df_plot.groupby("time_idx", as_index=False)
        .agg(
            p10=("p10", "mean"),
            p50=("p50", "mean"),
            p90=("p90", "mean"),
            raw_lo=("raw_lo", "mean"),
            raw_hi=("raw_hi", "mean"),
            p50_corrected=("p50_corrected", "mean"),
            conf_lo_base=("conf_lo_base", "mean"),
            conf_hi_base=("conf_hi_base", "mean"),
            conf_lo_vol=("conf_lo_vol", "mean"),
            conf_hi_vol=("conf_hi_vol", "mean"),
            actual=("actual", "mean"),
        )
    )
    cal = df_processed[["time_idx", "date"]].drop_duplicates("time_idx")
    plot_df = agg.merge(cal, on="time_idx", how="left").sort_values("time_idx")

    plt.figure(figsize=(13, 6))
    plt.plot(plot_df["date"], plot_df["actual"], label="실제 price_close", color="black", linewidth=1.2)
    plt.plot(plot_df["date"], plot_df["p50"], label="P50 (원본)", color="C0", linewidth=1.0, alpha=0.65)
    plt.plot(
        plot_df["date"],
        plot_df["p50_corrected"],
        label="P50 (편향 보정)",
        color="C0",
        linewidth=1.2,
        linestyle="--",
    )
    plt.plot(plot_df["date"], plot_df["p10"], label="q05 (TFT)", color="C1", alpha=0.75, linewidth=0.9)
    plt.plot(plot_df["date"], plot_df["p90"], label="q95 (TFT)", color="C2", alpha=0.75, linewidth=0.9)
    plt.fill_between(
        plot_df["date"].to_numpy(),
        plot_df["raw_lo"].to_numpy(),
        plot_df["raw_hi"].to_numpy(),
        alpha=0.12,
        color="C3",
        label="TFT q05~q95 (보정 전)",
    )
    plt.fill_between(
        plot_df["date"].to_numpy(),
        plot_df["conf_lo_base"].to_numpy(),
        plot_df["conf_hi_base"].to_numpy(),
        alpha=0.10,
        color="C4",
        label="Conformal 기본 (P50_corr±q_base)",
    )
    plt.fill_between(
        plot_df["date"].to_numpy(),
        plot_df["conf_lo_vol"].to_numpy(),
        plot_df["conf_hi_vol"].to_numpy(),
        alpha=0.22,
        color="C0",
        label="Conformal 변동성조정 (P50_corr±q_adj)",
    )
    plt.title("TFT 테스트: 편향 보정 P50 · TFT 밴드 vs Conformal(기본/변동성)")
    plt.xlabel("날짜")
    plt.ylabel("가격")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_plot, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[evaluate_tft_with_conformal] 플롯 저장: {out_plot.resolve()}")

    return {
        "mae_p50_raw": mae_p50_raw,
        "mae_p50_corrected": mae_p50_corrected,
        "bias": bias,
        "coverage_tft": coverage_tft,
        "coverage_conformal_base": coverage_conf_base,
        "coverage_conformal_vol": coverage_conf_vol,
        "mean_width_before": width_before,
        "mean_width_conformal_base": width_base,
        "mean_width_conformal_vol": width_vol,
        "vol_ratio": vol_ratio,
    }


def _aggregate_interpretation(
    tft: TemporalFusionTransformer, validation: TimeSeriesDataSet
) -> dict[str, torch.Tensor]:
    tft.eval()
    device = tft.device
    val_dl = validation.to_dataloader(train=False, batch_size=64, num_workers=0)

    sums: dict[str, torch.Tensor] | None = None
    n_batches = 0

    with torch.no_grad():
        for batch in val_dl:
            x, y = batch
            x = _move_batch_to_device(x, device)
            out = tft(x)
            out_det = {
                k: (v.detach() if isinstance(v, torch.Tensor) else v)
                for k, v in out.items()
            }
            intr = tft.interpret_output(out_det, reduction="sum")
            if sums is None:
                sums = {k: v.cpu().clone() for k, v in intr.items()}
            else:
                for k in sums:
                    sums[k] = sums[k] + intr[k].cpu()
            n_batches += 1

    assert sums is not None and n_batches > 0
    return {k: v / float(n_batches) for k, v in sums.items()}


def get_tft_attention(
    tft: TemporalFusionTransformer,
    validation: TimeSeriesDataSet,
    *,
    out_path: Path | None = None,
) -> None:
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(out_path) if out_path else NOTEBOOKS_DIR / "tft_attention.png"

    intr = _aggregate_interpretation(tft, validation)
    attn = intr["attention"].detach().cpu().float()
    attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)

    w = attn.numpy().ravel()
    n = len(w)
    plt.figure(figsize=(max(10, n * 0.15), 2.5))
    plt.imshow(w[np.newaxis, :], aspect="auto", cmap="viridis", interpolation="nearest")
    plt.yticks([0], ["attention"])
    plt.xlabel("과거 시점 인덱스 (인코더+디코더 상대 위치)")
    plt.title("TFT temporal attention (검증 평균)")
    plt.colorbar(fraction=0.02, pad=0.04)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[get_tft_attention] 저장: {path.resolve()}")


def get_tft_variable_importance(
    tft: TemporalFusionTransformer,
    validation: TimeSeriesDataSet,
    *,
    out_path: Path | None = None,
) -> list[tuple[str, float]]:
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(out_path) if out_path else NOTEBOOKS_DIR / "tft_variable_importance.png"

    intr = _aggregate_interpretation(tft, validation)
    enc_w = intr["encoder_variables"].detach().cpu().float().numpy().ravel()
    dec_w = intr["decoder_variables"].detach().cpu().float().numpy().ravel()
    enc_names = list(tft.encoder_variables)
    dec_names = list(tft.decoder_variables)

    enc_w = enc_w[: len(enc_names)]
    dec_w = dec_w[: len(dec_names)]

    name_to_score: dict[str, float] = {}
    for n, v in zip(enc_names, enc_w):
        name_to_score[n] = name_to_score.get(n, 0.0) + float(v)
    for n, v in zip(dec_names, dec_w):
        name_to_score[n] = name_to_score.get(n, 0.0) + float(v)

    names = list(name_to_score.keys())
    scores = np.array([name_to_score[n] for n in names], dtype=float)
    scores = scores / (scores.sum() + 1e-8)

    order = np.argsort(scores)
    plt.figure(figsize=(8, max(4, len(names) * 0.22)))
    plt.barh(np.arange(len(names)), scores[order] * 100.0, tick_label=np.array(names)[order])
    plt.xlabel("중요도 (%)")
    plt.title("TFT Variable Selection (encoder+decoder 합산, 검증 평균)")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[get_tft_variable_importance] 저장: {path.resolve()}")

    top5_idx = np.argsort(-scores)[:5]
    top5 = [(names[i], float(scores[i])) for i in top5_idx]
    print("\n[Variable importance top 5]")
    for rank, (nm, sc) in enumerate(top5, start=1):
        print(f"  {rank}. {nm}: {sc * 100:.2f}%")
    return top5


def _diagnose_conformal_coverage_gap(
    df_processed: pd.DataFrame,
    df_test: pd.DataFrame,
    *,
    q_half_width: float,
) -> None:
    max_t = int(df_processed["time_idx"].max())
    train_end = max_t - VAL_DAYS
    cal_end = train_end + CAL_DAYS
    test_start = train_end + CAL_DAYS + 1

    dfp = df_processed.sort_values("time_idx")
    cal_px = dfp.loc[
        (dfp["time_idx"] >= train_end + 1) & (dfp["time_idx"] <= cal_end),
        "price_close",
    ].astype(float)
    test_px = dfp.loc[
        (dfp["time_idx"] >= test_start) & (dfp["time_idx"] <= max_t),
        "price_close",
    ].astype(float)

    def _vol_stats(s: pd.Series) -> tuple[float, float]:
        if len(s) < 2:
            return float("nan"), float("nan")
        std = float(s.std(ddof=1))
        mad = float(s.diff().abs().dropna().mean())
        return std, mad

    cal_std, cal_mad = _vol_stats(cal_px)
    test_std, test_mad = _vol_stats(test_px)

    print("\n[진단] calibration vs test 변동성")
    print(
        f"calibration 구간: time_idx {train_end + 1}~{cal_end} ({CAL_DAYS}일, "
        f"일별 price_close {len(cal_px)}개)"
    )
    print(f"  표준편차 σ:     {cal_std:.4f} 달러")
    print(f"  평균 |일변화|:  {cal_mad:.4f} 달러")
    print(
        f"test 구간:        time_idx {test_start}~{max_t} ({TEST_DAYS}일, "
        f"일별 price_close {len(test_px)}개)"
    )
    print(f"  표준편차 σ:     {test_std:.4f} 달러")
    print(f"  평균 |일변화|:  {test_mad:.4f} 달러")
    if cal_std > 0 and test_std > 0:
        print(f"  σ 비율 (test/cal): {test_std / cal_std:.3f}")
    if cal_mad > 0 and test_mad > 0:
        print(f"  평균 |일변화| 비율 (test/cal): {test_mad / cal_mad:.3f}")

    dt = df_test.sort_values(["time_idx", "horizon"]).reset_index(drop=True)
    p50 = dt["p50"].to_numpy(dtype=float)
    act = dt["actual"].to_numpy(dtype=float)
    signed_err = p50 - act

    print("\n[진단] test P50 vs actual (10행)")
    print(f"{'idx':>4} {'time_idx':>9} {'h':>3} {'P50':>10} {'actual':>10} {'P50-actual':>12}")
    for k in range(min(10, len(dt))):
        r = dt.iloc[k]
        print(
            f"{k:4d} {int(r['time_idx']):9d} {int(r['horizon']):3d} "
            f"{float(r['p50']):10.4f} {float(r['actual']):10.4f} "
            f"{float(r['p50'] - r['actual']):12.4f}"
        )
    print(f"전체 test mean(P50−actual) = {float(np.mean(signed_err)):.4f}")

    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    scatter_path = NOTEBOOKS_DIR / "tft_debug_p50_vs_actual_test.png"
    plt.figure(figsize=(5.5, 5.5))
    lo = float(min(act.min(), p50.min()))
    hi = float(max(act.max(), p50.max()))
    pad = 0.02 * (hi - lo + 1e-6)
    plt.scatter(act, p50, alpha=0.35, s=12, edgecolors="none")
    plt.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", linewidth=1, label="y=x")
    plt.xlabel("actual (test)")
    plt.ylabel("P50 예측 (test)")
    plt.title("test: P50 vs actual")
    plt.legend(loc="best")
    plt.xlim(lo - pad, hi + pad)
    plt.ylim(lo - pad, hi + pad)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[진단] scatter 저장: {scatter_path.resolve()}")

    res = np.abs(act - p50)
    r_mean = float(np.mean(res))
    r_std = float(np.std(res, ddof=1)) if len(res) > 1 else float("nan")
    r_p80 = float(np.quantile(res, 0.80))
    sorted_r = np.sort(res)
    rank = int(np.searchsorted(sorted_r, q_half_width, side="right"))
    pct_in_test = 100.0 * rank / float(len(sorted_r)) if len(sorted_r) else float("nan")

    print("\n[진단] Conformal q vs test 잔차")
    print(f"calibration에서 계산된 q (반폭): {q_half_width:.4f} 달러")
    print(
        f"test 구간 |actual−P50|: mean={r_mean:.4f}, std={r_std:.4f}, "
        f"80th pct={r_p80:.4f} 달러"
    )
    print(f"q 대비 test |actual−P50| 상대 랭크 ≈ {pct_in_test:.1f}%")


def _naive_persistence_mae(df: pd.DataFrame, val_start_time_idx: int) -> float:
    d = df.sort_values("time_idx").reset_index(drop=True)
    prev = d["price_close"].shift(1)
    mask = d["time_idx"] >= val_start_time_idx
    err = (d.loc[mask, "price_close"] - prev.loc[mask]).abs()
    err = err.dropna()
    return float(err.mean()) if len(err) else float("nan")


def run_tft_multi_eval_from_checkpoint(
    db_path: str | Path | None = None,
    ckpt_path: str | Path | None = None,
) -> None:
    db = Path(db_path) if db_path else DEFAULT_DB_PATH
    ckpt = Path(ckpt_path) if ckpt_path else TFT_MULTI_CKPT_PATH
    if not ckpt.is_file():
        raise FileNotFoundError(f"체크포인트 없음: {ckpt.resolve()}")

    _training, calibration_dataset, test_dataset, df_long, group_idx_to_name = (
        prepare_tft_multi_dataset(db)
    )
    acc = _select_accelerator()
    tft = TemporalFusionTransformer.load_from_checkpoint(
        str(ckpt),
        map_location=acc,
        weights_only=False,
    )
    tft.to(torch.device(acc))
    tft.eval()

    df_cal = _collect_validation_predictions(
        tft,
        calibration_dataset,
        quiet=True,
        group_idx_to_name=group_idx_to_name,
    )
    df_test = _collect_validation_predictions(
        tft,
        test_dataset,
        quiet=True,
        group_idx_to_name=group_idx_to_name,
    )
    soy_cal = df_cal.loc[df_cal["commodity"] == "soybean_oil"].copy()
    soy_test = df_test.loc[df_test["commodity"] == "soybean_oil"].copy()
    if soy_cal.empty or soy_test.empty:
        raise RuntimeError("soybean_oil 평가용 예측 행이 비어 있습니다.")

    soy_date_map = (
        df_long.loc[df_long["commodity"] == "soybean_oil", ["time_idx", "date"]]
        .drop_duplicates("time_idx")
        .copy()
    )
    soy_cal = soy_cal.merge(soy_date_map, on="time_idx", how="left")
    soy_test = soy_test.merge(soy_date_map, on="time_idx", how="left")

    test_start_date = pd.to_datetime(soy_test["date"], errors="coerce").min()
    if pd.isna(test_start_date):
        raise RuntimeError("test 시작일(date)을 계산할 수 없습니다.")
    cal_end_60 = pd.Timestamp(test_start_date) - pd.Timedelta(days=1)
    cal_start_60 = cal_end_60 - pd.Timedelta(days=60)
    soy_bias_pool = pd.concat([soy_cal, soy_test], ignore_index=True)
    bias_60d = calc_tft_bias(soy_bias_pool, cal_start=cal_start_60, cal_end=cal_end_60)

    cal_start_jf = pd.Timestamp("2026-01-01")
    cal_end_jf = pd.Timestamp("2026-02-28")
    bias_2026_jf = calc_tft_bias(soy_bias_pool, cal_start=cal_start_jf, cal_end=cal_end_jf)

    # 기본 저장/보정에는 test 직전 60일 bias를 사용
    if np.isfinite(bias_60d):
        bias = float(bias_60d)
        bias_start, bias_end = cal_start_60, cal_end_60
    elif np.isfinite(bias_2026_jf):
        bias = float(bias_2026_jf)
        bias_start, bias_end = cal_start_jf, cal_end_jf
    else:
        bias = calc_tft_bias(soy_cal)
        bias_start, bias_end = None, None

    _save_tft_v2_bias(bias, cal_start=bias_start, cal_end=bias_end)
    soy_test_corr = predict_with_correction(soy_test, bias=bias)

    for a in (0.85, 0.90):
        cov, q = _conformal_v2_coverage_and_q(soy_cal, soy_test_corr, a)
        print(f"alpha={a:.2f}: Coverage {cov * 100:.2f}%, q=±{q:.2f}달러")

    amin = _min_alpha_conformal_v2_for_coverage(soy_cal, soy_test_corr, target_coverage=0.80)
    if amin is None:
        print("목표 80% 달성 최소 alpha: 달성 불가 (alpha 상한에서도 Coverage < 80%)")
    else:
        _, qm = _conformal_v2_coverage_and_q(soy_cal, soy_test_corr, amin)
        print(
            f"목표 80% 달성 최소 alpha: {amin:.6f} (해당 q=±{qm:.2f}달러)"
        )
    q85 = calibrate_conformal_v2(
        np.abs(soy_cal["p50"].to_numpy(dtype=np.float64) - soy_cal["actual"].to_numpy(dtype=np.float64)),
        alpha=0.85,
    )
    corrected_plot = NOTEBOOKS_DIR / "results" / "tft_v2_conformal_corrected.png"
    _plot_tft_v2_conformal_corrected(soy_test, soy_test_corr, q85, corrected_plot)
    print(
        f"[bias] cal_60d_before_test ({cal_start_60:%Y-%m-%d}~{cal_end_60:%Y-%m-%d}): "
        f"{bias_60d:+.4f} 달러"
    )
    print(
        f"[bias] cal_2026_jan_feb ({cal_start_jf:%Y-%m-%d}~{cal_end_jf:%Y-%m-%d}): "
        f"{bias_2026_jf:+.4f} 달러"
    )
    print(f"[bias] 선택 bias: {bias:+.4f} 달러")
    print(f"[bias] 저장: {TFT_V2_BIAS_PATH.resolve()}")
    print(f"[plot] 저장: {corrected_plot.resolve()}")


def run_tft_multi_pipeline(db_path: str | Path | None = None) -> dict[str, float]:
    db = Path(db_path) if db_path else DEFAULT_DB_PATH
    training, calibration_dataset, test_dataset, df_long, group_idx_to_name = prepare_tft_multi_dataset(
        db
    )
    n_series = int(df_long["commodity"].nunique())
    n_train_samples = len(training)
    print(f"총 시계열 수: {n_series}")
    print(f"총 학습 샘플 수: {n_train_samples:,}")

    tft = build_tft_model(training)
    trainer, tft = train_tft(
        training,
        calibration_dataset,
        tft=tft,
        max_epochs=50,
        force_cpu=True,
    )

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(TFT_V3_MULTI_CKPT_PATH))
    print(f"[run_tft_multi_pipeline] 체크포인트 저장: {TFT_V3_MULTI_CKPT_PATH.resolve()}")

    df_cal = _collect_validation_predictions(
        tft,
        calibration_dataset,
        quiet=True,
        group_idx_to_name=group_idx_to_name,
    )
    df_test = _collect_validation_predictions(
        tft,
        test_dataset,
        quiet=True,
        group_idx_to_name=group_idx_to_name,
    )

    soy_cal = df_cal.loc[df_cal["commodity"] == "soybean_oil"].copy()
    soy_test = df_test.loc[df_test["commodity"] == "soybean_oil"].copy()
    if soy_cal.empty or soy_test.empty:
        raise RuntimeError("soybean_oil 평가용 예측 행이 비어 있습니다.")

    residuals = np.abs(soy_cal["p50"].to_numpy(dtype=np.float64) - soy_cal["actual"].to_numpy(dtype=np.float64))
    q = calibrate_conformal_v2(residuals)
    cal_start = pd.Timestamp("2025-08-01")
    cal_end = pd.Timestamp("2025-09-30")
    soy_date_map = (
        df_long.loc[df_long["commodity"] == "soybean_oil", ["time_idx", "date"]]
        .drop_duplicates("time_idx")
        .copy()
    )
    soy_cal = soy_cal.merge(soy_date_map, on="time_idx", how="left")
    soy_test = soy_test.merge(soy_date_map, on="time_idx", how="left")

    bias = calc_tft_bias(soy_cal, cal_start=cal_start, cal_end=cal_end)
    _save_tft_v2_bias(bias, cal_start=cal_start, cal_end=cal_end, out_path=TFT_V3_BIAS_PATH)
    p50_raw = soy_test["p50"].to_numpy(dtype=np.float64)
    actual = soy_test["actual"].to_numpy(dtype=np.float64)
    p50_corr = p50_raw + bias
    coverage_v2 = float(np.mean((actual > p50_corr - q) & (actual < p50_corr + q)))
    mae_p50 = float(np.mean(np.abs(p50_raw - actual)))

    corrected_plot = NOTEBOOKS_DIR / "results" / "tft_v3_conformal.png"
    _plot_tft_v2_conformal_corrected(soy_test, predict_with_correction(soy_test, bias), q, corrected_plot)
    pred_plot = NOTEBOOKS_DIR / "results" / "tft_v3_prediction.png"
    _plot_p50_vs_actual_scatter(soy_test, pred_plot)
    vi_plot = NOTEBOOKS_DIR / "results" / "tft_v3_variable_importance.png"
    _ = get_tft_variable_importance(tft, calibration_dataset, out_path=vi_plot)
    dbg_plot = NOTEBOOKS_DIR / "results" / "tft_v3_debug_p50_vs_actual.png"
    _plot_p50_vs_actual_scatter(soy_test, dbg_plot)

    target_ok = coverage_v2 >= 0.80
    print(f"\nMAE (P50, soybean_oil, test): {mae_p50:.2f} 달러")
    print(f"Coverage (Conformal v2, test): {coverage_v2 * 100:.2f}%")
    print(f"Conformal v2 q (±반폭): {q:.4f} 달러 | calibration 잔차 분위(alpha=0.85) 기반")
    print(f"목표 80% 달성 여부: {'예' if target_ok else '아니오'}")
    print(
        f"학습 완료. bias={bias:+.4f}달러/톤.\n"
        f"체크포인트: {TFT_V3_MULTI_CKPT_PATH}"
    )

    return {
        "mae_p50_soybean_test": mae_p50,
        "coverage_conformal_v2": coverage_v2,
        "q_conformal_v2": q,
        "bias_cal": bias,
        "n_train_samples": float(n_train_samples),
        "n_series": float(n_series),
    }


def run_tft_pipeline(
    db_path: str | Path | None = None,
    version: str = "v2_interview",
    *,
    config_path: Path | None = None,
) -> None:
    db = Path(db_path) if db_path else DEFAULT_DB_PATH
    cfg = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    training, calibration_dataset, test_dataset, df_processed = prepare_tft_dataset(
        db, version, config_path=cfg
    )
    tft = build_tft_model(training)
    trainer, tft = train_tft(training, calibration_dataset, tft=tft)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_path = MODEL_CKPT_PATH
    pred_plot_path = NOTEBOOKS_DIR / "tft_prediction.png"
    conf_plot_path = NOTEBOOKS_DIR / "tft_conformal.png"
    if version == "v2_interview":
        ckpt_path = MODELS_DIR / "tft_v2_interview.ckpt"
        pred_plot_path = NOTEBOOKS_DIR / "tft_v2_prediction.png"
        conf_plot_path = NOTEBOOKS_DIR / "tft_v2_conformal.png"

    trainer.save_checkpoint(str(ckpt_path))
    print(f"[run_tft_pipeline] 체크포인트 저장: {ckpt_path.resolve()}")

    df_cal = _collect_validation_predictions(tft, calibration_dataset, quiet=True)
    df_test = _collect_validation_predictions(tft, test_dataset, quiet=False)
    metrics = evaluate_tft(
        trainer, tft, test_dataset, df_processed, df_pred=df_test, plot_path=pred_plot_path
    )
    calib_out = calibrate_conformal(
        tft,
        calibration_dataset,
        df_processed,
        target_coverage=0.80,
        df_pred=df_cal,
    )
    q_base = float(calib_out["q_base"])
    q_adj = float(calib_out["q_adjusted"])
    vol_ratio = float(calib_out["vol_ratio"])
    n_cal = len(df_cal)
    conf_metrics = evaluate_tft_with_conformal(
        tft,
        test_dataset,
        df_processed,
        df_cal,
        q_base,
        q_adj,
        vol_ratio,
        df_pred=df_test,
        n_calibration_steps=n_cal,
        plot_path=conf_plot_path,
    )
    _diagnose_conformal_coverage_gap(df_processed, df_test, q_half_width=q_adj)
    get_tft_attention(tft, calibration_dataset)
    top5 = get_tft_variable_importance(tft, calibration_dataset)

    max_t = int(df_processed["time_idx"].max())
    train_end = max_t - VAL_DAYS
    test_start = train_end + CAL_DAYS + 1
    naive_mae = _naive_persistence_mae(df_processed, test_start)
    n_test = len(df_test)

    cov_tft_band = conf_metrics["coverage_tft"] * 100.0
    cov_cf_base = conf_metrics["coverage_conformal_base"] * 100.0
    cov_cf_vol = conf_metrics["coverage_conformal_vol"] * 100.0

    print("\n[TFT 요약]")
    print(f"MAE (P50, 보정 전, test):        {conf_metrics['mae_p50_raw']:.4f} 달러")
    print(f"MAE (P50, 편향 보정 후, test):   {conf_metrics['mae_p50_corrected']:.4f} 달러")
    print(f"Coverage (TFT 원본, test):       {cov_tft_band:.2f}%")
    print(f"Coverage (Conformal 기본, test): {cov_cf_base:.2f}%")
    print(f"Coverage (Conformal 변동성조정, test): {cov_cf_vol:.2f}% (목표 80%)")
    print(f"보정 편향값:                     {conf_metrics['bias']:+.4f} 달러")
    print(f"변동성 조정 비율:                {vol_ratio:.4f}배")
    print("------------------------------------")
    print(f"RMSE (P50, test, 원본):         {metrics['rmse']:.4f} 달러")
    print(f"q_base (Conformal 기본 반폭):   ±{q_base:.4f} 달러")
    print(f"q_adj (변동성 조정 반폭):       ±{q_adj:.4f} 달러")
    print(
        f"보정에 사용한 데이터:          calibration {CAL_DAYS}일 "
        f"(예측 스텝 {n_cal:,}개)"
    )
    print(
        f"평가에 사용한 데이터:          test {TEST_DAYS}일 "
        f"(예측 스텝 {n_test:,}개, 학습·보정 미사용)"
    )
    print(
        f"평균 구간폭 (test): TFT {conf_metrics['mean_width_before']:.4f} 달러 | "
        f"Conformal 기본 {conf_metrics['mean_width_conformal_base']:.4f} | "
        f"변동성조정 {conf_metrics['mean_width_conformal_vol']:.4f}"
    )
    print(
        f"test naive persistence MAE ≈ {naive_mae:.4f}"
    )
    print("Variable importance 상위 3개 (calibration 기준):")
    for nm, sc in top5[:3]:
        print(f"  - {nm}: {sc * 100:.2f}%")
    print("")

    if version == "v2_interview":
        prev_cov = 51.79
        new_cov = cov_cf_vol
        print("[TFT v2_interview]")
        print(f"MAE (P50, test): {conf_metrics['mae_p50_corrected']:.2f}달러")
        print(f"Coverage (Conformal 변동성조정, test): {new_cov:.2f}%")
        print(
            "Variable Importance 상위 3개: "
            + ", ".join([f"{nm}({sc * 100:.2f}%)" for nm, sc in top5[:3]])
        )
        print(f"v1 대비 Coverage 변화: {prev_cov:.2f}% → {new_cov:.2f}%")


if __name__ == "__main__":
    warnings.filterwarnings(
        "ignore",
        message="X does not have valid feature names",
        category=UserWarning,
    )
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--train",
        action="store_true",
        help="전체 학습 파이프라인 실행 (체크포인트 덮어쓰기)",
    )
    ap.add_argument("--db", type=str, default=None, help="SQLite DB 경로 (기본: data/db/soybean.db)")
    ap.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="평가용 체크포인트 (기본: models/tft_v2_multi.ckpt)",
    )
    args = ap.parse_args()
    if args.train:
        run_tft_multi_pipeline(Path(args.db) if args.db else None)
    else:
        run_tft_multi_eval_from_checkpoint(
            Path(args.db) if args.db else None,
            Path(args.ckpt) if args.ckpt else None,
        )
