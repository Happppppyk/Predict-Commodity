from __future__ import annotations

import json
import sqlite3
import sys
import warnings
from pathlib import Path
from typing import Any

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.plot_style import configure_matplotlib_korean

configure_matplotlib_korean()
import numpy as np
import pandas as pd
import shap
import yaml
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

# 프로젝트 루트: src/models → parents[2] == soybean-oil-poc
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "db" / "soybean.db"
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "features.yaml"
MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
TARGET_COL = "target_updown_t7"

# 이전 master 기준 참고값 (데이터 재빌드 전)
PREVIOUS_SINGLE_SPLIT_AUC = 0.5532
PREVIOUS_WALK_FORWARD_MEAN_AUC = 0.5398
RERUN_SINGLE_SPLIT_AUC = 0.5563
RERUN_MODEL_PATH = MODELS_DIR / "xgb_v1_stationary_rerun.json"
RERUN_SHAP_PATH = NOTEBOOKS_DIR / "shap_summary_v1_stationary_rerun.png"

# Walk-forward: Optuna 튜닝으로 얻은 고정 하이퍼파라미터(반올림) + 단일 split 참고 AUC
WALK_FORWARD_FIXED_PARAMS: dict = {
    "max_depth": 6,
    "learning_rate": 0.0154,
    "n_estimators": 173,
    "subsample": 0.962,
    "colsample_bytree": 0.889,
    "min_child_weight": 5,
    "gamma": 0.145,
}

# models/xgb_v3_clean_t28_final.json Booster 설정과 트리 개수(180) 기준
V3_CLEAN_T28_XGB_PARAMS: dict[str, Any] = {
    "max_depth": 6,
    "learning_rate": 0.3,
    "n_estimators": 180,
    "subsample": 1.0,
    "colsample_bytree": 1.0,
    "min_child_weight": 1,
    "gamma": 0.0,
}

# 인터뷰/보고용 기준(동일 DB·설정에서 재현 시 run 결과로 대체 가능)
V3_CLEAN_T28_REFERENCE_SPLIT_AUC = 0.6692
V3_CLEAN_T28_REFERENCE_WF_MEAN_AUC = 0.6463
WALK_FORWARD_INITIAL_TRAIN = 500
WALK_FORWARD_STEP = 60
SINGLE_SPLIT_REFERENCE_AUC = PREVIOUS_SINGLE_SPLIT_AUC  # train/test 단일 분할(튜닝 후) 참고선
WALK_FORWARD_PASS_THRESHOLD = 0.52
WALK_FORWARD_CSV_PATH = NOTEBOOKS_DIR / "walk_forward_results.csv"
WALK_FORWARD_PLOT_PATH = NOTEBOOKS_DIR / "walk_forward_results.png"

# 앙상블 v3_clean / t28 최종 산출물 (DQN state 등 추론용)
ENSEMBLE_V3_CLEAN_T28_FINAL_DIR = MODELS_DIR / "ensemble_v3_clean_t28_final"
NOTEBOOKS_RESULTS_DIR = NOTEBOOKS_DIR / "results"
WALK_FORWARD_ENSEMBLE_FINAL_PLOT = NOTEBOOKS_RESULTS_DIR / "walk_forward_ensemble_final.png"
ENSEMBLE_FINAL_FEATURE_VERSION = "v3_clean"
ENSEMBLE_FINAL_TARGET_COL = "target_updown_t28"
ENSEMBLE_FINAL_SPLIT_AUC = 0.6139
ENSEMBLE_FINAL_WF_MEAN_AUC = 0.7022


def _load_yaml_config(config_path: Path | None = None) -> dict:
    path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_features(
    conn: sqlite3.Connection,
    version: str = "v1",
    *,
    target_col: str | None = None,
    config_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    features.yaml의 지정 버전 columns에서 피처 목록을 읽고, master_daily에서 로드한다.

    - target_*, feature_version 은 입력 피처에서 제외
    - ``target_col``(기본 ``target_updown_t7``)을 y로 로드 후 결측 행 제거
    - 반환 X는 일자 인덱스를 유지하여 시계열 split에 사용한다.
    """
    tcol = target_col or TARGET_COL
    cfg = _load_yaml_config(config_path)
    if version not in cfg.get("versions", {}):
        raise KeyError(f"features.yaml에 버전 '{version}' 이 없습니다.")

    all_cols: list[str] = list(cfg["versions"][version]["columns"])
    configured_feature_names = [
        c
        for c in all_cols
        if not str(c).startswith("target_") and c != "feature_version"
    ]
    schema_info = conn.execute("PRAGMA table_info(master_daily)").fetchall()
    existing_cols = {str(r[1]) for r in schema_info}
    if not existing_cols:
        raise RuntimeError("master_daily 테이블 스키마를 읽지 못했습니다.")

    missing_features = [c for c in configured_feature_names if c not in existing_cols]
    feature_names = [c for c in configured_feature_names if c in existing_cols]
    if missing_features:
        warnings.warn(
            f"master_daily에 없는 피처 {len(missing_features)}개는 제외하고 진행합니다: {missing_features}",
            UserWarning,
            stacklevel=2,
        )
    if tcol not in existing_cols:
        raise KeyError(f"master_daily에 타깃 컬럼이 없습니다: {tcol}")
    if "date" not in existing_cols:
        raise KeyError("master_daily에 date 컬럼이 없습니다.")

    load_cols = ["date"] + feature_names + [tcol]
    # SQL 인젝션 방지: 컬럼명은 yaml 기준 화이트리스트
    quoted = ", ".join(f'"{c}"' for c in load_cols)
    sql = f"SELECT {quoted} FROM master_daily ORDER BY date"
    raw = pd.read_sql_query(sql, conn)

    missing = [c for c in load_cols if c not in raw.columns]
    if missing:
        raise KeyError(f"master_daily에 없는 컬럼: {missing}")

    raw["date"] = pd.to_datetime(raw["date"], errors="coerce")
    raw = raw.dropna(subset=["date"]).sort_values("date").set_index("date")

    X = raw[feature_names]
    y = raw[tcol]

    if tcol in ("target_updown_t28", "target_thresh_t28"):
        tail_n = min(28, len(raw))
        tail_nan = int(raw[tcol].tail(tail_n).isna().sum()) if tail_n > 0 else 0
        print(
            f"[load_features] t28 계열 tail NaN 점검: 마지막 {tail_n}행 중 NaN {tail_nan}행"
        )

    # 결측이 있는 행 제거 (피처 + 타깃 모두 완전해야 함)
    mask = X.notna().all(axis=1) & y.notna()
    X = X.loc[mask]
    y = y.loc[mask]

    # 이진 라벨 정수화 (0/1)
    y = pd.Series(y.astype(float).astype(int), index=y.index, name=tcol)
    if not y.isin([0, 1]).all():
        bad = y[~y.isin([0, 1])].unique()
        raise ValueError(f"{tcol} 은 0/1만 허용됩니다. 발견 값: {bad}")

    n = len(X)
    print(
        f"[load_features] 버전={version!r} | 타깃={tcol!r} | "
        f"비결측 행 수: {n:,} | 피처 수: {len(feature_names)} "
        f"(설정 {len(configured_feature_names)}개)"
    )
    return X, y, feature_names


def time_series_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_start_date: str = "2024-01-01",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    날짜 인덱스 기준 시계열 분할. 랜덤 split 금지.

    - train: test_start_date **미만**
    - test: test_start_date **이상**
    """
    cutoff = pd.Timestamp(test_start_date)
    tr_mask = X.index < cutoff
    te_mask = X.index >= cutoff

    X_train, y_train = X.loc[tr_mask], y.loc[tr_mask]
    X_test, y_test = X.loc[te_mask], y.loc[te_mask]

    print(
        f"[time_series_split] 기준일={test_start_date!r} | "
        f"train {len(X_train):,}행 | test {len(X_test):,}행"
    )
    if len(X_train) == 0:
        warnings.warn("학습 구간 행이 0입니다.", UserWarning, stacklevel=2)
    if len(X_test) == 0:
        warnings.warn("테스트 구간 행이 0입니다.", UserWarning, stacklevel=2)

    return X_train, y_train, X_test, y_test


def train_xgboost(X_train: pd.DataFrame, y_train: pd.Series) -> XGBClassifier:
    """고정 하이퍼파라미터 XGBClassifier 학습 (early stopping 없음)."""
    # 시계열이므로 여기서는 검증 세트 분리·조기 종료를 쓰지 않는다.
    model = XGBClassifier(
        max_depth=4,
        learning_rate=0.05,
        n_estimators=300,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=42,
        objective="binary:logistic",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("[train_xgboost] 학습 완료 (n_estimators=300 고정)")
    return model


def evaluate_model(
    model: XGBClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: list[str],
    *,
    shap_summary_path: Path,
) -> dict:
    """
    분류 지표, 혼동행렬, TreeSHAP 요약 플롯 및 피처별 |SHAP| 평균 상위 10개 출력.

    반환 dict: 스칼라 지표 + shap_top10 (이름, 평균절대SHAP) — 파이프라인 요약용.
    """
    y_pred = model.predict(X_test)
    proba = model.predict_proba(X_test)[:, 1]

    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    print("\n=== 분류 지표 ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-score:  {f1:.4f}")

    if len(np.unique(y_test)) < 2:
        auc_roc = float("nan")
        print("AUC-ROC:   (테스트에 단일 클래스만 있어 계산 생략)")
    else:
        auc_roc = float(roc_auc_score(y_test, proba))
        print(f"AUC-ROC:   {auc_roc:.4f}")

    print("\n=== Confusion Matrix (행=실제, 열=예측) ===")
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    print(cm)

    # TreeSHAP (XGBoost + sklearn 래퍼)
    explainer = shap.TreeExplainer(model)
    shap_raw = explainer.shap_values(X_test)
    # 이진 분류: 리스트 [class0, class1] 또는 단일 배열
    if isinstance(shap_raw, list):
        shap_vals = np.asarray(shap_raw[1])
    else:
        shap_vals = np.asarray(shap_raw)

    mean_abs = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(-mean_abs)
    names = list(feature_names)
    if mean_abs.shape[0] != len(names):
        names = list(X_test.columns)

    shap_top10 = [
        (names[i], float(mean_abs[i])) for i in order[:10]
    ]

    print("\n=== SHAP (|값| 평균) 상위 10개 피처 ===")
    for rank, (name, val) in enumerate(shap_top10, start=1):
        print(f"  {rank:2d}. {name}: {val:.6f}")

    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    shap_summary_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_vals,
        X_test,
        feature_names=names,
        show=False,
        max_display=min(20, len(names)),
    )
    plt.tight_layout()
    plt.savefig(shap_summary_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[evaluate_model] SHAP Summary Plot 저장: {shap_summary_path.resolve()}")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc_roc": auc_roc,
        "confusion_matrix": cm,
        "shap_top10": shap_top10,
        "shap_mean_abs": {names[i]: float(mean_abs[i]) for i in range(len(names))},
    }


def objective(trial, X_train: pd.DataFrame, y_train: pd.Series) -> float:
    """
    Optuna 목적함수: TimeSeriesSplit(5) 교차검증 AUC-ROC 평균을 maximize한다.
    시계열 순서를 유지하기 위해 KFold 대신 TimeSeriesSplit만 사용한다.
    """
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 0.5),
    }

    tscv = TimeSeriesSplit(n_splits=5)
    aucs: list[float] = []

    for tr_idx, va_idx in tscv.split(X_train):
        Xi_tr = X_train.iloc[tr_idx]
        Xi_va = X_train.iloc[va_idx]
        yi_tr = y_train.iloc[tr_idx]
        yi_va = y_train.iloc[va_idx]

        model = XGBClassifier(
            **params,
            eval_metric="logloss",
            random_state=42,
            objective="binary:logistic",
            n_jobs=-1,
        )
        model.fit(Xi_tr, yi_tr)
        proba = model.predict_proba(Xi_va)[:, 1]
        if len(np.unique(yi_va)) < 2:
            continue
        aucs.append(float(roc_auc_score(yi_va, proba)))

    if not aucs:
        return 0.0
    return float(np.mean(aucs))


def tune_xgboost(X_train: pd.DataFrame, y_train: pd.Series, n_trials: int = 50) -> dict:
    """
    Optuna로 하이퍼파라미터 탐색. 로그는 WARNING 이상만 출력한다.
    반환: best_params, best_auc_cv (교차검증 AUC-ROC 평균 최댓값)
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    study = optuna.create_study(direction="maximize")

    def _obj(trial: optuna.Trial) -> float:
        return objective(trial, X_train, y_train)

    study.optimize(_obj, n_trials=n_trials, show_progress_bar=False)

    best_params = dict(study.best_params)
    best_auc = float(study.best_value)
    print(f"\n[tune_xgboost] CV AUC-ROC 최고: {best_auc:.4f} (n_trials={n_trials})")
    print("[tune_xgboost] 최적 파라미터:")
    for k in sorted(best_params.keys()):
        print(f"  {k}: {best_params[k]}")

    return {"best_params": best_params, "best_auc_cv": best_auc}


def run_tuned_pipeline(
    db_path: str | Path | None = None,
    version: str = "v1_stationary",
    n_trials: int = 50,
    test_start_date: str = "2024-01-01",
    *,
    target: str | None = None,
    config_path: Path | None = None,
    shap_summary_path: Path | None = None,
    artifact_tag: str | None = None,
) -> dict[str, Any]:
    """
    피처 로드·시계열 split 후 Optuna 튜닝 → 최적 파라미터로 전체 train 재학습 → test 평가·저장.
    모델: models/xgb_{version}_tuned.json, SHAP: notebooks/shap_summary_{version}_tuned.png
    (또는 ``shap_summary_path`` 지정 시 해당 경로)

    ``artifact_tag``가 있으면 파일명에 접미사를 붙여 같은 version·다른 타깃이 덮어쓰지 않게 한다.
    """
    tcol = target or TARGET_COL
    db = Path(db_path) if db_path else DEFAULT_DB_PATH
    if not db.is_file():
        raise FileNotFoundError(f"DB 파일이 없습니다: {db.resolve()}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tag = f"_{artifact_tag}" if artifact_tag else ""
    model_path = MODELS_DIR / f"xgb_{version}_tuned{tag}.json"
    shap_path = shap_summary_path or (NOTEBOOKS_DIR / f"shap_summary_{version}_tuned{tag}.png")

    with sqlite3.connect(db) as conn:
        X, y, feature_names = load_features(
            conn, version=version, target_col=tcol, config_path=config_path
        )

    X_train, y_train, X_test, y_test = time_series_split(X, y, test_start_date=test_start_date)

    tune_result = tune_xgboost(X_train, y_train, n_trials=n_trials)

    model = XGBClassifier(
        **tune_result["best_params"],
        eval_metric="logloss",
        random_state=42,
        objective="binary:logistic",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    print("\n[run_tuned_pipeline] 최적 파라미터로 전체 train 재학습 완료")

    metrics = evaluate_model(
        model, X_test, y_test, feature_names, shap_summary_path=shap_path
    )

    try:
        model.save_model(str(model_path))
    except AttributeError:  # pragma: no cover
        model.get_booster().save_model(str(model_path))
    print(f"\n[run_tuned_pipeline] 모델 저장: {model_path.resolve()}")

    tr_idx, te_idx = X_train.index, X_test.index
    print("\n========== 튜닝 파이프라인 요약 ==========")
    print(f"피처 버전: {version!r} | Optuna trials: {n_trials}")
    if len(tr_idx):
        print(f"train 기간: {tr_idx.min().date()} ~ {tr_idx.max().date()} | 행 수: {len(tr_idx):,}")
    if len(te_idx):
        print(f"test 기간:  {te_idx.min().date()} ~ {te_idx.max().date()} | 행 수: {len(te_idx):,}")
    print("튜닝 전 AUC-ROC: 0.5178 (v1_stationary 기준)")
    if not np.isnan(metrics["auc_roc"]):
        print(f"튜닝 후 AUC-ROC: {metrics['auc_roc']:.4f}")
    else:
        print("튜닝 후 AUC-ROC: N/A")
    print(f"CV 최적 AUC-ROC (TimeSeriesSplit 5): {tune_result['best_auc_cv']:.4f}")
    print("최적 파라미터 전체:")
    for k in sorted(tune_result["best_params"].keys()):
        print(f"  {k}: {tune_result['best_params'][k]}")
    print("==========================================\n")

    return {
        "target": tcol,
        "version": version,
        "metrics": metrics,
        "tune_result": tune_result,
        "train_index": tr_idx,
        "test_index": te_idx,
    }


def walk_forward_validation(
    db_path: str | Path | None = None,
    version: str = "v1_stationary",
    *,
    target: str | None = None,
    config_path: Path | None = None,
    initial_train: int = WALK_FORWARD_INITIAL_TRAIN,
    step: int = WALK_FORWARD_STEP,
    xgb_params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """
    시간순 전체 데이터에 대해 expanding window 워크포워드 검증.

    - 초기 학습: 첫 `initial_train`행
    - 매 스텝: 직전까지 누적 학습 → 다음 `step`행 테스트
    - 하이퍼파라미터: ``xgb_params``가 있으면 사용, 없으면 ``WALK_FORWARD_FIXED_PARAMS``
    """
    db = Path(db_path) if db_path else DEFAULT_DB_PATH
    if not db.is_file():
        raise FileNotFoundError(f"DB 파일이 없습니다: {db.resolve()}")

    tcol = target or TARGET_COL
    params = dict(xgb_params) if xgb_params is not None else dict(WALK_FORWARD_FIXED_PARAMS)
    with sqlite3.connect(db) as conn:
        X, y, _ = load_features(conn, version=version, target_col=tcol, config_path=config_path)

    if initial_train + step > len(X):
        raise ValueError(
            f"데이터 부족: 전체 {len(X)}행, 필요 최소 {initial_train + step}행 "
            f"(초기학습 {initial_train} + 테스트 {step})"
        )

    rows: list[dict] = []
    train_end = initial_train  # 학습에 쓰는 행 수 [0:train_end), 테스트 [train_end:train_end+step)
    step_idx = 0

    while train_end + step <= len(X):
        X_tr = X.iloc[:train_end]
        y_tr = y.iloc[:train_end]
        X_te = X.iloc[train_end : train_end + step]
        y_te = y.iloc[train_end : train_end + step]

        model = XGBClassifier(
            **params,
            eval_metric="logloss",
            random_state=42,
            objective="binary:logistic",
            n_jobs=-1,
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        proba = model.predict_proba(X_te)[:, 1]
        acc = float(accuracy_score(y_te, y_pred))
        f1 = float(f1_score(y_te, y_pred, zero_division=0))
        if len(np.unique(y_te)) < 2:
            auc_v = float("nan")
        else:
            auc_v = float(roc_auc_score(y_te, proba))

        tr_idx = X_tr.index
        te_idx = X_te.index
        rows.append(
            {
                "step": step_idx + 1,
                "train_end_date": tr_idx.max().date().isoformat(),
                "test_start_date": te_idx.min().date().isoformat(),
                "test_end_date": te_idx.max().date().isoformat(),
                "n_train": len(X_tr),
                "n_test": len(X_te),
                "accuracy": acc,
                "auc_roc": auc_v,
                "f1": f1,
            }
        )
        train_end += step
        step_idx += 1

    print(f"[walk_forward_validation] 스텝 수: {len(rows)} (초기학습={initial_train}, step={step})")
    return pd.DataFrame(rows)


def plot_walk_forward_results(results_df: pd.DataFrame) -> None:
    """스텝별 AUC-ROC 라인 + 평균(빨강)·단일 split 참고(파랑) 수평선. PNG 저장 및 요약 출력."""
    if results_df.empty:
        warnings.warn("plot_walk_forward_results: 결과가 비어 있습니다.", UserWarning, stacklevel=2)
        return

    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    mean_auc = float(results_df["auc_roc"].mean())
    min_auc = float(results_df["auc_roc"].min())
    max_auc = float(results_df["auc_roc"].max())
    idx_min = int(results_df["auc_roc"].idxmin())
    idx_max = int(results_df["auc_roc"].idxmax())
    worst_step = int(results_df.loc[idx_min, "step"])
    best_step = int(results_df.loc[idx_max, "step"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        results_df["step"],
        results_df["auc_roc"],
        marker="o",
        color="steelblue",
        label="Walk-forward AUC-ROC (스텝별)",
    )
    ax.axhline(
        mean_auc,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"평균 AUC-ROC ({mean_auc:.4f})",
    )
    ax.axhline(
        SINGLE_SPLIT_REFERENCE_AUC,
        color="blue",
        linestyle="--",
        linewidth=1.5,
        label=f"단일 train/test split ({SINGLE_SPLIT_REFERENCE_AUC:.4f})",
    )
    ax.set_xlabel("스텝")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Walk-forward 검증: 스텝별 AUC-ROC")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(WALK_FORWARD_PLOT_PATH, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[plot_walk_forward_results] 그래프 저장: {WALK_FORWARD_PLOT_PATH.resolve()}")

    print("\n=== 스텝별 AUC-ROC 테이블 ===")
    print(results_df.to_string(index=False))
    print(f"\n평균 AUC-ROC:   {mean_auc:.4f}")
    print(
        f"최저 AUC-ROC:   {min_auc:.4f} (스텝 {worst_step}, "
        f"{results_df.loc[idx_min, 'test_start_date']} ~ {results_df.loc[idx_min, 'test_end_date']})"
    )
    print(
        f"최고 AUC-ROC:   {max_auc:.4f} (스텝 {best_step}, "
        f"{results_df.loc[idx_max, 'test_start_date']} ~ {results_df.loc[idx_max, 'test_end_date']})"
    )


def plot_walk_forward_ensemble_final(
    results_df: pd.DataFrame,
    output_path: Path | None = None,
    *,
    split_ref_auc: float = ENSEMBLE_FINAL_SPLIT_AUC,
) -> None:
    """앙상블 WF 결과를 ``notebooks/results/walk_forward_ensemble_final.png`` 등에 저장."""
    if results_df.empty:
        warnings.warn(
            "plot_walk_forward_ensemble_final: 결과가 비어 있습니다.", UserWarning, stacklevel=2
        )
        return

    path = output_path or WALK_FORWARD_ENSEMBLE_FINAL_PLOT
    path.parent.mkdir(parents=True, exist_ok=True)

    mean_auc = float(results_df["auc_roc"].mean())
    min_auc = float(results_df["auc_roc"].min())
    max_auc = float(results_df["auc_roc"].max())
    idx_min = int(results_df["auc_roc"].idxmin())
    idx_max = int(results_df["auc_roc"].idxmax())
    worst_step = int(results_df.loc[idx_min, "step"])
    best_step = int(results_df.loc[idx_max, "step"])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        results_df["step"],
        results_df["auc_roc"],
        marker="o",
        color="steelblue",
        label="Walk-forward AUC-ROC (앙상블)",
    )
    ax.axhline(
        mean_auc,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"평균 AUC-ROC ({mean_auc:.4f})",
    )
    ax.axhline(
        split_ref_auc,
        color="blue",
        linestyle="--",
        linewidth=1.5,
        label=f"단일 split 참고 ({split_ref_auc:.4f})",
    )
    ax.set_xlabel("스텝")
    ax.set_ylabel("AUC-ROC")
    ax.set_title("Walk-forward: 앙상블 (XGBoost + LightGBM + CatBoost), v3_clean t28")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[plot_walk_forward_ensemble_final] 그래프 저장: {path.resolve()}")

    print("\n=== 앙상블 Walk-forward 스텝별 AUC-ROC ===")
    print(results_df.to_string(index=False))
    print(f"\n평균 AUC-ROC:   {mean_auc:.4f}")
    print(
        f"최저 AUC-ROC:   {min_auc:.4f} (스텝 {worst_step}, "
        f"{results_df.loc[idx_min, 'test_start_date']} ~ {results_df.loc[idx_min, 'test_end_date']})"
    )
    print(
        f"최고 AUC-ROC:   {max_auc:.4f} (스텝 {best_step}, "
        f"{results_df.loc[idx_max, 'test_start_date']} ~ {results_df.loc[idx_max, 'test_end_date']})"
    )


def run_walk_forward(
    db_path: str | Path | None = None,
    version: str = "v1_stationary",
    *,
    target: str | None = None,
    config_path: Path | None = None,
    xgb_params: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """워크포워드 검증 → CSV 저장 → 플롯 → 통과 여부 출력."""
    results_df = walk_forward_validation(
        db_path,
        version=version,
        target=target,
        config_path=config_path,
        xgb_params=xgb_params,
    )
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(WALK_FORWARD_CSV_PATH, index=False)
    print(f"\n[run_walk_forward] CSV 저장: {WALK_FORWARD_CSV_PATH.resolve()}")

    plot_walk_forward_results(results_df)

    mean_auc = float(results_df["auc_roc"].mean())
    print("\n========== Walk-forward 최종 판단 ==========")
    if mean_auc >= WALK_FORWARD_PASS_THRESHOLD:
        print(
            f"✅ Walk-forward 검증 통과 "
            f"(평균 AUC-ROC {mean_auc:.4f} >= {WALK_FORWARD_PASS_THRESHOLD})"
        )
    else:
        print(
            f"❌ 과적합 의심 — 피처 재검토 필요 "
            f"(평균 AUC-ROC {mean_auc:.4f} < {WALK_FORWARD_PASS_THRESHOLD})"
        )
    print("============================================\n")
    return results_df


def _fit_ensemble_base_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> tuple[XGBClassifier, Any, Any]:
    """XGB + LGBM + CatBoost를 동일 학습 행에 맞춰 학습한다."""
    from catboost import CatBoostClassifier
    from lightgbm import LGBMClassifier

    xgb_m = XGBClassifier(
        **V3_CLEAN_T28_XGB_PARAMS,
        eval_metric="logloss",
        random_state=42,
        objective="binary:logistic",
        n_jobs=-1,
    )
    xgb_m.fit(X_train, y_train)
    lgb_m = LGBMClassifier(
        num_leaves=31,
        learning_rate=0.05,
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        verbose=-1,
    )
    lgb_m.fit(X_train, y_train)
    cat_m = CatBoostClassifier(
        depth=6,
        learning_rate=0.05,
        iterations=300,
        random_seed=42,
        verbose=False,
        allow_writing_files=False,
    )
    cat_m.fit(X_train, y_train)
    return xgb_m, lgb_m, cat_m


def _save_ensemble_v3_clean_t28_artifacts(
    models_dir: Path,
    xgb_m: XGBClassifier,
    lgb_m: Any,
    cat_m: Any,
    feature_names: list[str],
) -> None:
    """최종 앙상블 가중치·피처 순서를 ``models/ensemble_v3_clean_t28_final/`` 형태로 저장."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    xgb_path = models_dir / "xgb_ensemble.json"
    lgb_path = models_dir / "lgb_ensemble.txt"
    cat_path = models_dir / "cat_ensemble.cbm"
    meta_path = models_dir / "feature_columns.json"

    try:
        xgb_m.save_model(str(xgb_path))
    except AttributeError:  # pragma: no cover
        xgb_m.get_booster().save_model(str(xgb_path))
    lgb_m.booster_.save_model(str(lgb_path))
    cat_m.save_model(str(cat_path))
    meta_path.write_text(
        json.dumps(
            {
                "feature_names": feature_names,
                "target": ENSEMBLE_FINAL_TARGET_COL,
                "feature_version": ENSEMBLE_FINAL_FEATURE_VERSION,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"[_save_ensemble_v3_clean_t28_artifacts] 저장 완료: "
        f"{xgb_path.name}, {lgb_path.name}, {cat_path.name}, {meta_path.name}"
    )


def walk_forward_ensemble_average(
    X: pd.DataFrame,
    y: pd.Series,
    *,
    initial_train: int = WALK_FORWARD_INITIAL_TRAIN,
    step: int = WALK_FORWARD_STEP,
) -> pd.DataFrame:
    """XGBoost(v3_clean 고정 하이퍼) + LightGBM + CatBoost 확률 평균으로 WF AUC 계산."""

    if initial_train + step > len(X):
        raise ValueError(
            f"데이터 부족: 전체 {len(X)}행, 필요 최소 {initial_train + step}행 "
            f"(초기학습 {initial_train} + 테스트 {step})"
        )

    rows: list[dict] = []
    train_end = initial_train
    step_idx = 0

    while train_end + step <= len(X):
        X_tr = X.iloc[:train_end]
        y_tr = y.iloc[:train_end]
        X_te = X.iloc[train_end : train_end + step]
        y_te = y.iloc[train_end : train_end + step]

        xgb_m, lgb_m, cat_m = _fit_ensemble_base_models(X_tr, y_tr)
        proba = (
            xgb_m.predict_proba(X_te)[:, 1]
            + lgb_m.predict_proba(X_te)[:, 1]
            + cat_m.predict_proba(X_te)[:, 1]
        ) / 3.0

        if len(np.unique(y_te)) < 2:
            auc_v = float("nan")
        else:
            auc_v = float(roc_auc_score(y_te, proba))

        tr_idx = X_tr.index
        te_idx = X_te.index
        rows.append(
            {
                "step": step_idx + 1,
                "train_end_date": tr_idx.max().date().isoformat(),
                "test_start_date": te_idx.min().date().isoformat(),
                "test_end_date": te_idx.max().date().isoformat(),
                "n_train": len(X_tr),
                "n_test": len(X_te),
                "auc_roc": auc_v,
            }
        )
        train_end += step
        step_idx += 1

    print(
        f"[walk_forward_ensemble_average] 스텝 수: {len(rows)} "
        f"(초기학습={initial_train}, step={step})"
    )
    return pd.DataFrame(rows)


def predict_ensemble(X: pd.DataFrame, models_dir: str | Path) -> np.ndarray:
    """
    저장된 앙상블 세 모델을 로드한 뒤 상승(클래스 1) 확률의 산술평균을 반환한다.

    - ``models_dir``: ``feature_columns.json``, ``xgb_ensemble.json``,
      ``lgb_ensemble.txt``, ``cat_ensemble.cbm`` 포함 디렉터리
    - ``X``: 학습 시와 동일한 피처명을 갖는 DataFrame (DQN state 파이프라인에서 전달)
    """
    import lightgbm as lgb
    from catboost import CatBoostClassifier

    d = Path(models_dir)
    meta = json.loads((d / "feature_columns.json").read_text(encoding="utf-8"))
    names: list[str] = list(meta["feature_names"])
    missing = [c for c in names if c not in X.columns]
    if missing:
        raise KeyError(f"입력 X에 없는 피처: {missing}")
    Xo = X[names]

    xgb_m = XGBClassifier()
    xgb_m.load_model(str(d / "xgb_ensemble.json"))
    p_xgb = xgb_m.predict_proba(Xo)[:, 1]

    booster = lgb.Booster(model_file=str(d / "lgb_ensemble.txt"))
    p_lgb = np.asarray(booster.predict(Xo), dtype=float)

    cat_m = CatBoostClassifier()
    cat_m.load_model(str(d / "cat_ensemble.cbm"))
    p_cat = cat_m.predict_proba(Xo)[:, 1]

    return (p_xgb + p_lgb + p_cat) / 3.0


def persist_ensemble_v3_clean_t28_final(
    db_path: str | Path | None = None,
    *,
    test_start_date: str = "2024-01-01",
    config_path: Path | None = None,
    models_dir: Path | None = None,
    wf_plot_path: Path | None = None,
) -> dict[str, Any]:
    """
    v3_clean / ``target_updown_t28`` 기준 단일 split 학습 모델을
    ``ensemble_v3_clean_t28_final``에 저장하고, WF 곡선을 ``notebooks/results``에 저장한다.
    """
    db = Path(db_path) if db_path else DEFAULT_DB_PATH
    cfg = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    out_dir = Path(models_dir) if models_dir else ENSEMBLE_V3_CLEAN_T28_FINAL_DIR
    plot_path = Path(wf_plot_path) if wf_plot_path else WALK_FORWARD_ENSEMBLE_FINAL_PLOT

    if not db.is_file():
        raise FileNotFoundError(f"DB 파일이 없습니다: {db.resolve()}")

    with sqlite3.connect(db) as conn:
        X, y, _fn = load_features(
            conn,
            version=ENSEMBLE_FINAL_FEATURE_VERSION,
            target_col=ENSEMBLE_FINAL_TARGET_COL,
            config_path=cfg,
        )

    X_train, y_train, _Xt, _yt = time_series_split(X, y, test_start_date=test_start_date)
    feature_names = list(X_train.columns)
    xgb_m, lgb_m, cat_m = _fit_ensemble_base_models(X_train, y_train)
    _save_ensemble_v3_clean_t28_artifacts(out_dir, xgb_m, lgb_m, cat_m, feature_names)

    wf_df = walk_forward_ensemble_average(X, y)
    plot_walk_forward_ensemble_final(wf_df, output_path=plot_path)

    print("\n=== 최종 확정 모델 ===")
    print("모델: 앙상블 (XGBoost + LightGBM + CatBoost)")
    print(f"피처: {ENSEMBLE_FINAL_FEATURE_VERSION} ({len(feature_names)}개)")
    print(f"타깃: {ENSEMBLE_FINAL_TARGET_COL}")
    print(f"AUC (split): {ENSEMBLE_FINAL_SPLIT_AUC}")
    print(f"WF 평균: {ENSEMBLE_FINAL_WF_MEAN_AUC}")
    print(f"저장 경로: {out_dir.resolve()}/")

    return {
        "models_dir": out_dir,
        "wf_plot": plot_path,
        "feature_names": feature_names,
        "walk_forward_df": wf_df,
        "split_auc_reference": ENSEMBLE_FINAL_SPLIT_AUC,
        "wf_mean_reference": ENSEMBLE_FINAL_WF_MEAN_AUC,
    }


def run_ensemble_pipeline(
    db_path: str | Path | None = None,
    *,
    version: str = "v3_clean",
    target: str = "target_updown_t28",
    test_start_date: str = "2024-01-01",
    config_path: Path | None = None,
) -> dict[str, Any]:
    """
    XGBoost(``V3_CLEAN_T28_XGB_PARAMS``) + LightGBM + CatBoost 학습 후
    ``predict_proba`` 산술평균으로 단일 split 및 walk-forward AUC-ROC를 산출한다.
    """
    db = Path(db_path) if db_path else DEFAULT_DB_PATH
    if not db.is_file():
        raise FileNotFoundError(f"DB 파일이 없습니다: {db.resolve()}")

    with sqlite3.connect(db) as conn:
        X, y, _feature_names = load_features(
            conn, version=version, target_col=target, config_path=config_path
        )

    X_train, y_train, X_test, y_test = time_series_split(X, y, test_start_date=test_start_date)

    xgb_m, lgb_m, cat_m = _fit_ensemble_base_models(X_train, y_train)

    proba = (
        xgb_m.predict_proba(X_test)[:, 1]
        + lgb_m.predict_proba(X_test)[:, 1]
        + cat_m.predict_proba(X_test)[:, 1]
    ) / 3.0
    if len(np.unique(y_test)) < 2:
        auc_split = float("nan")
    else:
        auc_split = float(roc_auc_score(y_test, proba))

    print(f"\n[run_ensemble_pipeline] 단일 split AUC-ROC (앙상블): {auc_split:.4f}")

    wf_df = walk_forward_ensemble_average(X, y)
    wf_mean = float(wf_df["auc_roc"].mean())
    print(f"[run_ensemble_pipeline] Walk-forward 평균 AUC-ROC (앙상블): {wf_mean:.4f}")

    return {
        "version": version,
        "target": target,
        "auc_split": auc_split,
        "walk_forward_mean_auc": wf_mean,
        "walk_forward_df": wf_df,
    }


def run_challenge_070_experiments(
    db_path: str | Path | None = None,
    *,
    test_start_date: str = "2024-01-01",
    config_path: Path | None = None,
    n_trials: int = 50,
    rebuild_master: bool = True,
) -> dict[str, Any]:
    """
    1) ``rebuild_master``이면 ``build_master.run_pipeline``으로 master_daily 갱신
    2) 실험1: v3_clean + target_thresh_t28, Optuna ``n_trials`` + WF(튜닝 최적 파라미터)
    3) 실험2: v3_minimal + target_updown_t28, Optuna ``n_trials`` + WF
    4) 실험3: v3_clean 앙상블(XGB+LGBM+Cat, 확률 평균)

    마지막에 기준 행(고정 참고 AUC)과 함께 비교표를 출력한다.
    """
    from features.build_master import run_pipeline as build_master_run_pipeline

    db = Path(db_path) if db_path else DEFAULT_DB_PATH
    cfg_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH

    if rebuild_master:
        print("\n[challenge_070] (1) build_master_daily / run_pipeline 실행 …")
        build_master_run_pipeline(config_path=cfg_path, db_path=db)

    if not db.is_file():
        raise FileNotFoundError(f"DB 없음: {db.resolve()}")

    print("\n[challenge_070] (2) 실험1: v3_clean + target_thresh_t28 …")
    pack_thresh = run_tuned_pipeline(
        db_path=db,
        version="v3_clean",
        n_trials=n_trials,
        test_start_date=test_start_date,
        target="target_thresh_t28",
        config_path=cfg_path,
        artifact_tag="thresh_t28",
    )
    wf_thresh = walk_forward_validation(
        db_path=db,
        version="v3_clean",
        target="target_thresh_t28",
        config_path=cfg_path,
        xgb_params=pack_thresh["tune_result"]["best_params"],
    )
    auc_thresh = float(pack_thresh["metrics"]["auc_roc"])
    wf_mean_thresh = float(wf_thresh["auc_roc"].mean())

    print("\n[challenge_070] (3) 실험2: v3_minimal + target_updown_t28 …")
    pack_min = run_tuned_pipeline(
        db_path=db,
        version="v3_minimal",
        n_trials=n_trials,
        test_start_date=test_start_date,
        target="target_updown_t28",
        config_path=cfg_path,
        artifact_tag="minimal_t28",
    )
    wf_min = walk_forward_validation(
        db_path=db,
        version="v3_minimal",
        target="target_updown_t28",
        config_path=cfg_path,
        xgb_params=pack_min["tune_result"]["best_params"],
    )
    auc_min = float(pack_min["metrics"]["auc_roc"])
    wf_mean_min = float(wf_min["auc_roc"].mean())

    print("\n[challenge_070] (4) 실험3: v3_clean 앙상블 …")
    ens = run_ensemble_pipeline(
        db_path=db,
        version="v3_clean",
        target="target_updown_t28",
        test_start_date=test_start_date,
        config_path=cfg_path,
    )
    auc_ens = float(ens["auc_split"])
    wf_mean_ens = float(ens["walk_forward_mean_auc"])

    print("\n=== 0.70 도전 결과 ===")
    print(f"{'실험':<28} {'AUC(split)':>12} {'WF평균':>10}")
    print(
        f"{'v3_clean t28 (기준)':<28} "
        f"{V3_CLEAN_T28_REFERENCE_SPLIT_AUC:>12.4f} {V3_CLEAN_T28_REFERENCE_WF_MEAN_AUC:>10.4f}"
    )
    print(f"{'v3_minimal t28':<28} {auc_min:>12.4f} {wf_mean_min:>10.4f}")
    print(f"{'v3_clean thresh_t28':<28} {auc_thresh:>12.4f} {wf_mean_thresh:>10.4f}")
    print(f"{'앙상블 v3_clean t28':<28} {auc_ens:>12.4f} {wf_mean_ens:>10.4f}")

    return {
        "baseline_split_auc_ref": V3_CLEAN_T28_REFERENCE_SPLIT_AUC,
        "baseline_wf_mean_ref": V3_CLEAN_T28_REFERENCE_WF_MEAN_AUC,
        "minimal_t28": {"split_auc": auc_min, "wf_mean_auc": wf_mean_min, "tuned": pack_min},
        "thresh_t28": {"split_auc": auc_thresh, "wf_mean_auc": wf_mean_thresh, "tuned": pack_thresh},
        "ensemble": ens,
    }


def run_xgboost_pipeline(
    db_path: str | Path | None = None,
    version: str = "v1",
    test_start_date: str = "2024-01-01",
    *,
    target: str | None = None,
    config_path: Path | None = None,
    model_save_path: Path | None = None,
    shap_summary_path: Path | None = None,
) -> dict[str, Any]:
    """DB 로드 → 피처 추출 → 시계열 분할 → 학습 → 평가 → 모델 저장까지 일괄 실행."""
    tcol = target or TARGET_COL
    db = Path(db_path) if db_path else DEFAULT_DB_PATH
    if not db.is_file():
        raise FileNotFoundError(f"DB 파일이 없습니다: {db.resolve()}")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = model_save_path or (MODELS_DIR / f"xgb_{version}.json")
    shap_path = shap_summary_path or (NOTEBOOKS_DIR / f"shap_summary_{version}.png")

    with sqlite3.connect(db) as conn:
        X, y, feature_names = load_features(
            conn, version=version, target_col=tcol, config_path=config_path
        )

    X_train, y_train, X_test, y_test = time_series_split(X, y, test_start_date=test_start_date)

    model = train_xgboost(X_train, y_train)
    metrics = evaluate_model(
        model, X_test, y_test, feature_names, shap_summary_path=shap_path
    )

    # JSON 포맷 (XGBoost sklearn API) — 버전별 파일명
    try:
        model.save_model(str(model_path))
    except AttributeError:  # pragma: no cover
        model.get_booster().save_model(str(model_path))
    print(f"\n[run_xgboost_pipeline] 모델 저장: {model_path.resolve()}")

    # 기간·행 수·핵심 지표 요약
    tr_idx, te_idx = X_train.index, X_test.index
    print("\n========== 파이프라인 요약 ==========")
    print(f"피처 버전: {version!r}")
    if len(tr_idx):
        print(f"train 기간: {tr_idx.min().date()} ~ {tr_idx.max().date()} | 행 수: {len(tr_idx):,}")
    else:
        print("train 기간: (데이터 없음)")
    if len(te_idx):
        print(f"test 기간:  {te_idx.min().date()} ~ {te_idx.max().date()} | 행 수: {len(te_idx):,}")
    else:
        print("test 기간: (데이터 없음)")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"AUC-ROC:   {metrics['auc_roc']:.4f}" if not np.isnan(metrics["auc_roc"]) else "AUC-ROC:   N/A")
    print("SHAP 상위 3개 피처:")
    for name, val in metrics["shap_top10"][:3]:
        print(f"  - {name}: {val:.6f}")
    print("=====================================\n")

    return {
        "target": tcol,
        "version": version,
        "metrics": metrics,
        "train_index": tr_idx,
        "test_index": te_idx,
    }


def run_v1_stationary_rerun_benchmark(
    db_path: str | Path | None = None,
    *,
    target: str = TARGET_COL,
    config_path: Path | None = None,
    n_trials: int = 50,
    test_start_date: str = "2024-01-01",
) -> dict[str, Any]:
    """
    master_daily 재빌드 후 ``v1_stationary`` 기준점 재실험.

    1) 고정 하이퍼 ``run_xgboost_pipeline`` → ``RERUN_MODEL_PATH``, ``RERUN_SHAP_PATH``
    2) ``run_tuned_pipeline`` (단일 split AUC·Accuracy)
    3) ``run_walk_forward`` (평균 AUC)

    요약 출력. 반환 dict는 프로그램용 요약(파일 추가 저장 없음).
    """
    ver = "v1_stationary"
    print("\n[v1_stationary 재실험 — 기준점] 실행 중 …\n")

    base_metrics = run_xgboost_pipeline(
        db_path=db_path,
        version=ver,
        test_start_date=test_start_date,
        target=target,
        config_path=config_path,
        model_save_path=RERUN_MODEL_PATH,
        shap_summary_path=RERUN_SHAP_PATH,
    )

    tuned_pack = run_tuned_pipeline(
        db_path=db_path,
        version=ver,
        n_trials=n_trials,
        test_start_date=test_start_date,
        target=target,
        config_path=config_path,
    )

    wf_df = run_walk_forward(db_path=db_path, version=ver, target=target, config_path=config_path)
    wf_mean = float(wf_df["auc_roc"].mean())

    tr_idx = base_metrics["train_index"]
    te_idx = base_metrics["test_index"]
    m_base = base_metrics["metrics"]
    m_tune = tuned_pack["metrics"]

    shap3_lines = [f"{n} ({v:.6f})" for n, v in m_tune["shap_top10"][:3]]

    print("[v1_stationary 재실험 — 기준점]")
    if len(tr_idx):
        print(f"train 기간: {tr_idx.min().date()} ~ {tr_idx.max().date()}")
    else:
        print("train 기간: (없음)")
    if len(te_idx):
        print(f"test 기간:  {te_idx.min().date()} ~ {te_idx.max().date()}")
    else:
        print("test 기간: (없음)")
    print(f"Accuracy (튜닝 split test): {m_tune['accuracy']:.4f}")
    auc_simple = m_tune["auc_roc"]
    print(
        f"AUC-ROC (단순 split): {auc_simple:.4f}"
        if not np.isnan(auc_simple)
        else "AUC-ROC (단순 split): N/A"
    )
    print(f"AUC-ROC (Walk-forward 평균): {wf_mean:.4f}")
    print("SHAP 상위 3개 피처 (튜닝 모델, 테스트 구간):")
    for line in shap3_lines:
        print(f"  - {line}")
    print(
        f"이전 결과 (참고): AUC-ROC {PREVIOUS_SINGLE_SPLIT_AUC:.4f} / "
        f"Walk-forward {PREVIOUS_WALK_FORWARD_MEAN_AUC:.4f}"
    )

    out: dict[str, Any] = {
        "version": ver,
        "target": target,
        "test_start_date": test_start_date,
        "train_date_min": str(tr_idx.min().date()) if len(tr_idx) else None,
        "train_date_max": str(tr_idx.max().date()) if len(tr_idx) else None,
        "test_date_min": str(te_idx.min().date()) if len(te_idx) else None,
        "test_date_max": str(te_idx.max().date()) if len(te_idx) else None,
        "baseline_accuracy": m_base["accuracy"],
        "baseline_auc_roc": m_base["auc_roc"],
        "tuned_accuracy": m_tune["accuracy"],
        "tuned_auc_roc": m_tune["auc_roc"],
        "tuned_cv_best_auc": tuned_pack["tune_result"]["best_auc_cv"],
        "walk_forward_mean_auc": wf_mean,
        "shap_top3_tuned": m_tune["shap_top10"][:3],
        "previous_reference_single_split_auc": PREVIOUS_SINGLE_SPLIT_AUC,
        "previous_reference_walk_forward_mean_auc": PREVIOUS_WALK_FORWARD_MEAN_AUC,
        "artifacts": {
            "model_baseline": str(RERUN_MODEL_PATH.resolve()),
            "shap_baseline": str(RERUN_SHAP_PATH.resolve()),
            "model_tuned": str((MODELS_DIR / f"xgb_{ver}_tuned.json").resolve()),
        },
    }
    return out


def run_v2_interview_experiments(
    db_path: str | Path | None = None,
    *,
    test_start_date: str = "2024-01-01",
    config_path: Path | None = None,
) -> dict[str, Any]:
    """
    ``v2_interview``에서 이진 타깃 2개를 각각 실험한다.

    - A: ``target_updown_t7``
    - B: ``target_updown_t28``
    """
    ver = "v2_interview"
    targets = [
        ("t7", "target_updown_t7"),
        ("t28", "target_updown_t28"),
    ]
    new_feature_set = {
        "dxy_close",
        "dxy_lag_1",
        "dxy_return_7d",
        "dxy_volatility_14d",
        "canola_close",
        "canola_lag_1",
        "canola_return_7d",
        "sunflower_close",
        "sunflower_return_30d",
        "biodiesel_production_kbbl",
        "biodiesel_chg_4w",
        "market_avg_price_30d",
        "price_vs_market_avg",
    }
    results: dict[str, Any] = {}

    for short, tcol in targets:
        print(f"\n[v2_interview] 실험 {short.upper()} 시작 — target={tcol}")
        pack = run_xgboost_pipeline(
            db_path=db_path,
            version=ver,
            test_start_date=test_start_date,
            target=tcol,
            config_path=config_path,
            model_save_path=MODELS_DIR / f"xgb_v2_interview_{short}.json",
            shap_summary_path=NOTEBOOKS_DIR / f"shap_summary_v2_interview_{short}.png",
        )
        metrics = pack["metrics"]
        contrib_all: dict[str, float] = metrics.get("shap_mean_abs", {})
        contrib_new = sorted(
            [
                (k, v)
                for k, v in contrib_all.items()
                if k in new_feature_set
            ],
            key=lambda x: -x[1],
        )
        print(f"[v2_interview] 신규 피처 SHAP 기여도 상위 (target={tcol}):")
        if contrib_new:
            for i, (nm, v) in enumerate(contrib_new[:5], start=1):
                print(f"  {i}. {nm}: {v:.6f}")
        else:
            print("  - 신규 피처 기여도 계산 결과 없음")

        results[short] = {
            "target": tcol,
            "accuracy": float(metrics["accuracy"]),
            "auc_roc": float(metrics["auc_roc"]),
            "train_start": str(pack["train_index"].min().date()) if len(pack["train_index"]) else None,
            "train_end": str(pack["train_index"].max().date()) if len(pack["train_index"]) else None,
            "test_start": str(pack["test_index"].min().date()) if len(pack["test_index"]) else None,
            "test_end": str(pack["test_index"].max().date()) if len(pack["test_index"]) else None,
        }

    print("\n=== XGBoost 버전별 비교 ===")
    print("버전              타깃   AUC-ROC   비고")
    print(f"v1_stationary    t7     {PREVIOUS_SINGLE_SPLIT_AUC:.4f}   기존 베이스라인")
    print(f"v1_stationary    t7     {RERUN_SINGLE_SPLIT_AUC:.4f}   재빌드 후 기준점")
    t7_auc = results["t7"]["auc_roc"]
    t28_auc = results["t28"]["auc_roc"]
    print(f"v2_interview     t7     {t7_auc:.4f}   인터뷰 반영")
    print(f"v2_interview     t28    {t28_auc:.4f}   현업 주기 반영")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="XGBoost master_daily 파이프라인")
    parser.add_argument(
        "--version",
        default="v1",
        help="features.yaml 의 versions 키 (예: v1, v1_stationary)",
    )
    parser.add_argument(
        "--tuned",
        action="store_true",
        help="Optuna 튜닝 파이프라인(run_tuned_pipeline) 실행",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Optuna 시도 횟수 (--tuned, --rerun-v1-stationary, --challenge-070)",
    )
    parser.add_argument(
        "--walkforward",
        action="store_true",
        help="Walk-forward 검증(run_walk_forward) 실행",
    )
    parser.add_argument(
        "--target",
        default=TARGET_COL,
        help="이진 타깃 컬럼명 (예: target_updown_t7)",
    )
    parser.add_argument(
        "--rerun-v1-stationary",
        action="store_true",
        help="v1_stationary 기준점 재실험(고정학습+튜닝+워크포워드) 및 rerun 산출물 저장",
    )
    parser.add_argument(
        "--v2-interview-exp",
        action="store_true",
        help="v2_interview에서 t7/t28 실험 및 비교표 출력",
    )
    parser.add_argument(
        "--challenge-070",
        action="store_true",
        help="build_master(선택) 후 thresh/minimal Optuna50 + v3_clean 앙상블 및 0.70 비교표",
    )
    parser.add_argument(
        "--no-rebuild-master",
        action="store_true",
        help="--challenge-070 시 master_daily 재빌드 생략",
    )
    parser.add_argument(
        "--save-ensemble-final",
        action="store_true",
        help="앙상블 최종본 학습·저장 및 WF 플롯 (models/ensemble_v3_clean_t28_final/)",
    )
    args = parser.parse_args()
    if args.rerun_v1_stationary:
        run_v1_stationary_rerun_benchmark(target=args.target, n_trials=args.n_trials)
    elif args.v2_interview_exp:
        run_v2_interview_experiments()
    elif args.challenge_070:
        run_challenge_070_experiments(
            rebuild_master=not args.no_rebuild_master,
            n_trials=args.n_trials,
        )
    elif args.save_ensemble_final:
        persist_ensemble_v3_clean_t28_final()
    elif args.tuned:
        run_tuned_pipeline(version=args.version, n_trials=args.n_trials, target=args.target)
    elif args.walkforward:
        run_walk_forward(db_path=None, version=args.version, target=args.target)
    else:
        run_xgboost_pipeline(version=args.version, target=args.target)
