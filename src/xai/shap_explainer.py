from __future__ import annotations

import re
import sqlite3
import sys
import warnings
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from utils.plot_style import configure_matplotlib_korean
from xgboost import XGBClassifier

from models.xgboost_model import DEFAULT_CONFIG_PATH, load_features

configure_matplotlib_korean()

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "db" / "soybean.db"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "xgb_v1_stationary_final.json"
V1_FINAL_MODEL_PATH = PROJECT_ROOT / "models" / "xgb_v1_stationary_final.json"
V2_T7_FINAL_MODEL_PATH = PROJECT_ROOT / "models" / "xgb_v2_t7_final.json"
V2_T28_FINAL_MODEL_PATH = PROJECT_ROOT / "models" / "xgb_v2_t28_final.json"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"

# 콘솔·반사실 서술용 (없으면 기술명 그대로)
FEATURE_LABELS_KO: dict[str, str] = {
    "wasde_soyoil_stock_to_use": "WASDE 대두유 재고·사용 비(stock-to-use)",
    "wasde_soy_prod_brazil": "WASDE 브라질 대두 생산",
    "wasde_world_production": "WASDE 세계 대두 생산",
    "usd_brl_return_7d": "달러/헤알 7일 수익률",
    "usd_brl_volatility_14d": "달러/헤알 14일 변동성",
    "usd_brl_lag_1": "달러/헤알 전일 값",
    "wti_return_7d": "WTI 7일 수익률",
    "wti_volatility_14d": "WTI 14일 변동성",
    "wti_lag_1": "WTI 전일 값",
    "crush_spread": "크러시 스프레드",
    "crush_spread_ma7": "크러시 스프레드 7일 평균",
    "crush_spread_lag_1": "크러시 스프레드 전일 값",
    "cftc_noncomm_net": "CFTC 비상업 순포지션",
    "cftc_noncomm_net_chg_1w": "CFTC 비상업 순포지션 주간 변화",
    "cftc_long_short_ratio": "CFTC 롱숏 비율",
    "return_1d": "1일 수익률",
    "return_3d": "3일 수익률",
    "return_7d": "7일 수익률",
    "return_14d": "14일 수익률",
    "volatility_5d": "5일 변동성",
    "volatility_10d": "10일 변동성",
    "volatility_20d": "20일 변동성",
    "price_to_ma7_ratio": "가격 / 7일 이동평균 비율",
    "price_to_ma14_ratio": "가격 / 14일 이동평균 비율",
    "price_to_ma30_ratio": "가격 / 30일 이동평균 비율",
    "month": "월",
    "week_of_year": "연중 주차",
    "quarter": "분기",
    "is_planting_season": "파종 시즌 여부",
    "is_flowering_season": "개화 시즌 여부",
    "is_harvest_season": "수확 시즌 여부",
    "days_to_next_harvest": "다음 수확까지 일수",
}


def _human_name(feature: str) -> str:
    return FEATURE_LABELS_KO.get(feature, feature.replace("_", " "))


def _safe_filename_part(name: str) -> str:
    return re.sub(r"[^\w\-.]+", "_", name).strip("_") or "feature"


def _align_X_to_model(model: XGBClassifier, X: pd.DataFrame) -> pd.DataFrame:
    booster = model.get_booster()
    names = booster.feature_names
    if not names:
        return X
    missing = set(names) - set(X.columns)
    if missing:
        raise ValueError(f"데이터에 모델 피처가 없습니다: {sorted(missing)}")
    return X[list(names)]


def _shap_values_positive_class(
    explainer: shap.TreeExplainer, X: pd.DataFrame
) -> tuple[np.ndarray, float | np.ndarray]:
    raw = explainer.shap_values(X)
    if isinstance(raw, list):
        shap_vals = np.asarray(raw[1])
    else:
        shap_vals = np.asarray(raw)
    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)) and np.size(ev) > 1:
        ev = float(np.asarray(ev).reshape(-1)[1])
    else:
        ev = float(np.asarray(ev).reshape(-1)[0])
    return shap_vals, ev


def load_model_and_data(
    db_path: str | Path,
    model_path: str | Path,
    version: str = "v1_stationary",
    *,
    target_col: str | None = None,
    config_path: Path | None = None,
) -> tuple[XGBClassifier, pd.DataFrame, pd.Series, list[str], pd.DatetimeIndex]:
    """
    XGBClassifier.load_model 로드 후 load_features와 동일 규칙으로 X, y 구성.
    반환 dates는 X와 동일한 DatetimeIndex.
    """
    db = Path(db_path)
    mp = Path(model_path)
    if not db.is_file():
        raise FileNotFoundError(f"DB 없음: {db.resolve()}")
    if not mp.is_file():
        raise FileNotFoundError(f"모델 파일 없음: {mp.resolve()}")

    cfg = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    with sqlite3.connect(db) as conn:
        X, y, feature_names = load_features(
            conn, version=version, target_col=target_col, config_path=cfg
        )

    model = XGBClassifier()
    model.load_model(str(mp))
    X = _align_X_to_model(model, X)
    feature_names = list(X.columns)
    dates = pd.DatetimeIndex(X.index)
    return model, X, y, feature_names, dates


def plot_shap_summary(
    model: XGBClassifier,
    X: pd.DataFrame,
    feature_names: list[str],
    *,
    out_path: Path | None = None,
) -> list[tuple[str, float]]:
    """TreeSHAP summary(beeswarm) 저장 및 상위 10개 |SHAP| 평균 테이블 출력."""
    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    path = Path(out_path) if out_path else NOTEBOOKS_DIR / "xai_shap_summary.png"

    explainer = shap.TreeExplainer(model)
    shap_vals, _ = _shap_values_positive_class(explainer, X)
    names = list(X.columns)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(-mean_abs)
    top10 = [(names[i], float(mean_abs[i])) for i in order[:10]]

    table = pd.DataFrame(top10, columns=["feature", "mean_abs_shap"])
    print("\n=== SHAP 요약: 상위 10개 피처 (평균 |SHAP|) ===")
    print(table.to_string(index=False))

    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_vals,
        X,
        feature_names=names,
        show=False,
        max_display=min(20, len(names)),
    )
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[plot_shap_summary] 저장: {path.resolve()}")
    return top10


def _resolve_target_row(
    X: pd.DataFrame, dates: pd.DatetimeIndex, target_date: pd.Timestamp | str | None
) -> pd.Timestamp:
    if target_date is None:
        return pd.Timestamp(X.index.max())
    t = pd.Timestamp(target_date)
    if t not in X.index:
        raise KeyError(f"인덱스에 없는 날짜: {t.date()}")
    return t


def plot_shap_waterfall(
    model: XGBClassifier,
    X: pd.DataFrame,
    feature_names: list[str],
    dates: pd.DatetimeIndex,
    target_date: pd.Timestamp | str | None = None,
    *,
    out_dir: Path | None = None,
) -> None:
    """단일 일자 워터폴 플롯 및 예측·상위 3개 기여 출력."""
    t = _resolve_target_row(X, dates, target_date)
    row = X.loc[[t]]
    proba = float(np.asarray(model.predict_proba(row)[:, 1]).reshape(-1)[0])
    print(f"\n=== Waterfall 대상일 {t.date()} ===")
    print(f"상승 확률(클래스 1): {proba * 100:.2f}%")

    explainer = shap.TreeExplainer(model)
    sv_raw = explainer.shap_values(row)
    if isinstance(sv_raw, list):
        sv = np.asarray(sv_raw[1]).reshape(-1)
    else:
        sv = np.asarray(sv_raw).reshape(-1)

    ev = explainer.expected_value
    if isinstance(ev, (list, np.ndarray)) and np.size(ev) > 1:
        base = float(np.asarray(ev).reshape(-1)[1])
    else:
        base = float(np.asarray(ev).reshape(-1)[0])

    names = list(X.columns)
    order = np.argsort(-np.abs(sv))
    print("\n기여도 상위 3개 피처 (|SHAP| 기준, 방향=상승 확률에 대한 효과):")
    for rank in range(min(3, len(names))):
        j = order[rank]
        direction = "상승 쪽" if sv[j] >= 0 else "하락 쪽"
        print(f"  {rank + 1}. {names[j]}: SHAP={sv[j]:+.6f} → {direction}")

    exp = shap.Explanation(
        values=sv,
        base_values=base,
        data=row.values.flatten(),
        feature_names=names,
    )
    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(exp, max_display=min(15, len(names)), show=False)
    plt.tight_layout()
    out = Path(out_dir) if out_dir else NOTEBOOKS_DIR
    out.mkdir(parents=True, exist_ok=True)
    fname = out / f"xai_waterfall_{t.strftime('%Y-%m-%d')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[plot_shap_waterfall] 저장: {fname.resolve()}")


def plot_shap_dependence(
    model: XGBClassifier,
    X: pd.DataFrame,
    feature_names: list[str],
    feature: str = "usd_brl_return_7d",
    *,
    out_dir: Path | None = None,
) -> None:
    """특정 피처에 대한 SHAP dependence (interaction_index='auto')."""
    if feature not in X.columns:
        raise KeyError(f"피처 없음: {feature!r}")

    explainer = shap.TreeExplainer(model)
    shap_vals, _ = _shap_values_positive_class(explainer, X)
    names = list(X.columns)
    ind = names.index(feature)

    plt.figure(figsize=(9, 6))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        shap.dependence_plot(
            ind,
            shap_vals,
            X,
            feature_names=names,
            interaction_index="auto",
            show=False,
        )
    plt.tight_layout()
    out = Path(out_dir) if out_dir else NOTEBOOKS_DIR
    out.mkdir(parents=True, exist_ok=True)
    fname = out / f"xai_dependence_{_safe_filename_part(feature)}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n[plot_shap_dependence] 저장: {fname.resolve()}")


def _proba_one(model: XGBClassifier, row_df: pd.DataFrame) -> float:
    return float(np.asarray(model.predict_proba(row_df)[:, 1]).reshape(-1)[0])


def generate_counterfactual(
    model: XGBClassifier,
    X: pd.DataFrame,
    feature_names: list[str],
    dates: pd.DatetimeIndex,
    target_date: pd.Timestamp | str | None = None,
) -> None:
    """피처별 ±10% What-if 및 단일·쌍 조합 반전 탐색 후 자연어 요약."""
    t = _resolve_target_row(X, dates, target_date)
    base = X.loc[[t]].copy()
    base_p = _proba_one(model, base)
    base_pred_up = base_p >= 0.5

    print(f"\n=== 반사실 분석 대상일 {t.date()} ===")
    print(f"현재 예측: 상승 확률 {base_p * 100:.2f}%")
    print("\n[What-if 분석]")
    flip_singles: list[tuple[str, str, float]] = []  # (feature, tag, new_p)

    for feat in feature_names:
        if feat not in base.columns:
            continue
        v0 = float(base[feat].iloc[0])
        for mult, tag in ((1.1, "+10%"), (0.9, "-10%")):
            pert = base.copy()
            pert[feat] = v0 * mult
            new_p = _proba_one(model, pert)
            delta_pct = (new_p - base_p) * 100
            print(f"{feat} {tag} → 상승확률 {new_p * 100:.2f}% (변화: {delta_pct:+.2f}%p)")
            new_up = new_p >= 0.5
            if new_up != base_pred_up:
                flip_singles.append((feat, tag, new_p))

    # --- 반전 조건 (자연어) ---
    print("\n[반전 조건]")
    lines: list[str] = []
    for feat, tag, new_p in flip_singles:
        label = _human_name(feat)
        to_down = new_p < 0.5
        adj = "값이 10% 높았다면(×1.1)" if tag == "+10%" else "값이 10% 낮았다면(×0.9)"
        outcome = "하락 쪽(상승 확률 50% 미만)" if to_down else "상승 쪽(상승 확률 50% 이상)"
        lines.append(
            f'"{label}이(가) {adj} 예측이 {outcome}으로 바뀌었을 것" '
            f"(상승확률 {new_p * 100:.1f}%)"
        )

    # 쌍 조합: |SHAP| 상위 피처 위주
    explainer = shap.TreeExplainer(model)
    sv_raw = explainer.shap_values(base)
    if isinstance(sv_raw, list):
        sv = np.asarray(sv_raw[1]).reshape(-1)
    else:
        sv = np.asarray(sv_raw).reshape(-1)
    cols = list(X.columns)
    top_idx = np.argsort(-np.abs(sv))[:8]
    top_feats = [cols[i] for i in top_idx if i < len(cols) and cols[i] in base.columns]

    pair_found = 0
    for a in range(len(top_feats)):
        for b in range(a + 1, len(top_feats)):
            f1, f2 = top_feats[a], top_feats[b]
            for m1, t1 in ((1.1, "+10%"), (0.9, "-10%")):
                for m2, t2 in ((1.1, "+10%"), (0.9, "-10%")):
                    pert = base.copy()
                    pert[f1] = float(base[f1].iloc[0]) * m1
                    pert[f2] = float(base[f2].iloc[0]) * m2
                    new_p = _proba_one(model, pert)
                    if (new_p >= 0.5) != base_pred_up:
                        pair_found += 1
                        if pair_found <= 5:
                            l1, l2 = _human_name(f1), _human_name(f2)
                            side_ko = "하락" if new_p < 0.5 else "상승"
                            lines.append(
                                f'"{l1}({t1})와 {l2}({t2})를 동시에 적용하면 예측이 '
                                f'{side_ko} 쪽으로 바뀜" (상승확률 {new_p * 100:.1f}%)'
                            )
            if pair_found >= 5:
                break
        if pair_found >= 5:
            break

    if not lines:
        print(
            "단일 피처 ±10% 및 주요 피처 쌍(±10% 조합) 탐색에서 예측 반전이 발견되지 않았습니다."
        )
    else:
        for s in lines:
            print(s)


def run_xai_pipeline(
    db_path: str | Path | None = None,
    model_path: str | Path | None = None,
    version: str = "v1_stationary",
    *,
    config_path: Path | None = None,
) -> None:
    """전체 XAI 파이프라인: 요약 → 워터폴 → 의존성(1위 피처) → 반사실."""
    db = Path(db_path) if db_path else DEFAULT_DB_PATH
    mp = Path(model_path) if model_path else DEFAULT_MODEL_PATH

    model, X, y, feature_names, dates = load_model_and_data(
        db, mp, version=version, config_path=config_path
    )
    top10 = plot_shap_summary(model, X, feature_names)
    target = X.index.max()
    plot_shap_waterfall(model, X, feature_names, dates, target_date=target)
    top_feature = top10[0][0]
    print(f"\n[run_xai_pipeline] Dependence 피처(SHAP 1위): {top_feature}")
    plot_shap_dependence(model, X, feature_names, feature=top_feature)
    generate_counterfactual(model, X, feature_names, dates, target_date=target)


def _top_shap_table(
    model: XGBClassifier, X: pd.DataFrame, *, top_n: int = 15
) -> pd.DataFrame:
    """모델별 평균 |SHAP| 상위 N 테이블."""
    explainer = shap.TreeExplainer(model)
    shap_vals, _ = _shap_values_positive_class(explainer, X)
    names = list(X.columns)
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order = np.argsort(-mean_abs)
    rows = [
        {"rank": i + 1, "feature": names[j], "mean_abs_shap": float(mean_abs[j])}
        for i, j in enumerate(order[:top_n])
    ]
    return pd.DataFrame(rows)


def _feature_rank_and_value(df_top: pd.DataFrame, feature: str) -> tuple[int | None, float | None]:
    hit = df_top.loc[df_top["feature"] == feature]
    if hit.empty:
        return None, None
    r = int(hit["rank"].iloc[0])
    v = float(hit["mean_abs_shap"].iloc[0])
    return r, v


def compare_shap_versions(db_path: str | Path | None = None) -> None:
    """
    v1_stationary(final), v2_interview(t7 final), v2_interview(t28 final) SHAP 상위 15개 비교.

    출력:
    - notebooks/shap_comparison_final.png
    - Q1~Q3 자동 해석 콘솔 출력
    """
    db = Path(db_path) if db_path else DEFAULT_DB_PATH
    if not db.is_file():
        raise FileNotFoundError(f"DB 없음: {db.resolve()}")

    m1, X1, _, _, _ = load_model_and_data(
        db, V1_FINAL_MODEL_PATH, version="v1_stationary", target_col="target_updown_t7"
    )
    m2, X2, _, _, _ = load_model_and_data(
        db, V2_T7_FINAL_MODEL_PATH, version="v2_interview", target_col="target_updown_t7"
    )
    m3, X3, _, _, _ = load_model_and_data(
        db, V2_T28_FINAL_MODEL_PATH, version="v2_interview", target_col="target_updown_t28"
    )

    top1 = _top_shap_table(m1, X1, top_n=15)
    top2 = _top_shap_table(m2, X2, top_n=15)
    top3 = _top_shap_table(m3, X3, top_n=15)

    NOTEBOOKS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = NOTEBOOKS_DIR / "shap_comparison_final.png"
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), sharex=False)

    a1 = top1.sort_values("mean_abs_shap", ascending=True)
    axes[0].barh(a1["feature"], a1["mean_abs_shap"], color="steelblue", alpha=0.9)
    axes[0].set_title("v1_stationary (final) SHAP Top15")
    axes[0].set_xlabel("mean |SHAP|")

    a2 = top2.sort_values("mean_abs_shap", ascending=True)
    axes[1].barh(a2["feature"], a2["mean_abs_shap"], color="darkorange", alpha=0.9)
    axes[1].set_title("v2_interview (t7 final) SHAP Top15")
    axes[1].set_xlabel("mean |SHAP|")

    a3 = top3.sort_values("mean_abs_shap", ascending=True)
    axes[2].barh(a3["feature"], a3["mean_abs_shap"], color="seagreen", alpha=0.9)
    axes[2].set_title("v2_interview (t28 final) SHAP Top15")
    axes[2].set_xlabel("mean |SHAP|")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[compare_shap_versions] 저장: {out_path.resolve()}")

    # Q1
    r_u, v_u = _feature_rank_and_value(top1, "usd_brl_lag_1")
    r_d, v_d = _feature_rank_and_value(top2, "dxy_close")
    print("\nQ1: DXY vs usd_brl_lag_1")
    print(
        "  v1_stationary: usd_brl_lag_1 순위: "
        f"{r_u if r_u is not None else 'Top15 밖'}위 (SHAP: {v_u:.4f})"
        if v_u is not None
        else "  v1_stationary: usd_brl_lag_1 순위: Top15 밖 (SHAP: N/A)"
    )
    print(
        "  v2_interview:  dxy_close 순위: "
        f"{r_d if r_d is not None else 'Top15 밖'}위 (SHAP: {v_d:.4f})"
        if v_d is not None
        else "  v2_interview:  dxy_close 순위: Top15 밖 (SHAP: N/A)"
    )
    if v_d is not None and v_u is not None and v_d > v_u:
        print('  → "인터뷰 방향 검증됨"')
    else:
        print('  → "BRL이 여전히 더 강한 신호"')

    # Q2
    r_can, v_can = _feature_rank_and_value(top2, "canola_close")
    r_sun, v_sun = _feature_rank_and_value(top2, "sunflower_close")
    r_mavg, v_mavg = _feature_rank_and_value(top2, "market_avg_price_30d")
    r_eia, v_eia = _feature_rank_and_value(top2, "soybean_oil_biofuel_mmlb")
    print("\nQ2: 신규 피처 기여도")
    print(
        f"  canola_close SHAP: {v_can:.4f} (순위: {r_can}위)"
        if v_can is not None
        else "  canola_close SHAP: N/A (순위: Top15 밖 또는 컬럼 없음)"
    )
    print(
        f"  sunflower_close SHAP: {v_sun:.4f} (순위: {r_sun}위)"
        if v_sun is not None
        else "  sunflower_close SHAP: N/A (순위: Top15 밖 또는 컬럼 없음)"
    )
    print(
        f"  market_avg_price_30d SHAP: {v_mavg:.4f} (순위: {r_mavg}위)"
        if v_mavg is not None
        else "  market_avg_price_30d SHAP: N/A (순위: Top15 밖 또는 컬럼 없음)"
    )
    print(
        f"  soybean_oil_biofuel_mmlb SHAP: {v_eia:.4f} (순위: {r_eia}위)"
        if v_eia is not None
        else "  soybean_oil_biofuel_mmlb SHAP: N/A (순위: Top15 밖 또는 컬럼 없음)"
    )
    eval_pairs = [
        ("canola_close", r_can),
        ("sunflower_close", r_sun),
        ("market_avg_price_30d", r_mavg),
        ("soybean_oil_biofuel_mmlb", r_eia),
    ]
    for feat, rank in eval_pairs:
        if rank is not None and rank <= 10:
            print(f"  - {feat}: 유효한 피처")
        else:
            print(f"  - {feat}: v3에서 재검토")

    # Q3: v2_interview t7 vs t28
    top_t7 = top2["feature"].head(5).tolist()
    top_t28 = top3["feature"].head(5).tolist()
    inter = sorted(set(top_t7) & set(top_t28))
    print("\nQ3: t7 vs t28 피처 중요도 차이")
    print(f"  t7 상위5: {top_t7}")
    print(f"  t28 상위5: {top_t28}")
    print(f"  공통 피처 수: {len(inter)}개 ({inter})")
    if top_t7 != top_t28:
        print('  → "예측 시계에 따라 다른 피처가 중요"')
    else:
        print("  → 상위 피처 구성이 동일")


if __name__ == "__main__":
    run_xai_pipeline()
