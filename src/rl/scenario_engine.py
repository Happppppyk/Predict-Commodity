"""
TFT 분위수(P10/P50/P90)와 master_daily 최신 가격을 바탕으로
Buy/Split/Wait 권고를 산출하고 decision_log를 기록한다.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "db" / "soybean.db"

DEFAULT_LEAD_DAYS = 50
WAREHOUSE_MONTHS = 4.0
MOQ_MIN_TON = 2000.0
CBOT_TO_USD_PER_TON = 22.0462
SPLIT_INTERVAL_DAYS = 7


@dataclass(frozen=True)
class TFTQuantileForecast:
    """TFT 분위수 예측 (CBOT 달러/100파운드)."""

    p10: float
    p50: float
    p90: float


@dataclass
class ScenarioRequest:
    """현업 입력."""

    current_inventory_ton: float
    safety_stock_ton: float
    quantity_ton: float
    monthly_consumption_ton: float
    open_po_ton: float = 0.0


def _parse_date(s: str) -> date:
    return datetime.strptime(s[:10], "%Y-%m-%d").date()


def _iso_now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def cbot_to_usd_per_ton(cbot_price: float) -> float:
    return cbot_price * CBOT_TO_USD_PER_TON


def calc_inventory_at_due(
    current_inventory_ton: float,
    monthly_consumption_ton: float,
    days_to_due: int = DEFAULT_LEAD_DAYS,
) -> float:
    consumption = monthly_consumption_ton * (days_to_due / 30.0)
    return current_inventory_ton - consumption


def load_master_procurement_context(
    conn: sqlite3.Connection,
    *,
    as_of_date: date | None = None,
) -> dict[str, Any]:
    cols = {str(r[1]) for r in conn.execute("PRAGMA table_info(master_daily)").fetchall()}
    for c in ("price_close", "market_avg_price_30d"):
        if c not in cols:
            raise KeyError(f"master_daily에 {c} 컬럼이 없습니다.")

    if as_of_date is not None:
        row = conn.execute(
            """
            SELECT date, price_close, market_avg_price_30d
            FROM master_daily
            WHERE date <= ?
            ORDER BY date DESC
            LIMIT 1
            """,
            (as_of_date.isoformat(),),
        ).fetchone()
    else:
        row = conn.execute(
            """
            SELECT date, price_close, market_avg_price_30d
            FROM master_daily
            ORDER BY date DESC
            LIMIT 1
            """
        ).fetchone()

    if row is None:
        raise RuntimeError("master_daily에 행이 없습니다.")

    pc, mkt = row[1], row[2]
    if pc is None:
        raise RuntimeError("최신 행의 price_close가 NULL입니다.")
    return {
        "as_of_date": _parse_date(str(row[0])),
        "current_price_cbot": float(pc),
        "market_avg_price_30d_cbot": float(mkt) if mkt is not None else None,
        "current_price": cbot_to_usd_per_ton(float(pc)),
        "market_avg_price_30d": cbot_to_usd_per_ton(float(mkt)) if mkt is not None else None,
    }


def create_decision_log_table(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS decision_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            decision_date TEXT,
            current_price REAL,
            market_avg_price REAL,
            p10 REAL,
            p50 REAL,
            p90 REAL,
            current_inventory_ton REAL,
            safety_stock_ton REAL,
            quantity_requested_ton REAL,
            recommended_action TEXT,
            actual_action TEXT,
            quantity_actual_ton REAL,
            reason_override TEXT,
            created_at TEXT
        )
        """
    )
    conn.commit()


def save_recommendation(
    scenario_result: dict[str, Any],
    db_path: str | Path | None = None,
) -> int:
    db = Path(db_path) if db_path else DEFAULT_DB_PATH
    with sqlite3.connect(str(db)) as conn:
        create_decision_log_table(conn)
        cur = conn.execute(
            """
            INSERT INTO decision_log (
                decision_date, current_price, market_avg_price,
                p10, p50, p90,
                current_inventory_ton, safety_stock_ton, quantity_requested_ton,
                recommended_action, actual_action, quantity_actual_ton, reason_override,
                created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL, NULL, ?)
            """,
            (
                scenario_result["decision_date"],
                scenario_result["current_price"],
                scenario_result.get("market_avg_price_30d"),
                scenario_result["forecast"]["p10"],
                scenario_result["forecast"]["p50"],
                scenario_result["forecast"]["p90"],
                scenario_result["current_inventory_ton"],
                scenario_result["safety_stock_ton"],
                scenario_result["quantity_ton"],
                scenario_result["recommended_action"],
                scenario_result.get("created_at") or _iso_now(),
            ),
        )
        conn.commit()
        return int(cur.lastrowid)


def update_actual_decision(
    row_id: int,
    actual_action: str,
    quantity_actual_ton: float | None,
    reason_override: str | None,
    db_path: str | Path | None = None,
) -> None:
    db = Path(db_path) if db_path else DEFAULT_DB_PATH
    with sqlite3.connect(str(db)) as conn:
        conn.execute(
            """
            UPDATE decision_log
            SET actual_action = ?, quantity_actual_ton = ?, reason_override = ?
            WHERE id = ?
            """,
            (actual_action, quantity_actual_ton, reason_override, row_id),
        )
        conn.commit()


def _recommend_action(*, inv: float, safety: float, current_price: float, p50: float) -> tuple[str, str]:
    if inv < safety:
        return "Buy", "inventory_below_safety"
    if p50 > current_price * 1.03:
        return "Buy", "expected_price_up_3pct"
    if p50 < current_price * 0.97:
        return "Wait", "expected_price_down_3pct"
    return "Split", "expected_price_neutral_band"


def _reason_text(code: str, *, p50: float, current_price: float, inv: float, safety: float) -> str:
    pct = ((p50 - current_price) / current_price * 100.0) if current_price else 0.0
    if code == "inventory_below_safety":
        return f"실질재고 {inv:.0f}톤이 안전재고 {safety:.0f}톤 미만이므로 Buy를 권고합니다."
    if code == "expected_price_up_3pct":
        return f"P50 예상가({p50:,.0f}달러/톤)가 현재가({current_price:,.0f}달러/톤)보다 {pct:.2f}% 높아 Buy가 유리합니다."
    if code == "expected_price_down_3pct":
        return f"P50 예상가({p50:,.0f}달러/톤)가 현재가({current_price:,.0f}달러/톤)보다 {abs(pct):.2f}% 낮아 Wait가 유리합니다."
    if code == "wait_blocked_inventory_due":
        return "가격 신호는 Wait지만 납기 시점 재고가 안전재고 미달로 예상되어 Split으로 전환했습니다."
    return f"P50 예상가가 현재가 대비 ±3% 이내라 Split을 권고합니다."


def _option_block(
    *,
    name: str,
    expected_cost_usd: float,
    risk_p10_total_usd: float,
    risk_p90_total_usd: float,
    reward_usd: float,
    feasible: bool,
    notes: str,
) -> dict[str, Any]:
    return {
        "action": name,
        "expected_cost_usd": round(expected_cost_usd, 2),
        "risk": {
            "p10_total_usd": round(risk_p10_total_usd, 2),
            "p90_total_usd": round(risk_p90_total_usd, 2),
            "span_usd": round(risk_p90_total_usd - risk_p10_total_usd, 2),
        },
        "reward_usd": round(reward_usd, 2),
        "feasible": feasible,
        "notes": notes,
    }


def _build_decision_summary(options: dict[str, dict[str, Any]], recommended_action: str) -> dict[str, Any]:
    opt = options[recommended_action]
    violations = [
        {"action": k, "reason": v["notes"]}
        for k, v in options.items()
        if not v["feasible"]
    ]
    return {
        "recommendation": recommended_action,
        "expected_cost": opt["expected_cost_usd"],
        "risk_range": opt["risk"],
        "reward_usd": opt["reward_usd"],
        "reward_vs_market_pct": None,
        "constraint_violations": violations,
    }


def evaluate_procurement_scenario(
    req: ScenarioRequest,
    forecast: TFTQuantileForecast,
    *,
    db_path: str | Path | None = None,
    as_of_date: date | None = None,
    persist_recommendation: bool = False,
) -> dict[str, Any]:
    if req.quantity_ton <= 0 or req.monthly_consumption_ton <= 0:
        raise ValueError("quantity_ton, monthly_consumption_ton은 양수여야 합니다.")
    if forecast.p10 > forecast.p50 or forecast.p50 > forecast.p90:
        raise ValueError("분위수 순서는 p10 <= p50 <= p90 이어야 합니다.")

    db = Path(db_path) if db_path else DEFAULT_DB_PATH
    with sqlite3.connect(str(db)) as conn:
        ctx = load_master_procurement_context(conn, as_of_date=as_of_date)

    current_price = float(ctx["current_price"])
    current_price_cbot = float(ctx["current_price_cbot"])
    mkt = ctx["market_avg_price_30d"]

    # TFT도 price_close 기준 학습 가정: 동일 변환
    fc = TFTQuantileForecast(
        p10=cbot_to_usd_per_ton(forecast.p10),
        p50=cbot_to_usd_per_ton(forecast.p50),
        p90=cbot_to_usd_per_ton(forecast.p90),
    )

    inv = float(req.current_inventory_ton)
    open_po = float(req.open_po_ton)
    effective_inventory = inv + open_po
    safety = float(req.safety_stock_ton)
    Q = float(req.quantity_ton)

    due_date = date.today() + timedelta(days=DEFAULT_LEAD_DAYS)
    days_to_due = DEFAULT_LEAD_DAYS
    monthly = float(req.monthly_consumption_ton)
    warehouse_cap = monthly * WAREHOUSE_MONTHS
    inventory_at_due = calc_inventory_at_due(effective_inventory, monthly, days_to_due)

    rec, reason_code = _recommend_action(
        inv=effective_inventory,
        safety=safety,
        current_price=current_price,
        p50=fc.p50,
    )

    buy_reward = (current_price - fc.p50) * Q
    wait_reward = -buy_reward
    split_reward = buy_reward * 0.5

    buy_cost = current_price * Q
    wait_cost = fc.p50 * Q
    split_cost = current_price * (Q / 2.0) + fc.p50 * (Q / 2.0)

    moq_ok = Q >= MOQ_MIN_TON
    split_half_ok = (Q / 2.0) >= MOQ_MIN_TON
    cap_ok = (effective_inventory + Q) <= warehouse_cap

    wait_reasons: list[str] = []
    if effective_inventory < safety:
        wait_reasons.append(
            f"현재 실질재고 {effective_inventory:.0f}톤이 안전재고 {safety:.0f}톤에 미달합니다"
        )
    if inventory_at_due < safety:
        wait_reasons.append(
            f"납기 시점(50일 후) 예상 재고 {inventory_at_due:.0f}톤이 안전재고 {safety:.0f}톤에 미달합니다"
        )

    wait_feasible = moq_ok and not wait_reasons
    if rec == "Wait" and not wait_feasible:
        rec = "Split"
        reason_code = "wait_blocked_inventory_due"

    options = {
        "Buy": _option_block(
            name="Buy",
            expected_cost_usd=buy_cost,
            risk_p10_total_usd=Q * fc.p10,
            risk_p90_total_usd=Q * fc.p90,
            reward_usd=buy_reward,
            feasible=moq_ok,
            notes="창고 상한 초과" if not cap_ok else "일괄 구매",
        ),
        "Split": _option_block(
            name="Split",
            expected_cost_usd=split_cost,
            risk_p10_total_usd=Q * fc.p10,
            risk_p90_total_usd=Q * fc.p90,
            reward_usd=split_reward,
            feasible=moq_ok and split_half_ok,
            notes=(
                f"분할 구매: 오늘 {Q/2:.0f}톤 계약 + {SPLIT_INTERVAL_DAYS}일 후 {Q/2:.0f}톤 계약 (납기일 동일: {due_date.isoformat()})"
                if split_half_ok
                else f"분할 1회당 {Q/2:.0f}톤 < MOQ {MOQ_MIN_TON:.0f}톤"
            ),
        ),
        "Wait": _option_block(
            name="Wait",
            expected_cost_usd=wait_cost,
            risk_p10_total_usd=Q * fc.p10,
            risk_p90_total_usd=Q * fc.p90,
            reward_usd=wait_reward,
            feasible=wait_feasible,
            notes="; ".join(wait_reasons) if wait_reasons else "대기 가능",
        ),
    }

    summary = _build_decision_summary(options, rec)
    if current_price > 0:
        summary["reward_vs_market_pct"] = round((fc.p50 - current_price) / current_price * 100.0, 4)

    direction = "보합"
    if fc.p50 > current_price * 1.03:
        direction = "상승 예상"
    elif fc.p50 < current_price * 0.97:
        direction = "하락 예상"

    human = {
        "current_price_summary": (
            f"현재 선물가 {current_price:,.0f}달러/톤 (CBOT {current_price_cbot:.2f} × {CBOT_TO_USD_PER_TON:.2f}), "
            f"시장평균(30일) {float(mkt or 0.0):,.0f}달러/톤"
        ),
        "forecast_summary": (
            f"50일 후 예상가 — 하단(P10): {fc.p10:,.0f}달러/톤 / "
            f"중간(P50): {fc.p50:,.0f}달러/톤 / 상단(P90): {fc.p90:,.0f}달러/톤"
        ),
        "price_direction": direction,
        "recommendation": rec,
        "reason": _reason_text(
            reason_code,
            p50=fc.p50,
            current_price=current_price,
            inv=effective_inventory,
            safety=safety,
        ),
        "risk_note": f"단, 낙관 시나리오(P10)에서는 {fc.p10:,.0f}달러/톤까지 내릴 수 있습니다.",
        "inventory_status": (
            f"현재 재고 {inv:.0f}톤 + 미납잔량 {open_po:.0f}톤 = 실질재고 {effective_inventory:.0f}톤 / "
            f"납기 시점(50일 후) 예상 재고 {inventory_at_due:.0f}톤 / 안전재고 {safety:.0f}톤"
        ),
        "constraint_warnings": [
            f"입고 후 재고 {effective_inventory + Q:.0f}톤이 창고 상한 {warehouse_cap:.0f}톤 초과"
        ] if not cap_ok else [],
        "split_plan": (
            f"분할 구매: 오늘 {Q/2:.0f}톤 계약 + {SPLIT_INTERVAL_DAYS}일 후 {Q/2:.0f}톤 계약 "
            f"(납기일 동일: {due_date.isoformat()})"
        ),
    }
    human["summary"] = f"현재 선물가 {current_price:,.0f}달러/톤, 50일 후 예상가(P50) {fc.p50:,.0f}달러/톤"

    out = {
        "decision_date": date.today().isoformat(),
        "created_at": _iso_now(),
        "as_of_date": ctx["as_of_date"].isoformat(),
        "due_date": due_date.isoformat(),
        "days_to_due_default": days_to_due,
        "current_price": round(current_price, 6),
        "market_avg_price_30d": None if mkt is None else round(float(mkt), 6),
        "forecast": {"p10": fc.p10, "p50": fc.p50, "p90": fc.p90},
        "forecast_cbot": {"p10": forecast.p10, "p50": forecast.p50, "p90": forecast.p90},
        "current_inventory_ton": inv,
        "open_po_ton": open_po,
        "effective_inventory_ton": effective_inventory,
        "safety_stock_ton": safety,
        "quantity_ton": Q,
        "inventory_at_due_ton": round(inventory_at_due, 4),
        "system_defaults": {
            "monthly_consumption_ton": monthly,
            "warehouse_cap_ton": warehouse_cap,
            "moq_min_ton": MOQ_MIN_TON,
            "lead_days_default": DEFAULT_LEAD_DAYS,
            "cbot_to_usd_per_ton": CBOT_TO_USD_PER_TON,
            "split_interval_days": SPLIT_INTERVAL_DAYS,
        },
        "recommended_action": rec,
        "recommendation_rule": reason_code,
        "options": options,
        "decision_summary": summary,
        "human_readable": human,
    }

    if persist_recommendation:
        out["decision_log_id"] = save_recommendation(out, db_path=db)
    else:
        out["decision_log_id"] = None
    return out


def evaluate_procurement_scenario_json(
    req: ScenarioRequest,
    forecast: TFTQuantileForecast,
    db_path: str | Path | None = None,
    *,
    indent: int = 2,
    persist_recommendation: bool = False,
) -> str:
    return json.dumps(
        evaluate_procurement_scenario(
            req,
            forecast,
            db_path=db_path,
            persist_recommendation=persist_recommendation,
        ),
        ensure_ascii=False,
        indent=indent,
    )


def _cli() -> None:
    req = ScenarioRequest(
        current_inventory_ton=3000,
        open_po_ton=1000,
        safety_stock_ton=3500,
        quantity_ton=3000,
        monthly_consumption_ton=1200,
    )
    fc = TFTQuantileForecast(p10=68.0, p50=72.0, p90=78.0)
    print(evaluate_procurement_scenario_json(req, fc))


if __name__ == "__main__":
    _cli()
