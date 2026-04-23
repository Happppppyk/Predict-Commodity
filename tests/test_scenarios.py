"""시나리오 엔진 통합 테스트 (단위변환/실질재고/분할로직)."""

from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rl.scenario_engine import (  # noqa: E402
    TFTQuantileForecast,
    ScenarioRequest,
    evaluate_procurement_scenario,
    save_recommendation,
)

DB_PATH = PROJECT_ROOT / "data" / "db" / "soybean.db"


def _assert_human_readable(result: dict) -> None:
    hr = result.get("human_readable")
    assert isinstance(hr, dict), "human_readable 누락"
    for k in (
        "current_price_summary",
        "forecast_summary",
        "price_direction",
        "recommendation",
        "reason",
        "risk_note",
        "inventory_status",
        "constraint_warnings",
    ):
        assert k in hr, f"human_readable.{k} 누락"


def _print_scenario(name: str, result: dict) -> None:
    print(f"\n{'=' * 60}\n{name}\n{'=' * 60}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def run_all() -> None:
    assert DB_PATH.is_file(), f"DB 없음: {DB_PATH}"
    saved_ids: list[int] = []

    # 시나리오 1 — 가격 상승 + open PO 있음
    r1 = evaluate_procurement_scenario(
        ScenarioRequest(
            current_inventory_ton=3000.0,
            open_po_ton=1000.0,
            safety_stock_ton=3500.0,
            quantity_ton=3000.0,
            monthly_consumption_ton=1200.0,
        ),
        TFTQuantileForecast(p10=68.0, p50=72.0, p90=78.0),
        db_path=DB_PATH,
    )
    _assert_human_readable(r1)
    _print_scenario("시나리오 1 — 가격 상승 + open PO", r1)
    assert r1["recommended_action"] == "Buy"
    assert r1["effective_inventory_ton"] == 4000.0
    assert isinstance(r1["options"]["Buy"]["reward_usd"], float)
    assert "달러/톤" in r1["human_readable"]["current_price_summary"]
    lid1 = save_recommendation(r1, db_path=DB_PATH)
    saved_ids.append(lid1)

    # 시나리오 2 — 안전재고 위기
    r2 = evaluate_procurement_scenario(
        ScenarioRequest(
            current_inventory_ton=3000.0,
            open_po_ton=0.0,
            safety_stock_ton=3500.0,
            quantity_ton=3000.0,
            monthly_consumption_ton=1200.0,
        ),
        TFTQuantileForecast(p10=60.0, p50=64.0, p90=68.0),
        db_path=DB_PATH,
    )
    _assert_human_readable(r2)
    _print_scenario("시나리오 2 — 안전재고 위기", r2)
    assert r2["recommended_action"] == "Buy"
    assert r2["effective_inventory_ton"] < r2["safety_stock_ton"]
    lid2 = save_recommendation(r2, db_path=DB_PATH)
    saved_ids.append(lid2)

    # 시나리오 3 — 가격 하락 + 납기 재고 부족
    r3 = evaluate_procurement_scenario(
        ScenarioRequest(
            current_inventory_ton=5000.0,
            open_po_ton=0.0,
            safety_stock_ton=3500.0,
            quantity_ton=2000.0,
            monthly_consumption_ton=1200.0,
        ),
        TFTQuantileForecast(p10=58.0, p50=63.0, p90=68.0),
        db_path=DB_PATH,
    )
    _assert_human_readable(r3)
    _print_scenario("시나리오 3 — 가격 하락 + 납기재고 부족", r3)
    assert r3["recommended_action"] == "Split"
    wait_viol = [
        v for v in r3["decision_summary"]["constraint_violations"] if v["action"] == "Wait"
    ]
    assert wait_viol and "납기 시점" in wait_viol[0]["reason"]
    lid3 = save_recommendation(r3, db_path=DB_PATH)
    saved_ids.append(lid3)

    with sqlite3.connect(str(DB_PATH)) as conn:
        rows = conn.execute(
            "SELECT id, recommended_action FROM decision_log WHERE id IN (?, ?, ?) ORDER BY id",
            (saved_ids[0], saved_ids[1], saved_ids[2]),
        ).fetchall()
    assert len(rows) == 3
    print(f"\n[OK] decision_log 저장 확인: ids={saved_ids}")


if __name__ == "__main__":
    run_all()
