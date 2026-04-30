"""
대두유 AI 구매 의사결정 데모 스크립트
현업 입력 → 예측 → 시나리오 분석 → JSON 출력

실행:
PYTHONPATH=src python3 scripts/run_demo.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# 프로젝트 루트 (soybean-oil-poc)
_ROOT = Path(__file__).resolve().parents[1]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from rl.scenario_engine import (  # noqa: E402
    TFTQuantileForecast,
    ScenarioRequest,
    evaluate_procurement_scenario_json,
    save_recommendation,
)

def _print_result(title: str, result_dict: dict) -> None:
    print("=" * 60)
    print(title)
    print("=" * 60)
    hr = result_dict.get("human_readable", {})
    print(f"\n[가격 현황]")
    print(f"  {hr.get('current_price_summary')}")
    print(f"\n[가격 전망]")
    print(f"  {hr.get('forecast_summary')}")
    print(f"  단위: {hr.get('forecast_unit')}")
    print(f"\n[재고 현황]")
    print(f"  {hr.get('inventory_status')}")
    print(f"\n[권고 액션]")
    print(f"  → {hr.get('recommendation')}")
    print(f"  근거: {hr.get('reason')}")
    print(f"  리스크: {hr.get('risk_note')}")

    warnings = hr.get("constraint_warnings", [])
    if warnings:
        print(f"\n[제약 위반]")
        for w in warnings:
            print(f"  ⚠️ {w}")

    print(f"\n[전체 JSON]")
    print(json.dumps(result_dict, ensure_ascii=False, indent=2))


db_path = _ROOT / "data" / "db" / "soybean.db"

# ──────────────────────────────────────
# 섹션 1 — 수동 입력 모드 (CBOT 스케일 입력)
# ──────────────────────────────────────
manual_request = ScenarioRequest(
    current_inventory_ton=4000.0,
    safety_stock_ton=3500.0,
    quantity_ton=5000.0,
    monthly_consumption_ton=1200.0,
    open_po_ton=0.0,
    forecast_unit="cbot",
)
manual_forecast = TFTQuantileForecast(
    p10=68.0,
    p50=72.0,
    p90=78.0,
)
manual_result = json.loads(
    evaluate_procurement_scenario_json(manual_request, manual_forecast, db_path=db_path)
)
_print_result("대두유 AI 구매 의사결정 결과 (수동/CBOT 입력)", manual_result)
saved_id = save_recommendation(manual_result, db_path=db_path)
print(f"\n✅ decision_log 저장 완료 (id={saved_id})")
print("=" * 60)

# ──────────────────────────────────────
# 섹션 2 — TFT 자동 연동 모드 (준비)
# USD/톤 값을 그대로 입력할 때 forecast_unit='usd_per_ton'
# ──────────────────────────────────────
"""
from models.tft_model import predict_latest

forecast_usd = predict_latest()  # {'p10': ..., 'p50': ..., 'p90': ...} (USD/톤)
auto_request = ScenarioRequest(
    current_inventory_ton=4000.0,
    safety_stock_ton=3500.0,
    quantity_ton=5000.0,
    monthly_consumption_ton=1200.0,
    open_po_ton=0.0,
    forecast_unit="usd_per_ton",
)
auto_forecast = TFTQuantileForecast(
    p10=forecast_usd["p10"],
    p50=forecast_usd["p50"],
    p90=forecast_usd["p90"],
)
auto_result = json.loads(
    evaluate_procurement_scenario_json(auto_request, auto_forecast, db_path=db_path)
)
_print_result("대두유 AI 구매 의사결정 결과 (자동/TFT USD 입력)", auto_result)
"""
