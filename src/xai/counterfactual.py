"""
역할: '가격이 X였다면 예측/권고는?' 형태의 반사실(counterfactual) 시나리오 생성.
레이어: XAI (5) — SHAP과 함께 Joule 자연어 설명의 근거로 사용.
"""

# 핵심 import (구현 시 사용 예정)
# import numpy as np
# import pandas as pd


def define_counterfactual_mask(base_row: dict, perturbations: dict) -> dict:
    """기준 행에 대해 변경할 피처와 목표값을 정의한다."""
    pass


def generate_counterfactuals(model, base_row, candidate_grid):
    """모델 입력 공간에서 반사실 샘플을 생성하고 예측을 비교한다."""
    pass


def rank_counterfactuals(results: list[dict], objective: str) -> list[dict]:
    """목표(예: 비용 최소, 리스크 상한)에 따라 시나리오를 정렬한다."""
    pass
