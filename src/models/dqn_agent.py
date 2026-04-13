"""
역할: 구매 타이밍/수량을 이산 행동 공간에서 학습하는 DQN 에이전트 (검증용).
레이어: 강화학습 (4) — 환경 정의는 별도 모듈 또는 본 파일 확장 시 연결.
"""

# 핵심 import (구현 시 사용 예정)
# import numpy as np
# import torch
# import torch.nn as nn


def build_env_from_master(config: dict):
    """master_daily·예측값을 기반으로 Gymnasium/커스텀 환경 인스턴스를 반환한다."""
    pass


def create_q_network(state_dim: int, action_dim: int):
    """Q 네트워크를 생성한다."""
    pass


def train_dqn(env, q_net, hyperparams: dict):
    """DQN 루프를 돌려 정책(가중치)을 학습한다."""
    pass


def select_action(q_net, state, epsilon: float) -> int:
    """ε-greedy 행동 선택."""
    pass


def evaluate_dqn(q_net, env, episodes: int) -> dict:
    """검증 에피소드 메트릭을 반환한다."""
    pass
