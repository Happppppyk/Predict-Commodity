"""
역할: 연속/혼합 행동 공간에서 구매 정책을 최적화하는 PPO 에이전트.
레이어: 강화학습 (4) — DQN 검증 이후 본 최적화 단계.
"""

# 핵심 import (구현 시 사용 예정)
# import numpy as np
# import torch
# from torch.distributions import Normal


def build_policy_network(state_dim: int, action_dim: int):
    """Actor-Critic 또는 단일 정책 네트워크를 생성한다."""
    pass


def train_ppo(env, policy, hyperparams: dict):
    """PPO 업데이트 루프를 실행한다."""
    pass


def act(policy, state, deterministic: bool = False):
    """정책으로부터 행동(및 로그 확률 등)을 샘플링한다."""
    pass


def save_policy(policy, path: str) -> None:
    """학습된 정책 가중치를 저장한다."""
    pass


def load_policy(path: str):
    """저장된 정책을 로드한다."""
    pass
