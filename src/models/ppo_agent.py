from __future__ import annotations


def build_policy_network(state_dim: int, action_dim: int):
    pass


def train_ppo(env, policy, hyperparams: dict):
    pass


def act(policy, state, deterministic: bool = False):
    pass


def save_policy(policy, path: str) -> None:
    pass


def load_policy(path: str):
    pass
