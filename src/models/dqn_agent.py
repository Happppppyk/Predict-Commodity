from __future__ import annotations


def build_env_from_master(config: dict):
    pass


def create_q_network(state_dim: int, action_dim: int):
    pass


def train_dqn(env, q_net, hyperparams: dict):
    pass


def select_action(q_net, state, epsilon: float) -> int:
    pass


def evaluate_dqn(q_net, env, episodes: int) -> dict:
    pass
