"""
Tests for FactorizedSwarmPolicy.

Key properties:
- Actor is decentralized: changing agent i's observation changes only agent i's logits.
- Critic is centralized: changing any agent's observation changes the value estimate.
- Output shapes match SB3 expectations.
"""

import numpy as np
import pytest
import torch as th
from gymnasium import spaces

from localized_actions import FactorizedSwarmPolicy


N, C, P = 4, 1, 5  # r=2 → P=5; no abs_pos → C=1
OBS_DIM = N * C * P * P


@pytest.fixture
def policy() -> FactorizedSwarmPolicy:
    obs_space = spaces.Box(low=0.0, high=1.0, shape=(OBS_DIM,), dtype=np.float32)
    act_space = spaces.MultiDiscrete(np.full(N, 5, dtype=np.int64))
    return FactorizedSwarmPolicy(
        obs_space,
        act_space,
        lr_schedule=lambda _: 1e-4,
        N=N,
        C=C,
        P=P,
        per_agent_dim=32,
        critic_hidden=(64, 64),
    )


def _random_obs(batch: int, seed: int = 0) -> th.Tensor:
    rng = np.random.default_rng(seed)
    return th.tensor(
        rng.random((batch, OBS_DIM), dtype=np.float32), dtype=th.float32
    )


def test_forward_output_shapes(policy: FactorizedSwarmPolicy) -> None:
    """forward() returns actions (B,N), values (B,1), log_prob (B,)."""
    B = 3
    obs = _random_obs(B)
    with th.no_grad():
        actions, values, log_prob = policy.forward(obs)
    assert actions.shape == (B, N)
    assert values.shape == (B, 1)
    assert log_prob.shape == (B,)
    assert actions.dtype == th.int64


def test_evaluate_actions_output_shapes(policy: FactorizedSwarmPolicy) -> None:
    """evaluate_actions() returns values (B,1), log_prob (B,), entropy (B,)."""
    B = 3
    obs = _random_obs(B)
    with th.no_grad():
        actions, _, _ = policy.forward(obs)
        values, log_prob, entropy = policy.evaluate_actions(obs, actions)
    assert values.shape == (B, 1)
    assert log_prob.shape == (B,)
    assert entropy.shape == (B,)


def test_predict_values_shape(policy: FactorizedSwarmPolicy) -> None:
    """predict_values() returns (B, 1)."""
    B = 5
    obs = _random_obs(B)
    with th.no_grad():
        values = policy.predict_values(obs)
    assert values.shape == (B, 1)


def test_actor_is_decentralized(policy: FactorizedSwarmPolicy) -> None:
    """Changing agent i's observation patch changes only agent i's action logits."""
    policy.eval()
    B = 2
    obs1 = _random_obs(B, seed=1)
    obs2 = obs1.clone()

    # Replace agent 1's patch with different values in all batch items
    agent_start = 1 * C * P * P
    agent_end = 2 * C * P * P
    obs2[:, agent_start:agent_end] = th.zeros(B, C * P * P)

    with th.no_grad():
        emb1 = policy._encode_agents(obs1)
        emb2 = policy._encode_agents(obs2)
        # Actor logits (B, N, 5) — only agent 1's column should differ
        logits1 = policy.actor_head(emb1)  # (B, N, 5)
        logits2 = policy.actor_head(emb2)

    assert not th.allclose(logits1[:, 1, :], logits2[:, 1, :]), (
        "Agent 1's logits should differ after changing its observation"
    )
    for i in [0, 2, 3]:
        th.testing.assert_close(
            logits1[:, i, :],
            logits2[:, i, :],
            msg=f"Agent {i}'s logits should be unchanged",
        )


def test_masking_blocks_occupied_neighbors(policy: FactorizedSwarmPolicy) -> None:
    """Directions pointing to occupied cells get logit -inf; unblocked dirs are finite."""
    r = P // 2  # P=5, r=2
    B = 1
    obs = th.zeros(B, OBS_DIM, dtype=th.float32)

    # Each agent's center cell must be 1.0 (always occupied in a valid env state).
    for i in range(N):
        obs[0, i * C * P * P + r * P + r] = 1.0

    # Block agent 0's right neighbor (action 4) and up neighbor (action 1).
    obs[0, 0 * C * P * P + r * P + (r + 1)] = 1.0  # right
    obs[0, 0 * C * P * P + (r - 1) * P + r] = 1.0  # up

    policy.eval()
    with th.no_grad():
        emb = policy._encode_agents(obs)
        logits_flat = policy._actor_logits(emb, obs)

    # logits_flat is (B, N*5); agent 0 occupies the first 5 logits.
    logits0 = logits_flat[0, :5]
    assert logits0[4] == float("-inf"), "right (blocked) should be -inf"
    assert logits0[1] == float("-inf"), "up (blocked) should be -inf"
    assert logits0[0].isfinite(), "stay should be finite"
    assert logits0[2].isfinite(), "down (unblocked) should be finite"
    assert logits0[3].isfinite(), "left (unblocked) should be finite"

    # Agents 1-3 have no blocked neighbours; all their logits should be finite.
    for i in range(1, N):
        assert logits_flat[0, i * 5 : (i + 1) * 5].isfinite().all(), (
            f"Agent {i} has no blocked neighbours; all logits should be finite"
        )


def test_critic_uses_all_agents(policy: FactorizedSwarmPolicy) -> None:
    """Changing any single agent's observation changes the value estimate.

    The centralized critic mean-pools all agent embeddings, so any change
    propagates to the value.
    """
    policy.eval()
    B = 2
    obs_base = _random_obs(B, seed=2)

    with th.no_grad():
        base_value = policy.predict_values(obs_base)

    for i in range(N):
        obs_mod = obs_base.clone()
        start = i * C * P * P
        end = (i + 1) * C * P * P
        obs_mod[:, start:end] = th.ones(B, C * P * P)

        with th.no_grad():
            mod_value = policy.predict_values(obs_mod)

        assert not th.allclose(base_value, mod_value), (
            f"Value should change when agent {i}'s observation changes"
        )
