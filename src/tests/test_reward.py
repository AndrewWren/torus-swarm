"""
Tests for SwarmLifePatternEnv._get_reward_and_num_good.

Reward formula:
  individual_reward[i] = min(num_manhattan_neighbors[i], 4) / 4
  in_zone[i,j] = Chebyshev(i,j) <= r  (square patch, includes self)
  reward[i] = (in_zone[i] @ individual_reward) / N
  return reward.mean(), num_neighbors.mean()
"""

import numpy as np
import pytest

from swarm.swarm_life_sb3 import SwarmLifePatternConfig, SwarmLifePatternEnv

from conftest import set_positions as _set_positions


@pytest.fixture
def env4() -> SwarmLifePatternEnv:
    """L=16, N=4, r=2."""
    cfg = SwarmLifePatternConfig(
        L=16, N=4, r=2, T=100, seed=0
    )
    env = SwarmLifePatternEnv(cfg)
    env.reset()
    return env


def test_all_isolated_zero_reward(env4: SwarmLifePatternEnv) -> None:
    """Agents with no neighbors yield zero reward and num_good=0."""
    # Spacing > 2*r=4 so no agent is in any other's zone.
    positions = np.array([[0, 0], [8, 0], [0, 8], [8, 8]], dtype=np.int16)
    _set_positions(env4, positions)
    matched, num_good = env4._get_reward_and_num_good()
    assert matched == pytest.approx(0.0)
    assert num_good == pytest.approx(0.0)


def test_2x2_cluster_full_reward(env4: SwarmLifePatternEnv) -> None:
    """2×2 cluster: each agent has exactly 2 Manhattan neighbors.

    All four agents are within Chebyshev r=2 of each other.
    individual_reward = min(2,4)/4 = 0.5 for all.
    reward[i] = (0.5+0.5+0.5+0.5) / 4 = 0.5  →  mean = 0.5, num_good = 2.0.
    """
    # (5,5)-(6,5)
    # (5,6)-(6,6)  — each agent has 2 Manhattan-distance-1 neighbors
    positions = np.array([[5, 5], [6, 5], [5, 6], [6, 6]], dtype=np.int16)
    _set_positions(env4, positions)
    matched, num_good = env4._get_reward_and_num_good()
    assert matched == pytest.approx(0.5)
    assert num_good == pytest.approx(2.0)


def test_num_good_counts_mean_manhattan_neighbors(
    env4: SwarmLifePatternEnv,
) -> None:
    """num_good is the mean number of Manhattan-distance-1 neighbors."""
    # Chain (5,5)-(6,5)-(7,5): neighbors = [1,2,1], isolated agent = 0
    # mean = (1+2+1+0)/4 = 1.0
    positions = np.array([[5, 5], [6, 5], [7, 5], [0, 0]], dtype=np.int16)
    _set_positions(env4, positions)
    _, num_good = env4._get_reward_and_num_good()
    assert num_good == pytest.approx(1.0)

    # All isolated → num_good = 0
    positions = np.array([[0, 0], [8, 0], [0, 8], [8, 8]], dtype=np.int16)
    _set_positions(env4, positions)
    _, num_good = env4._get_reward_and_num_good()
    assert num_good == pytest.approx(0.0)


def test_chain_of_three_reward(env4: SwarmLifePatternEnv) -> None:
    """Chain of 3 agents, 4th isolated.

    Positions: (5,5)-(6,5)-(7,5), agent 3 at (0,0).

    individual_reward: edges=0.25, middle=0.5, isolated=0.0.

    Zone of each chain agent (r=2): all three chain members are included
    because the endpoint-to-endpoint Chebyshev distance = max(2,0) = 2 = r.
    Zone of agent 3: only itself.

    reward[0] = reward[1] = reward[2] = (0.25+0.5+0.25)/4 = 1.0/4 = 0.25
    reward[3] = 0.0/4 = 0.0
    mean = (0.25+0.25+0.25+0.0)/4 = 0.75/4
    num_good = (1+2+1+0)/4 = 1.0
    """
    positions = np.array([[5, 5], [6, 5], [7, 5], [0, 0]], dtype=np.int16)
    _set_positions(env4, positions)
    matched, num_good = env4._get_reward_and_num_good()
    assert num_good == pytest.approx(1.0)
    assert matched == pytest.approx(0.75 / 4)


def test_zone_chebyshev_boundary_included_excluded(
    env4: SwarmLifePatternEnv,
) -> None:
    """Agent at Chebyshev distance exactly r is in zone; at r+1 is not.

    Positions: 0=(3,5), 1=(5,5), 2=(5,6), 3=(6,5).

    Manhattan neighbors:
      1 and 2: dist=1  →  neighbor
      1 and 3: dist=1  →  neighbor
      2 and 3: dist=2  →  not a neighbor
      0: no Manhattan neighbors

    individual_reward = [0.0, 0.5, 0.25, 0.25], num_good = 1.0

    Chebyshev distances from agent 0=(3,5):
      to 1=(5,5): max(2,0)=2 = r  → IN zone
      to 2=(5,6): max(2,1)=2 = r  → IN zone
      to 3=(6,5): max(3,0)=3 > r  → NOT in zone

    Zone memberships:
      0's zone: {0,1,2}   sum = 0.0+0.5+0.25 = 0.75   reward[0] = 0.75/4
      1's zone: {0,1,2,3} sum = 0.0+0.5+0.25+0.25=1.0  reward[1] = 1.0/4
      2's zone: {0,1,2,3} sum = 1.0                     reward[2] = 1.0/4
      3's zone: {1,2,3}   sum = 0.5+0.25+0.25=1.0       reward[3] = 1.0/4

    mean = (0.75+1.0+1.0+1.0) / (4*4) = 3.75/16
    """
    positions = np.array([[3, 5], [5, 5], [5, 6], [6, 5]], dtype=np.int16)
    _set_positions(env4, positions)
    matched, num_good = env4._get_reward_and_num_good()
    assert num_good == pytest.approx(1.0)
    assert matched == pytest.approx(3.75 / 16)


def test_reward_in_step_matches_get_reward(env4: SwarmLifePatternEnv) -> None:
    """info['match'] and info['num_good'] from step() match _get_reward_and_num_good."""
    env4.reset(seed=0)
    positions = np.array([[5, 5], [6, 5], [5, 6], [6, 6]], dtype=np.int16)
    _set_positions(env4, positions)
    env4._prev_pos_set = frozenset(tuple(p) for p in env4.pos)

    action = np.zeros(
        env4.N, dtype=np.int64
    )  # all stay → no move/collision costs
    _, reward, _, _, info = env4.step(action)

    assert info["match"] == pytest.approx(0.5)
    assert info["num_good"] == pytest.approx(2.0)
    # With no moves or collisions, scalar reward equals matched
    assert reward == pytest.approx(0.5)
