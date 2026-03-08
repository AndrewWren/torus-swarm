"""
Tests for SwarmLifePatternEnv collision and swap rejection.

Actions: 0 stay, 1 up, 2 down, 3 left, 4 right.
"""

import numpy as np
import pytest

from swarm.swarm_life_sb3 import SwarmLifePatternConfig, SwarmLifePatternEnv

from conftest import set_positions as _set_positions


@pytest.fixture
def small_env() -> SwarmLifePatternEnv:
    """Env with L=8, N=4 for deterministic collision/swap tests."""
    cfg = SwarmLifePatternConfig(
        L=8,
        N=4,
        r=2,
        T=256,
        include_abs_pos=True,
        move_penalty=1e-3,
        collision_penalty=0.1,
        seed=0,
    )
    return SwarmLifePatternEnv(cfg)


def test_same_cell_collision_both_bounce_back(small_env: SwarmLifePatternEnv) -> None:
    """Two agents moving to the same empty cell both bounce back with penalty."""
    # Agent 0 at (1,1) moves right -> (2,1); agent 1 at (3,1) moves left -> (2,1). Collision.
    positions = np.array([[1, 1], [3, 1], [5, 5], [6, 6]], dtype=np.int16)
    _set_positions(small_env, positions)
    action = np.array([4, 3, 0, 0], dtype=np.int64)  # 0 right, 1 left -> both to (2,1)
    moved, bounced, penalized = small_env._apply_moves(action)
    assert not moved[0] and not moved[1]
    assert bounced[0] and bounced[1]
    assert penalized[0] and penalized[1]  # same-cell collision is penalized
    assert not moved[2] and not moved[3]
    assert not bounced[2] and not bounced[3]
    np.testing.assert_array_equal(small_env.pos[0], [1, 1])
    np.testing.assert_array_equal(small_env.pos[1], [3, 1])


def test_pairwise_swap_both_bounce_back(small_env: SwarmLifePatternEnv) -> None:
    """Two agents swapping places both bounce: destinations are pre-action occupied."""
    # Agent 0 at (1,1) moves right -> (2,1) which is occupied by agent 1.
    # Agent 1 at (2,1) moves left -> (1,1) which is occupied by agent 0.
    positions = np.array([[1, 1], [2, 1], [5, 5], [6, 6]], dtype=np.int16)
    _set_positions(small_env, positions)
    action = np.array([4, 3, 0, 0], dtype=np.int64)  # 0 right, 1 left
    moved, bounced, penalized = small_env._apply_moves(action)
    assert not moved[0] and not moved[1]
    assert bounced[0] and bounced[1]
    assert not penalized[0] and not penalized[1]  # occupancy-blocked, not penalized
    np.testing.assert_array_equal(small_env.pos[0], [1, 1])
    np.testing.assert_array_equal(small_env.pos[1], [2, 1])


def test_four_cycle_all_bounce_back(small_env: SwarmLifePatternEnv) -> None:
    """Four agents in a rotation cycle all bounce: each destination is pre-action occupied."""
    positions = np.array(
        [[0, 0], [1, 0], [1, 1], [0, 1]],
        dtype=np.int16,
    )
    _set_positions(small_env, positions)
    action = np.array([4, 2, 3, 1], dtype=np.int64)  # right, down, left, up
    moved, bounced, penalized = small_env._apply_moves(action)
    assert not moved.any()
    assert bounced.all()
    assert not penalized.any()  # occupancy-blocked, not penalized
    np.testing.assert_array_equal(small_env.pos[0], [0, 0])
    np.testing.assert_array_equal(small_env.pos[1], [1, 0])
    np.testing.assert_array_equal(small_env.pos[2], [1, 1])
    np.testing.assert_array_equal(small_env.pos[3], [0, 1])


def test_successful_move_not_bounced(small_env: SwarmLifePatternEnv) -> None:
    """Single agent moving into an empty cell moves and is not bounced."""
    positions = np.array([[0, 0], [2, 0], [5, 5], [6, 6]], dtype=np.int16)
    _set_positions(small_env, positions)
    action = np.array([4, 0, 0, 0], dtype=np.int64)  # only agent 0 moves right
    moved, bounced, penalized = small_env._apply_moves(action)
    assert moved[0] and not bounced[0]
    assert not moved[1] and not bounced[1]
    np.testing.assert_array_equal(small_env.pos[0], [1, 0])


def test_step_returns_num_bounced_and_collision_cost(small_env: SwarmLifePatternEnv) -> None:
    """step() info contains num_bounced; swap is occupancy-blocked so no collision cost."""
    small_env.reset(seed=0)
    positions = np.array([[1, 1], [2, 1], [5, 5], [6, 6]], dtype=np.int16)
    _set_positions(small_env, positions)
    small_env._prev_pos_set = frozenset(tuple(p) for p in small_env.pos)
    action = np.array([4, 3, 0, 0], dtype=np.int64)  # swap -> both bounce (occupancy-blocked)
    obs, reward, term, trunc, info = small_env.step(action)
    assert "num_bounced" in info
    assert info["num_bounced"] == 2
    assert "collision_cost" in info
    assert info["collision_cost"] == pytest.approx(0.0)  # occupancy-blocked, not penalized
    assert info["num_penalized"] == 0
    matched = info["match"]
    move_cost = small_env.cfg.move_penalty * (action != 0).mean()
    expected_reward = matched - move_cost
    assert reward == pytest.approx(expected_reward)


def test_collision_penalty_in_config() -> None:
    """SwarmLifePatternConfig has collision_penalty with default."""
    cfg = SwarmLifePatternConfig()
    assert hasattr(cfg, "collision_penalty")
    assert cfg.collision_penalty == 0.01
    cfg_custom = SwarmLifePatternConfig(collision_penalty=0.5)
    assert cfg_custom.collision_penalty == 0.5


def test_occupancy_blocked_bounced(small_env: SwarmLifePatternEnv) -> None:
    """Agent moving into a cell occupied by a non-mover is blocked and bounces."""
    # Agent 0 at (0,0), agent 1 at (1,0). 0 moves right into (1,0); 1 stays. 0 blocked.
    positions = np.array([[0, 0], [1, 0], [5, 5], [6, 6]], dtype=np.int16)
    _set_positions(small_env, positions)
    action = np.array([4, 0, 0, 0], dtype=np.int64)  # 0 right, 1 stay
    moved, bounced, penalized = small_env._apply_moves(action)
    assert not moved[0] and bounced[0]
    assert not penalized[0]  # occupancy-blocked, not penalized
    assert not moved[1]
    np.testing.assert_array_equal(small_env.pos[0], [0, 0])
    np.testing.assert_array_equal(small_env.pos[1], [1, 0])


def test_stay_never_bounced(small_env: SwarmLifePatternEnv) -> None:
    """Agents that stay (action 0) are never in moved or bounced."""
    positions = np.array([[0, 0], [1, 0], [2, 0], [3, 0]], dtype=np.int16)
    _set_positions(small_env, positions)
    action = np.array([0, 0, 0, 0], dtype=np.int64)
    moved, bounced, penalized = small_env._apply_moves(action)
    assert not moved.any() and not bounced.any()


def test_action_masks_empty_neighbors(small_env: SwarmLifePatternEnv) -> None:
    """action_masks returns True for directions pointing to empty cells."""
    # Place agent 0 alone with no neighbours; all 4 directions should be valid.
    positions = np.array([[4, 4], [0, 0], [7, 7], [3, 0]], dtype=np.int16)
    _set_positions(small_env, positions)
    masks = small_env.action_masks().reshape(small_env.N, 5)
    # Agent 0: neighbours at (3,4),(5,4),(4,3),(4,5) — all empty
    assert masks[0].all()


def test_action_masks_blocked_directions(small_env: SwarmLifePatternEnv) -> None:
    """action_masks returns False for directions pointing to occupied cells."""
    # Agent 0 at (1,1); agent 1 at (2,1) (to the right of agent 0)
    positions = np.array([[1, 1], [2, 1], [5, 5], [6, 6]], dtype=np.int16)
    _set_positions(small_env, positions)
    masks = small_env.action_masks().reshape(small_env.N, 5)
    # For agent 0: action 4 (right) would go to (2,1) which is occupied
    assert not masks[0, 4]
    # Action 0 (stay) always valid
    assert masks[0, 0]
    # Other directions are empty
    assert masks[0, 1] and masks[0, 2] and masks[0, 3]


def _assert_pos_occ_invariant(env: SwarmLifePatternEnv) -> None:
    """Assert N agents, N distinct positions, occ matches pos."""
    N, L = env.N, env.L
    assert env.occ.sum() == N, "occ should have exactly N cells True"
    positions = [tuple(env.pos[i]) for i in range(N)]
    assert len(set(positions)) == N, "all agent positions must be distinct"
    for i in range(N):
        x, y = env.pos[i, 0], env.pos[i, 1]
        assert 0 <= x < L and 0 <= y < L, f"agent {i} pos ({x},{y}) out of bounds"
        assert env.occ[y, x], f"occ at agent {i} pos ({x},{y}) must be True"


def test_pos_occ_invariant_over_many_steps() -> None:
    """Run many steps with random actions; pos/occ must stay consistent."""
    cfg = SwarmLifePatternConfig(L=32, N=64, r=2, T=500, seed=42)
    env = SwarmLifePatternEnv(cfg)
    rng = np.random.default_rng(123)
    obs, _ = env.reset(seed=0)
    _assert_pos_occ_invariant(env)
    for _ in range(200):
        action = rng.integers(0, 5, size=env.N, dtype=np.int64)
        obs, reward, term, trunc, info = env.step(action)
        _assert_pos_occ_invariant(env)
        if term or trunc:
            obs, _ = env.reset()
            _assert_pos_occ_invariant(env)
