"""Shared fixtures and helpers for swarm_life tests."""

import numpy as np

from swarm.swarm_life_sb3 import SwarmLifePatternEnv


def set_positions(env: SwarmLifePatternEnv, positions: np.ndarray) -> None:
    """Set env to exact positions and sync occupancy. positions shape (N, 2) [x, y]."""
    env.pos[:] = positions
    env.occ[:] = False
    for x, y in positions:
        env.occ[y, x] = True
