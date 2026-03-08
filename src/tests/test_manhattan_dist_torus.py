"""
Tests for manhattan_dist_torus function.
"""

import numpy as np
import pytest

from swarm.swarm_life_sb3 import manhattan_dist_torus


@pytest.mark.parametrize(
    "pos1,pos2,L,expected",
    [
        # Same position
        (
            np.array([[5, 5]], dtype=np.int16),
            np.array([[5, 5]], dtype=np.int16),
            10,
            0,
        ),
        # Adjacent horizontal
        (
            np.array([[5, 5]], dtype=np.int16),
            np.array([[6, 5]], dtype=np.int16),
            10,
            1,
        ),
        # Adjacent vertical
        (
            np.array([[5, 5]], dtype=np.int16),
            np.array([[5, 6]], dtype=np.int16),
            10,
            1,
        ),
        # Diagonal
        (
            np.array([[5, 5]], dtype=np.int16),
            np.array([[6, 6]], dtype=np.int16),
            10,
            2,
        ),
        # Wrapping horizontal
        (
            np.array([[0, 5]], dtype=np.int16),
            np.array([[9, 5]], dtype=np.int16),
            10,
            1,
        ),
        # Wrapping vertical
        (
            np.array([[5, 0]], dtype=np.int16),
            np.array([[5, 9]], dtype=np.int16),
            10,
            1,
        ),
        # Wrapping corner
        (
            np.array([[0, 0]], dtype=np.int16),
            np.array([[9, 9]], dtype=np.int16),
            10,
            2,
        ),
        # No wrap when direct shorter
        (
            np.array([[2, 5]], dtype=np.int16),
            np.array([[5, 5]], dtype=np.int16),
            10,
            3,
        ),
        # Edge wrapping midpoint
        (
            np.array([[0, 5]], dtype=np.int16),
            np.array([[5, 5]], dtype=np.int16),
            10,
            5,
        ),
        # Small torus wrapping
        (
            np.array([[0, 0]], dtype=np.int16),
            np.array([[4, 4]], dtype=np.int16),
            5,
            2,
        ),
        # Large torus direct
        (
            np.array([[0, 0]], dtype=np.int16),
            np.array([[50, 50]], dtype=np.int16),
            100,
            100,
        ),
    ],
)
def test_manhattan_dist_single(
    pos1: np.ndarray, pos2: np.ndarray, L: int, expected: int
) -> None:
    """Test Manhattan distance for single position pairs."""
    dist = manhattan_dist_torus(pos1, pos2, L)
    assert dist.shape == (1, 1)
    assert dist[0, 0] == expected


def test_manhattan_dist_multiple() -> None:
    """Test pairwise distances for multiple positions."""
    pos1 = np.array([[0, 0], [5, 5], [9, 9]], dtype=np.int16)
    pos2 = np.array([[1, 1], [6, 6], [0, 0]], dtype=np.int16)
    L = 10
    dist = manhattan_dist_torus(pos1, pos2, L)
    assert dist.shape == (3, 3)

    # Distance from (0,0) to (1,1) = 2
    assert dist[0, 0] == 2
    # Distance from (0,0) to (6,6) wraps: min(6,4) + min(6,4) = 8
    assert dist[0, 1] == 8
    # Distance from (0,0) to (0,0) = 0
    assert dist[0, 2] == 0

    # Distance from (5,5) to (1,1) = 8
    assert dist[1, 0] == 8
    # Distance from (5,5) to (6,6) = 2
    assert dist[1, 1] == 2
    # Distance from (5,5) to (0,0) = 10
    assert dist[1, 2] == 10


def test_manhattan_dist_symmetric() -> None:
    """Test that distance matrix is symmetric (when pos1 == pos2)."""
    pos = np.array([[0, 0], [5, 5], [9, 9]], dtype=np.int16)
    L = 10
    dist = manhattan_dist_torus(pos, pos, L)
    assert dist.shape == (3, 3)
    # Should be symmetric
    np.testing.assert_array_equal(dist, dist.T)
    # Diagonal should be zeros
    np.testing.assert_array_equal(np.diag(dist), [0, 0, 0])


@pytest.mark.parametrize(
    "L,pos1,pos2,expected",
    [
        (
            5,
            np.array([[0, 0]], dtype=np.int16),
            np.array([[4, 4]], dtype=np.int16),
            2,
        ),
        (
            100,
            np.array([[0, 0]], dtype=np.int16),
            np.array([[50, 50]], dtype=np.int16),
            100,
        ),
        (
            20,
            np.array([[0, 0]], dtype=np.int16),
            np.array([[10, 10]], dtype=np.int16),
            20,
        ),
    ],
)
def test_manhattan_dist_different_torus_sizes(
    L: int, pos1: np.ndarray, pos2: np.ndarray, expected: int
) -> None:
    """Test with different torus sizes."""
    dist = manhattan_dist_torus(pos1, pos2, L)
    assert dist[0, 0] == expected
