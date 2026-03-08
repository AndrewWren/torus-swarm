"""Smoke test for load_and_eval.py: train a tiny model, then load and run it."""

from __future__ import annotations

import sys
from unittest.mock import patch

import pytest

from swarm.run_artifacts import write_config
from localized_actions import train_from_config

_SMOKE_CONFIG = {
    "env": {
        "L": 8,
        "N": 4,
        "r": 1,
        "T": 8,
        "include_abs_pos": False,
        "move_penalty": 1e-3,
        "collision_penalty": 0.01,
        "seed": 0,
    },
    "ppo": {
        "n_steps": 8,
        "batch_size": 8,
        "n_epochs": 1,
        "learning_rate": 1e-4,
        "gamma": 0.99,
        "ent_coef": 0.01,
        "clip_range": 0.2,
        "policy_kwargs": {"per_agent_dim": 16, "critic_hidden": (32, 32)},
    },
    "training": {
        "total_timesteps": 8,
        "eval_freq": 8,
        "n_envs": 1,
        "script": "smoke",
    },
    "vec_normalize": None,
}


@pytest.fixture(scope="module")
def trained_run_dir(tmp_path_factory):
    run_dir = tmp_path_factory.mktemp("load_eval_run")
    write_config(run_dir, _SMOKE_CONFIG)
    train_from_config(
        _SMOKE_CONFIG, run_dir, total_timesteps=8, verbose=0, record_video_on_end=False
    )
    return run_dir


def test_load_and_eval_runs(trained_run_dir) -> None:
    """load_and_eval.main() loads a trained model and completes a short episode without error."""
    import load_and_eval

    argv = [
        "load_and_eval",
        "--run-dir", str(trained_run_dir),
        "--n-steps", "4",
        "--no-deterministic",
    ]
    with patch.object(sys, "argv", argv):
        load_and_eval.main()
