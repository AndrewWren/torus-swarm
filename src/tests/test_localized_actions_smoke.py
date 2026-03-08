"""
Smoke test: train_from_config runs end-to-end and produces expected artifacts.
"""

from pathlib import Path

from swarm.run_artifacts import (
    FINAL_EVAL_FILENAME,
    MODEL_BASENAME,
    VECNORM_FILENAME,
)
from localized_actions import train_from_config

# Minimal config: tiny env, 1 PPO rollout, no VecNormalize.
N, r, L = 4, 1, 8

_SMOKE_CONFIG = {
    "env": {
        "L": L,
        "N": N,
        "r": r,
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
        "policy_kwargs": {
            "per_agent_dim": 16,
            "critic_hidden": (32, 32),
        },
    },
    "training": {
        "total_timesteps": 8,
        "eval_freq": 8,
        "n_envs": 1,
        "script": "smoke_test",
    },
    "vec_normalize": None,
}


def test_train_from_config_produces_artifacts(tmp_path: Path) -> None:
    """train_from_config completes and saves model.zip and final_eval.json."""
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    train_from_config(_SMOKE_CONFIG, run_dir, total_timesteps=8, verbose=0)
    assert (run_dir / (MODEL_BASENAME + ".zip")).exists()
    assert (run_dir / FINAL_EVAL_FILENAME).exists()


def test_train_from_config_with_vec_normalize(tmp_path: Path) -> None:
    """train_from_config saves vecnorm.pkl when vec_normalize is enabled."""
    config = {
        **_SMOKE_CONFIG,
        "vec_normalize": {
            "norm_obs": True,
            "norm_reward": True,
            "clip_obs": 10.0,
            "clip_reward": 10.0,
        },
    }
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    train_from_config(config, run_dir, total_timesteps=8, verbose=0)
    assert (run_dir / (MODEL_BASENAME + ".zip")).exists()
    assert (run_dir / VECNORM_FILENAME).exists()
