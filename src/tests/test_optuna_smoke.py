"""Smoke tests for optuna_optimize.py."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import optuna
import pytest

from optuna_optimize import (
    create_objective,
    read_mean_num_good,
    suggest_params,
)
from swarm.run_artifacts import FINAL_EVAL_FILENAME


def test_suggest_params_structure() -> None:
    """suggest_params returns a config dict with correct keys and derived rollout values."""
    study = optuna.create_study(direction="maximize")
    captured: dict = {}

    def objective(trial: optuna.Trial) -> float:
        captured["params"] = suggest_params(
            trial, total_timesteps_override=1000
        )
        return 1.0

    study.optimize(objective, n_trials=1)
    params = captured["params"]
    assert set(params) == {"env", "ppo", "training", "vec_normalize"}
    assert params["env"]["L"] == 16
    assert params["env"]["N"] == 16
    T = params["env"]["T"]
    assert params["ppo"]["n_steps"] == 8 * T
    assert params["ppo"]["batch_size"] == 16 * T


def test_read_mean_num_good_missing(tmp_path: Path) -> None:
    """Returns None when final_eval.json is absent."""
    assert read_mean_num_good(tmp_path) is None


def test_read_mean_num_good_present(tmp_path: Path) -> None:
    """Returns the stored float when final_eval.json exists."""
    (tmp_path / FINAL_EVAL_FILENAME).write_text(
        json.dumps({"mean_num_good": 2.5})
    )
    assert read_mean_num_good(tmp_path) == pytest.approx(2.5)


def test_create_objective_one_trial(tmp_path: Path) -> None:
    """create_objective returns a callable; a 1-trial study completes and reports the objective."""
    runs_root = tmp_path / "runs"
    runs_root.mkdir()

    def fake_train(
        config: dict, run_dir, total_timesteps: int, **kwargs
    ) -> None:
        (run_dir / FINAL_EVAL_FILENAME).write_text(
            json.dumps({"mean_num_good": 1.5, "mean_reward": 0.5})
        )

    with patch("optuna_optimize.train_from_config", side_effect=fake_train):
        objective = create_objective(
            runs_root=runs_root,
            study_name="test-study",
            total_timesteps_override=512,
            run_start_ts="2000-01-01T00-00-00",
        )
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=1)

    assert len(study.trials) == 1
    assert study.best_value == pytest.approx(1.5)
