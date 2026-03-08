"""
Optuna optimisation for env and training config parameters.

Maximises mean num_good from final evaluation of the localized_actions
(factorized PPO) pipeline. Trial budget (e.g. total_timesteps) is configurable via CLI.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import optuna

from localized_actions import train_from_config
from swarm.run_artifacts import (
    CHECKPOINTS_DIR,
    EVAL_DIR,
    FINAL_EVAL_FILENAME,
    RUNS_ROOT_DEFAULT,
    write_config,
)

# Penalty when final eval or mean_num_good is missing (minimise = maximise negative)
MISSING_OBJECTIVE_PENALTY = -1e6


def suggest_params(
    trial: optuna.Trial,
    total_timesteps_override: int,
    *,
    env_seed_base: int | None = None,
) -> dict[str, Any]:
    """Suggest high-priority env and training params from the trial.

    Fixed params (not searched):
      n_envs=8, n_epochs=10, clip_range=0.2, target_kl=0.02,
      critic_hidden=(256,256), VecNormalize on.

    n_steps is derived from T so each rollout always covers exactly
    8*n_envs=64 complete episodes regardless of episode length:
      n_steps = 8 * T
      batch_size = 16 * T  (4 equal minibatches per update, always divides evenly)
    """
    # Env: L, N, r fixed
    L, N, r = 16, 16, 4

    # --- High-priority env params ---
    T = trial.suggest_categorical("T", [32, 48, 64, 80])
    move_penalty = trial.suggest_float("move_penalty", 5e-5, 5e-3, log=True)
    collision_penalty = trial.suggest_float(
        "collision_penalty", 3e-3, 7e-2, log=True
    )

    # Deterministic per trial (reproducible), not optimized.
    env_seed = (
        env_seed_base if env_seed_base is not None else 0
    ) + trial.number

    # --- Derived rollout params (fixed relative to T) ---
    n_envs = 8
    n_steps = 8 * T  # 64 complete episodes per rollout update
    batch_size = 16 * T  # 4 minibatches; 16*T always divides 64*T evenly

    # --- High-priority PPO params ---
    learning_rate = trial.suggest_float("learning_rate", 5e-5, 5e-4, log=True)
    gamma = trial.suggest_float("gamma", 0.970, 0.992)
    ent_coef = trial.suggest_float("ent_coef", 5e-3, 0.15, log=True)

    # --- High-priority policy params ---
    per_agent_dim = trial.suggest_categorical("per_agent_dim", [64, 128, 256])

    total_timesteps = (
        math.ceil(total_timesteps_override / batch_size) * batch_size
    )
    eval_freq = n_steps

    return {
        "env": {
            "L": L,
            "N": N,
            "r": r,
            "T": T,
            "include_abs_pos": False,
            "move_penalty": move_penalty,
            "collision_penalty": collision_penalty,
            "seed": env_seed,
        },
        "ppo": {
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": 10,
            "learning_rate": learning_rate,
            "gamma": gamma,
            "ent_coef": ent_coef,
            "clip_range": 0.2,
            "target_kl": 0.02,
            "policy_kwargs": dict(
                per_agent_dim=per_agent_dim, critic_hidden=(256, 256)
            ),
        },
        "training": {
            "total_timesteps": total_timesteps,
            "eval_freq": eval_freq,
            "n_envs": n_envs,
            "script": "optuna_optimize",
        },
        "vec_normalize": {
            "norm_obs": False,
            "norm_reward": True,
            "clip_reward": 10.0,
        },
    }


def read_mean_num_good(run_dir: Path) -> float | None:
    """Read mean_num_good from run_dir/final_eval.json. Returns None if missing."""
    path = run_dir / FINAL_EVAL_FILENAME
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    return data.get("mean_num_good")


def create_objective(
    runs_root: Path,
    study_name: str,
    total_timesteps_override: int,
    run_start_ts: str,
    *,
    env_seed_base: int | None = None,
):
    """Build the Optuna objective that runs one trial and returns mean_num_good."""

    def objective(trial: optuna.Trial) -> float:
        pruned = [False]

        def on_eval_end(step_index: int, mean_num_good: float) -> bool:
            trial.report(mean_num_good, step=step_index)
            if trial.should_prune():
                pruned[0] = True
                return False
            return True

        params = suggest_params(
            trial,
            total_timesteps_override,
            env_seed_base=env_seed_base,
        )
        trial_run_dir = (
            runs_root
            / f"optuna_{run_start_ts}_{study_name}"
            / f"trial_{trial.number}"
        )
        trial_run_dir.mkdir(parents=True, exist_ok=True)
        (trial_run_dir / EVAL_DIR).mkdir(exist_ok=True)
        (trial_run_dir / CHECKPOINTS_DIR).mkdir(exist_ok=True)

        config = {
            "env": params["env"],
            "ppo": params["ppo"],
            "training": params["training"],
            "vec_normalize": params["vec_normalize"],
        }
        write_config(trial_run_dir, config)

        train_from_config(
            config,
            trial_run_dir,
            total_timesteps_override,
            verbose=0,
            on_eval_end=on_eval_end,
            record_video_on_end=True,
        )

        if pruned[0]:
            pruned_dir = trial_run_dir.parent / (
                trial_run_dir.name + "_pruned"
            )
            trial_run_dir.rename(pruned_dir)
            raise optuna.TrialPruned()

        mean_num_good = read_mean_num_good(trial_run_dir)
        if mean_num_good is None:
            return MISSING_OBJECTIVE_PENALTY
        return float(mean_num_good)

    return objective


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Optuna optimisation for env and training config (maximise mean num_good)."
    )
    parser.add_argument(
        "--study-name",
        type=lambda s: s.strip().replace(" ", "-"),
        default="swarm_life",
        help="Study name (used for run dirs and storage)",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=20,
        help="Number of Optuna trials",
    )
    parser.add_argument(
        "--total-timesteps",
        type=int,
        default=100_000,
        help="Training timesteps per trial (configurable budget)",
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default=None,
        help=f"Root for run dirs (default: {RUNS_ROOT_DEFAULT.resolve()})",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (e.g. sqlite:///optuna.db for resume)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for Optuna sampler (reproducibility)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Stop study after this many seconds",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel trials (each uses unique run dir)",
    )
    args = parser.parse_args()

    runs_root = (
        Path(args.runs_root).resolve()
        if args.runs_root
        else RUNS_ROOT_DEFAULT.resolve()
    )
    runs_root.mkdir(parents=True, exist_ok=True)

    run_start_ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    optuna_run_dir = runs_root / f"optuna_{run_start_ts}_{args.study_name}"
    optuna_run_dir.mkdir(parents=True, exist_ok=True)

    storage = args.storage
    if storage is None:
        storage = f"sqlite:///{optuna_run_dir / 'optuna_study.db'}"
    load_if_exists = bool(args.storage)

    study = optuna.create_study(
        study_name=args.study_name,
        direction="maximize",
        storage=storage,
        load_if_exists=load_if_exists,
        sampler=optuna.samplers.TPESampler(
            seed=args.seed,
            n_startup_trials=min(5, args.n_trials),
        ),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5,
            n_warmup_steps=20,
            interval_steps=1,
        ),
    )

    objective = create_objective(
        runs_root,
        args.study_name,
        args.total_timesteps,
        run_start_ts,
        env_seed_base=args.seed,
    )

    study.optimize(
        objective,
        n_trials=args.n_trials,
        timeout=args.timeout,
        n_jobs=args.n_jobs,
        show_progress_bar=True,
    )

    # Save study results to the run directory
    completed = [t for t in study.trials if t.value is not None]
    results = {
        "study_name": args.study_name,
        "n_trials": len(study.trials),
        "n_completed": len(completed),
        "total_timesteps_per_trial": args.total_timesteps,
    }
    if completed:
        best = study.best_trial
        results["best_trial_number"] = best.number
        results["best_value"] = best.value
        results["best_params"] = best.params
    results["trials"] = [
        {
            "number": t.number,
            "value": t.value,
            "params": t.params,
            "state": t.state.name,
        }
        for t in study.trials
    ]
    results_path = optuna_run_dir / "study_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nStudy results saved to {results_path}")

    if completed:
        print("Best trial:", study.best_trial.number)
        print("Best value (mean_num_good):", study.best_value)
        print("Best params:", study.best_params)


if __name__ == "__main__":
    main()
