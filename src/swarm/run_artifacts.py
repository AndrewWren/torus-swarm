"""
Shared helpers for run directories: one per training/eval run so artifacts
(logs, model, optional vecnorm, videos) do not overwrite and can be recovered with config.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Subdir and file names used under each run dir (keep in sync with plan)
RUNS_ROOT_DEFAULT = Path("runs")
EVAL_DIR = "eval"
VIDEOS_DIR = "videos"
CHECKPOINTS_DIR = "checkpoints"
TENSORBOARD_DIR = "tensorboard"
CONFIG_FILENAME = "config.json"
MODEL_BASENAME = "model"
VECNORM_FILENAME = "vecnorm.pkl"
FINAL_EVAL_FILENAME = "final_eval.json"


def get_run_dir(
    base_dir: Path | None = None,
    run_name: str | None = None,
    script_name: str | None = None,
) -> Path:
    """Create and return a run directory under base_dir.

    Name is <datetime>_<script_name>, plus _<run_name> if run_name is not None
    (e.g. 2025-02-14T12-30-00_script or 2025-02-14T12-30-00_script_my_run).
    Creates the run dir and subdirs: eval, videos, checkpoints.
    """
    base = base_dir if base_dir is not None else RUNS_ROOT_DEFAULT.resolve()
    ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S")
    script = script_name or "run"
    name = f"{ts}_{script}"
    if run_name is not None:
        name = f"{name}_{run_name}"
    run_dir = base / name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / EVAL_DIR).mkdir(exist_ok=True)
    # Don't make `VIDEOS_DIR` here; it's created by
    # `gymnasium/wrappers/rendering.py::RecordVideo`.
    (run_dir / CHECKPOINTS_DIR).mkdir(exist_ok=True)
    return run_dir


def write_config(run_dir: Path, config: dict[str, Any]) -> None:
    """Write config.json into run_dir (env + PPO + training info)."""
    path = run_dir / CONFIG_FILENAME
    with open(path, "w") as f:
        json.dump(config, f, indent=2)


def read_config(run_dir: Path) -> dict[str, Any] | None:
    """Read config.json from run_dir if present."""
    path = run_dir / CONFIG_FILENAME
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)
