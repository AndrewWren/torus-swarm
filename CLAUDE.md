# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup

All source code lives in `src/`. Install dependencies and activate the venv:
```bash
pip install -r requirements.txt
cd src && source .venv/bin/activate
```

## Commands

Run from the `src/` directory with the venv active.

```bash
# Run all tests
pytest tests/

# Run a single test
pytest tests/test_swarm_life_collision_swap.py::test_pairwise_swap_both_bounce_back

# Single training run
python localized_actions.py

# Hyperparameter optimisation
python optuna_optimize.py --study-name my-study --n-trials 20 --total-timesteps 500000
# Resume a study using SQLite storage:
python optuna_optimize.py --study-name my-study --storage sqlite:///runs/optuna_.../optuna_study.db

# Load and evaluate a saved model
python load_and_eval.py
```

## Architecture

The project trains decentralized swarm agents on a toroidal grid to form connected clusters, using SB3's PPO with a custom factorized policy.

### Environment — `swarm/swarm_life_sb3.py`

`SwarmLifePatternEnv` is a standard Gymnasium env. Key facts:
- **Observation**: flat `(N*C*P*P,)` float32 vector — each agent sees a `P×P` (P=2r+1) square occupancy patch around itself (Chebyshev neighborhood), plus optionally 2 channels of normalized absolute x/y position. `C = 1 + (2 if include_abs_pos else 0)`.
- **Action**: `MultiDiscrete([5]*N)` — one action per agent (0=stay, 1=up, 2=down, 3=left, 4=right).
- **Collision resolution** in `_apply_moves`: synchronous update with three rejection cases — same-cell multi-agent collision, swap/cycle detection, occupancy-blocked destination. Iterates until stable.
- **Reward** computed in `_get_reward_and_num_good`: returns a scalar reward and `num_good` (mean number of Manhattan-distance-1 neighbors per agent; maximum achievable is 3.0, by a solid 4×4 block — see `docs/num_good_maximum_proof.md` at repo root).
- `SwarmLifePatternConfig` is a frozen dataclass; pass it to the env constructor.

### Policy — `localized_actions.py`

`FactorizedSwarmPolicy` (subclass of SB3's `ActorCriticPolicy`) enforces `a_i = π(o_i)`:
- **`PerAgentCNN`**: shared 2-layer CNN + MLP encoder applied independently to each agent's `(C, P, P)` patch → `(per_agent_dim,)` embedding.
- **Actor**: single shared linear `(per_agent_dim → 5)` applied per agent — no cross-agent mixing.
- **Critic**: centralized `CriticMLP` taking mean-pooled agent embeddings `(B, per_agent_dim)` → scalar value.
- `policy_kwargs` must include `N`, `C`, `P`, `per_agent_dim`, `critic_hidden`.

### Training pipeline

1. `make_vec_env` wraps `SwarmLifePatternEnv` × n_envs, optionally with `VecNormalize`.
2. PPO uses `FactorizedSwarmPolicy` and `EvalCallbackWithNumGood` from `swarm/callbacks.py`.
3. `EvalCallbackWithNumGood` logs mean reward, `num_good`, `last_move_step`, renders `rgb_array` videos, and saves `final_eval.json` at the end.
4. Artifacts organized by `swarm/run_artifacts.py`: each run gets a timestamped directory under `runs/` containing `config.json`, `final_eval.json`, `model.zip`, `vecnorm.pkl`, `eval/`, `videos/`, `checkpoints/`.

### Hyperparameter optimization — `optuna_optimize.py`

Wraps the training pipeline in an Optuna objective that maximises `mean_num_good` from `final_eval.json`. Fixed env params: L=16, N=16, r=4. Tuned params: T, penalties, PPO params (lr, gamma, ent_coef), policy arch (per_agent_dim). Uses TPE sampler with SQLite storage (auto-created per study) for resumable runs. Derived rollout params: `n_steps = 8*T`, `batch_size = 16*T`.

### Coordinate convention

`pos` is `(N, 2)` with columns `[x, y]`. Occupancy grid `occ` is indexed `[y, x]`. This asymmetry is intentional throughout — be careful when indexing.
