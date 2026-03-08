"""
Factorized (decentralized) SB3 PPO policy for SwarmLife-style MultiDiscrete actions.

Goal:
- Enforce: a_i = π(o_i) for each agent i, with shared parameters.
- i.e., NO cross-agent mixing in the actor by construction.
- Critic can be centralized (mean-pool agent embeddings) or decentralized; here we do a
  simple centralized critic (often much easier to learn).

Assumptions about env observation:
- Env returns a 1D float vector obs_flat of shape (N*C*P*P,)
  where it is a flattening of (N, C, P, P).
- Action space is MultiDiscrete([5]*N).

Works with:
- stable_baselines3>=1.8-ish (SB3 2.x also fine)
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path
from typing import Any

import gymnasium
import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.distributions import MultiCategoricalDistribution
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from swarm.callbacks import EvalCallbackWithNumGood, FinalStepNumGoodCallback
from swarm.run_artifacts import (
    CHECKPOINTS_DIR,
    EVAL_DIR,
    MODEL_BASENAME,
    VECNORM_FILENAME,
    VIDEOS_DIR,
    get_run_dir,
    write_config,
)
from swarm.swarm_life_sb3 import SwarmLifePatternConfig, SwarmLifePatternEnv
from utils.make_eval_grid import make_eval_grid

# ------------------- Small helper modules -------------------


def add_video_recorder(
    env: gymnasium.Env | gymnasium.Wrapper,
    video_folder: str | Path = "../videos",
    name_prefix: str = "test",
    episode_trigger: Callable[[int], bool] | None = None,
    fps: int = 5,
) -> gymnasium.Env | gymnasium.Wrapper:
    """Wrap an env with RecordVideo for recording evaluation episodes."""
    if episode_trigger is None:

        def episode_trigger(x: int) -> bool:
            return x == 0  # Record only first episode

    return RecordVideo(
        env,
        video_folder=str(video_folder),
        name_prefix=name_prefix,
        episode_trigger=episode_trigger,
        fps=fps,
    )


class PerAgentCNN(nn.Module):
    """
    Shared encoder applied independently to each agent patch (C,P,P) -> D embedding.
    """

    def __init__(self, C: int, P: int, D: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(C, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * P * P, D),
            nn.ReLU(),
        )

    def forward(self, x: th.Tensor) -> th.Tensor:
        # x: (B*N, C, P, P) -> (B*N, D)
        return self.net(x)


class CriticMLP(nn.Module):
    """
    Simple centralized critic: pool agent embeddings -> value.
    """

    def __init__(self, D: int, hidden: tuple[int, int] = (256, 256)) -> None:
        super().__init__()
        h1, h2 = hidden
        self.net = nn.Sequential(
            nn.Linear(D, h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def forward(self, pooled: th.Tensor) -> th.Tensor:
        # pooled: (B, D) -> (B, 1)
        return self.net(pooled)


# ------------------- The factorized policy -------------------


class FactorizedSwarmPolicy(ActorCriticPolicy):
    """
    Actor: per-agent independent categorical logits computed from each agent's
    local patch,
           with shared weights across agents. No pooling, no mixing.
    Critic: mean-pool agent embeddings, then MLP -> scalar value.

    This is still a standard SB3 ActorCriticPolicy; PPO will call:
      - forward()
      - evaluate_actions()
      - get_distribution()
      - predict_values()

    Important: We DO NOT use SB3's default mlp_extractor/action_net. We
    override the relevant pieces so the actor can't accidentally mix agents.
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Callable[[float], float],
        *,
        N: int,  # number of agents
        C: int,  # number of channels
        P: int,  # patch size
        per_agent_dim: int = 64,
        critic_hidden: tuple[int, int] = (256, 256),
        **kwargs,
    ) -> None:
        # Call parent but keep its init minimal; we won't use its default nets.
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            lr_schedule=lr_schedule,
            # We won't rely on net_arch/features_extractor from parent; keep
            # defaults
            **kwargs,
        )

        assert isinstance(action_space, spaces.MultiDiscrete), (
            "This policy expects MultiDiscrete action space."
        )
        assert len(action_space.nvec) == N, (
            f"Expected action_space.nvec length {N}, "
            f"got {len(action_space.nvec)}"
        )
        assert np.all(action_space.nvec == 5), (
            "This implementation assumes 5 actions per agent."
        )

        self.N = N
        self.C = C
        self.P = P
        self.D = per_agent_dim

        # Distribution: one categorical per agent, each size 5
        self._dist = MultiCategoricalDistribution(action_dims=[5] * self.N)

        # Shared per-agent encoder
        self.per_agent_encoder = PerAgentCNN(C=self.C, P=self.P, D=self.D)

        # Actor head: shared linear map D -> 5 logits, applied per agent
        # (no mixing!)
        self.actor_head = nn.Linear(self.D, 5)

        # Critic: centralized (mean pool), easier learning
        self.critic = CriticMLP(D=self.D, hidden=critic_hidden)

        # IMPORTANT: disable the parent’s default action/value nets usage
        # (We still keep optimizer, etc., from ActorCriticPolicy)
        self._build(lr_schedule)

    # ---------- Core encode/pool utilities ----------

    def _reshape_obs(self, obs: th.Tensor) -> th.Tensor:
        """
        obs: (B, obs_dim) flattened
        returns: (B, N, C, P, P)
        """
        # TODO Consider using a more efficient one-step reshaping operation
        B = obs.shape[0]
        expected = self.N * self.C * self.P * self.P
        if obs.shape[1] != expected:
            raise ValueError(
                f"Obs dim mismatch: got {obs.shape[1]}, "
                f"expected {expected} = N*C*P*P"
            )
        return obs.view(B, self.N, self.C, self.P, self.P)

    def _encode_agents(self, obs: th.Tensor) -> th.Tensor:
        """
        obs: (B, obs_dim) float
        returns agent embeddings: (B, N, D)
        """
        x = self._reshape_obs(obs)  # (B,N,C,P,P)
        B = x.shape[0]
        x = x.reshape(B * self.N, self.C, self.P, self.P)
        e = self.per_agent_encoder(x)  # (B*N,D)
        e = e.view(B, self.N, self.D)  # (B,N,D)
        return e

    def _actor_logits(self, agent_emb: th.Tensor, obs: th.Tensor) -> th.Tensor:
        """
        agent_emb: (B,N,D)
        obs: (B, obs_dim) — used to extract per-agent action masks from
            channel 0 of each agent's occupancy patch.
        returns logits_flat: (B, 5*N) in the ordering expected by
        `MultiCategoricalDistribution`, with -inf for blocked directions.
        """
        B = agent_emb.shape[0]
        logits = self.actor_head(agent_emb)  # (B,N,5)

        # Apply action mask derived from the occupancy patch (channel 0).
        # The center cell (r, r) is the agent's own position — always occupied.
        # Any neighbour cell with the same normalized value is also occupied.
        # Empty cells normalise to a strictly smaller value, so comparing
        # neighbour >= center is robust to VecNormalize.
        occ = self._reshape_obs(obs)[:, :, 0, :, :]  # (B,N,P,P)
        r = self.P // 2
        center = occ[
            :, :, r, r
        ]  # (B,N), always the occupied-cell normalized value
        blocked = th.stack(
            [
                th.zeros(
                    B, self.N, dtype=th.bool, device=logits.device
                ),  # 0: stay
                occ[:, :, r - 1, r] >= center,  # 1: up
                occ[:, :, r + 1, r] >= center,  # 2: down
                occ[:, :, r, r - 1] >= center,  # 3: left
                occ[:, :, r, r + 1] >= center,  # 4: right
            ],
            dim=-1,
        )  # (B,N,5)
        logits = logits.masked_fill(blocked, float("-inf"))

        return logits.reshape(B, 5 * self.N)

    def _critic_value(self, agent_emb: th.Tensor) -> th.Tensor:
        """
        agent_emb: (B,N,D) -> pooled (B,D) -> value (B,1)
        """
        # TODO Consider using attention pooling
        pooled = agent_emb.mean(dim=1)
        return self.critic(pooled)

    # ---------- SB3-required overrides ----------

    def get_distribution(
        self, obs: th.Tensor | dict[str, th.Tensor]
    ) -> MultiCategoricalDistribution:
        assert isinstance(obs, th.Tensor), "obs must be a tensor"
        agent_emb = self._encode_agents(obs)
        logits_flat = self._actor_logits(agent_emb, obs)
        return self._dist.proba_distribution(action_logits=logits_flat)

    def predict_values(
        self, obs: th.Tensor | dict[str, th.Tensor]
    ) -> th.Tensor:
        assert isinstance(obs, th.Tensor), "obs must be a tensor"
        agent_emb = self._encode_agents(obs)
        return self._critic_value(agent_emb)

    def forward(
        self,
        obs: th.Tensor | dict[str, th.Tensor],
        deterministic: bool = False,
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        Returns:
          actions: (B, N) int64
          values:  (B, 1)
          log_prob:(B,)
        """
        assert isinstance(obs, th.Tensor), "obs must be a tensor"
        agent_emb = self._encode_agents(obs)
        logits_flat = self._actor_logits(agent_emb, obs)
        dist = self._dist.proba_distribution(action_logits=logits_flat)

        if deterministic:
            actions = dist.mode()
        else:
            actions = dist.sample()

        # MultiCategoricalDistribution returns actions shape (B, N)
        log_prob = dist.log_prob(actions)
        values = self._critic_value(agent_emb)
        return actions, values, log_prob

    def evaluate_actions(
        self, obs: th.Tensor | dict[str, th.Tensor], actions: th.Tensor
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
        """
        PPO calls this for loss computation.
        Returns:
          values, log_prob, entropy
        """
        assert isinstance(obs, th.Tensor), "obs must be a tensor"
        assert isinstance(actions, th.Tensor), "actions must be a tensor"
        agent_emb = self._encode_agents(obs)
        logits_flat = self._actor_logits(agent_emb, obs)
        dist = self._dist.proba_distribution(action_logits=logits_flat)

        log_prob = dist.log_prob(actions)
        entropy = dist.entropy()
        values = self._critic_value(agent_emb)
        return values, log_prob, entropy


# ------------------- Shared training entry point -------------------


def train_from_config(
    config: dict[str, Any],
    run_dir: Path,
    total_timesteps: int,
    verbose: int = 1,
    on_eval_end: Callable[[int, float], bool] | None = None,
    record_video_on_end: bool = True,
) -> None:
    """Build env, model, and callbacks from a config dict and run PPO training."""
    env_cfg = config["env"]
    cfg = SwarmLifePatternConfig(**env_cfg)
    ppo_cfg = config["ppo"]
    n_envs = config["training"]["n_envs"]
    eval_freq = config["training"]["eval_freq"]

    total_timesteps = (
        math.ceil(total_timesteps / ppo_cfg["batch_size"])
        * ppo_cfg["batch_size"]
    )

    def make_env() -> SwarmLifePatternEnv:
        return SwarmLifePatternEnv(cfg, render_mode="rgb_array")

    env = make_vec_env(make_env, n_envs=n_envs, seed=cfg.seed or 0)

    vec_normalize_cfg = None
    if config["vec_normalize"] is not None:
        vec_normalize_cfg = config["vec_normalize"].copy()
        # VecNormalize needs the same gamma as PPO for correct return normalisation.
        vec_normalize_cfg["gamma"] = ppo_cfg["gamma"]
        env = VecNormalize(env, **vec_normalize_cfg)

    # Derive P and C from env config so they don't need to live in stored config.
    P = 2 * cfg.r + 1
    C = 1 + (2 if cfg.include_abs_pos else 0)

    ppo_kw = ppo_cfg.copy()
    # Policy is always FactorizedSwarmPolicy — the string key is not functional.
    ppo_kw.pop("total_timesteps", None)
    policy_kwargs = ppo_kw.pop("policy_kwargs").copy()
    # Inject derived fields; stored config only needs to carry per_agent_dim/critic_hidden.
    policy_kwargs.update(N=cfg.N, C=C, P=P)
    model = PPO(
        FactorizedSwarmPolicy,
        env,
        verbose=verbose,
        policy_kwargs=policy_kwargs,
        **ppo_kw,
    )

    eval_env = DummyVecEnv([lambda: Monitor(make_env())])
    if vec_normalize_cfg is not None:
        eval_env = VecNormalize(eval_env, **vec_normalize_cfg)

    def make_inner_eval_env_for_video(
        video_folder: str | Path,
        video_name_prefix: str,
        video_fps: int,
        n_episodes: int,
    ) -> gymnasium.Env:
        return add_video_recorder(
            Monitor(make_env()),
            video_folder=video_folder,
            name_prefix=video_name_prefix,
            fps=video_fps,
            episode_trigger=lambda episode_id: episode_id < n_episodes,
        )

    eval_callback = EvalCallbackWithNumGood(
        eval_env,
        eval_freq=eval_freq,
        best_model_save_path=str(run_dir / CHECKPOINTS_DIR),
        log_path=str(run_dir / EVAL_DIR),
        final_eval_dir=run_dir,
        n_eval_episodes=20,
        render=False,
        verbose=verbose,
        warn=True,
        record_video_on_end=record_video_on_end,
        video_folder=run_dir / VIDEOS_DIR,
        video_name_prefix="final_eval",
        video_fps=5,
        make_inner_eval_env_for_video=make_inner_eval_env_for_video,
        on_eval_end=on_eval_end,
    )

    model.learn(
        total_timesteps=total_timesteps,
        callback=[FinalStepNumGoodCallback(), eval_callback],
    )

    if vec_normalize_cfg is not None:
        env.save(str(run_dir / VECNORM_FILENAME))
    model.save(str(run_dir / MODEL_BASENAME))
    if record_video_on_end:
        make_eval_grid(run_dir)


# ------------------- Usage with PPO -------------------

if __name__ == "__main__":
    import argparse

    from run_artifacts import RUNS_ROOT_DEFAULT

    parser = argparse.ArgumentParser(
        description="Train factorized PPO on SwarmLife"
    )
    parser.add_argument(
        "--run-name",
        type=lambda s: s.strip().replace(" ", "-"),
        default=None,
        help=(
            "Run directory name (leading/trailing spaces removed, spaces "
            "replaced with hyphens; default: timestamp_localized_actions)"
        ),
    )
    parser.add_argument(
        "--runs-root",
        type=str,
        default=None,
        help=f"Root for run dirs (default: {RUNS_ROOT_DEFAULT})",
    )
    args_main = parser.parse_args()

    runs_root = (
        Path(args_main.runs_root).resolve()
        if args_main.runs_root
        else RUNS_ROOT_DEFAULT.resolve()
    )
    run_dir = get_run_dir(
        base_dir=runs_root,
        run_name=args_main.run_name,
        script_name="localized_actions",
    )
    print(f"Run directory: {run_dir}")

    cfg = SwarmLifePatternConfig(
        L=16,
        N=16,
        r=4,
        T=16,
        include_abs_pos=False,
        seed=0,
    )
    n_envs = 8
    n_steps = 2048
    batch_size = 8192
    timesteps_target = 2**21
    total_timesteps = math.ceil(timesteps_target / batch_size) * batch_size
    eval_freq = n_steps

    config = {
        "env": asdict(cfg),
        "ppo": {
            "n_steps": n_steps,
            "batch_size": batch_size,
            "n_epochs": 10,
            "learning_rate": 1e-4,
            "gamma": 0.995,
            "ent_coef": 0.05,
            "clip_range": 0.2,
            "target_kl": 0.02,
            "policy_kwargs": dict(per_agent_dim=64, critic_hidden=(256, 256)),
        },
        "training": {
            "total_timesteps": total_timesteps,
            "eval_freq": eval_freq,
            "n_envs": n_envs,
            "script": "localized_actions",
        },
        "vec_normalize": {
            "norm_obs": False,
            "norm_reward": True,
            "clip_reward": 10.0,
        },
    }
    write_config(run_dir, config)

    print(f"Training for {total_timesteps} timesteps\n\n\n")
    train_from_config(config, run_dir, total_timesteps, verbose=1)
