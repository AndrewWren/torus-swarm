"""
Load saved PPO model and VecNormalize stats from localized_actions training,
and run evaluation on a new test environment with a new seed.
"""

from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from localized_actions import FactorizedSwarmPolicy, add_video_recorder
from swarm.run_artifacts import (
    MODEL_BASENAME,
    VECNORM_FILENAME,
    VIDEOS_DIR,
    read_config,
)
from swarm.swarm_life_sb3 import SwarmLifePatternConfig, SwarmLifePatternEnv


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load saved model and run on new test env"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Run directory (load model, vecnorm, config from it; save video to run_dir/videos)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for test environment"
    )
    parser.add_argument(
        "--model-path", type=str, default="ppo_factorized_swarmlife.zip"
    )
    parser.add_argument("--vecnorm-path", type=str, default="vecnorm.pkl")
    parser.add_argument(
        "--n-steps",
        type=int,
        default=None,
        help="Max steps per episode (default: cfg.T)",
    )
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument(
        "--no-deterministic", action="store_false", dest="deterministic"
    )
    parser.add_argument(
        "--video-dir",
        type=str,
        default="../videos",
        help="Directory for saved videos (ignored if --run-dir is set)",
    )
    parser.add_argument(
        "--video-prefix",
        type=str,
        default="load_eval",
        help=(
            "Title/prefix for the saved video file (e.g. load_test-episode-0)"
        ),
    )
    args = parser.parse_args()

    if args.run_dir is not None:
        run_dir = Path(args.run_dir).resolve()
        model_path = str(run_dir / MODEL_BASENAME)
        vecnorm_path = run_dir / VECNORM_FILENAME
        video_dir = run_dir / VIDEOS_DIR
        video_prefix = args.video_prefix
        config = read_config(run_dir)
        if config and "env" in config:
            env_cfg = config["env"]
            cfg = SwarmLifePatternConfig(**env_cfg)
            cfg = dataclasses.replace(cfg, seed=args.seed)
            print(f"Using env config from {run_dir / 'config.json'}")
        else:
            cfg = SwarmLifePatternConfig(
                L=16, N=16, r=4, T=32, include_abs_pos=False, seed=args.seed
            )
    else:
        model_path = args.model_path
        vecnorm_path = Path(args.vecnorm_path)
        video_dir = Path(args.video_dir).resolve()
        video_prefix = args.video_prefix
        cfg = SwarmLifePatternConfig(
            L=16,
            N=16,
            r=4,
            T=32,
            include_abs_pos=False,
            seed=args.seed,
        )
        config = None

    max_steps = args.n_steps if args.n_steps is not None else cfg.T

    print(
        f"Video will be saved to: {video_dir / (video_prefix + '-episode-0.mp4')}"
    )

    def make_env():
        env = SwarmLifePatternEnv(cfg, render_mode="rgb_array")
        return add_video_recorder(env, video_dir, video_prefix)

    # Single-env vec env for testing (video recorder wraps the inner env)
    test_venv = DummyVecEnv([make_env])
    # Load saved normalization stats when training used VecNormalize
    if args.run_dir is not None and config is not None:
        use_vecnorm = bool(config.get("vec_normalize"))
    else:
        use_vecnorm = True  # no config: try to load if file exists
    if use_vecnorm and vecnorm_path.exists():
        test_env = VecNormalize.load(
            str(vecnorm_path),
            test_venv,
        )
        test_env.training = False  # don't update running mean/std
        test_env.norm_reward = False  # report raw rewards when evaluating
    else:
        if use_vecnorm and not vecnorm_path.exists():
            print(
                f"Warning: {vecnorm_path} not found, "
                "running without observation normalization"
            )
        test_env = test_venv

    # Load model with custom policy (policy_kwargs are stored in the zip)
    model = PPO.load(
        model_path,
        env=test_env,
        custom_objects={"policy": FactorizedSwarmPolicy},
    )

    try:
        obs = test_env.reset()
        episode_return = 0.0
        step = 0

        for _ in range(max_steps):
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, rewards, dones, infos = test_env.step(action)
            test_env.env_method("render", indices=[0])
            episode_return += float(rewards[0])
            step += 1

            if len(infos) > 0 and "num_good" in infos[0]:
                ig = infos[0]
                print(
                    f"  step {step}  r={rewards[0]:.4f}  "
                    f"num_good={ig['num_good']}  match={ig.get('match', '?')}"
                )

            if dones[0]:
                print(
                    f"Episode finished at step {step}  "
                    f"total_return={episode_return:.4f}"
                )
                obs = test_env.reset()
                episode_return = 0.0
                step = 0
    finally:
        test_env.close()
    print("Done.")


if __name__ == "__main__":
    main()
