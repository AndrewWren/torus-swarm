from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


def _print_num_good_table(
    history: list[tuple[int, float]],
    final_num_good: float | None,
    final_timesteps: int,
) -> None:
    """Print a table of mean ep_final_step_num_good per eval checkpoint."""
    rows: list[tuple[int, float, bool]] = [(ts, v, False) for ts, v in history]
    if final_num_good is not None:
        rows.append((final_timesteps, final_num_good, True))
    if not rows:
        return
    ts_w = max(len("timesteps"), max(len(str(r[0])) for r in rows))
    print("\nep_final_step_num_good by timestep:")
    print(f"  {'timesteps':>{ts_w}} | mean_num_good")
    print(f"  {'-' * ts_w}-+-{'-' * 13}")
    for ts, v, is_final in rows:
        marker = " *" if is_final else ""
        print(f"  {ts:>{ts_w}} | {v:13.2f}{marker}")
    if any(r[2] for r in rows):
        print("  (* = final eval with render)")
    print()


def _print_histogram(
    values: Sequence[float | int],
    title: str,
    *,
    integer_bins: bool = False,
    num_bins: int = 10,
    max_bar_width: int = 50,
) -> None:
    """Print a text histogram to the terminal."""
    if not values:
        return
    arr = np.asarray(values)
    if integer_bins or np.all(arr == np.asarray(values, dtype=int)):
        # Count per integer value
        bin_vals = sorted(set(int(round(x)) for x in arr))
        counts = [int(np.sum(arr == b)) for b in bin_vals]
        labels = [str(b) for b in bin_vals]
    else:
        # Float bins
        counts, bin_edges = np.histogram(arr, bins=num_bins)
        counts = counts.tolist()
        labels = [
            f"{bin_edges[b]:.2f}-{bin_edges[b + 1]:.2f}"
            for b in range(num_bins)
        ]

    print(f"\n{title}")
    print("-" * (len(title) + 2))
    max_count = max(counts) if counts else 0
    for label, c in zip(labels, counts):
        bar_len = int(max_bar_width * c / max_count) if max_count > 0 else 0
        bar = "#" * bar_len
        print(f"  {label:>14}: {bar} ({c})")
    print()


class FinalStepNumGoodCallback(BaseCallback):
    """Track mean reward at the final step of each episode.

    Add
    `callback=FinalStepRewardCallback()`
    to the arguments of `model.learn`.

    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.final_step_num_good = []

    def _on_step(self) -> bool:
        # Check if any episode ended
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        if isinstance(dones, np.ndarray) and len(dones) > 0:
            for i, done in enumerate(dones):
                if done and i < len(infos) and infos[i] is not None:
                    num_good = infos[i].get("num_good")
                    if num_good is not None:
                        # Record the num_good at the final step
                        self.final_step_num_good.append(num_good)
        return True

    def _on_rollout_end(self) -> None:
        # Log the mean of final-step rewards
        if len(self.final_step_num_good) > 0:
            mean_num_good = np.mean(self.final_step_num_good)
            self.logger.record("rollout/ep_final_step_num_good", mean_num_good)
            # Clear the buffer for next rollout
            self.final_step_num_good.clear()


class EvalCallbackWithNumGood(EvalCallback):
    """EvalCallback that also tracks ep_final_step_num_good metric.

    This extends EvalCallback to collect and log the num_good value
    from the info dict at the final step of each evaluation episode.

    Optional: record_video_on_end, video_folder, video_name_prefix, video_fps,
    and make_inner_eval_env_for_video (callable (folder, prefix, fps) -> env)
    to run a final evaluation with render=True and RecordVideo in the chain.
    """

    def __init__(
        self,
        *args,
        final_eval_dir: str | Path | None = None,
        record_video_on_end: bool = False,
        video_folder: str | Path | None = None,
        video_name_prefix: str = "final_eval",
        video_fps: int = 5,
        make_inner_eval_env_for_video: Callable[
            [str | Path, str, int, int], Any
        ]
        | None = None,
        on_eval_end: Callable[[int, float], bool] | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._num_good_buffer: list[float] = []
        self._mean_action_prob_buffer: list[float] = []
        self._last_move_step_buffer: list[int | None] = []
        self._last_actual_move_step_buffer: list[int | None] = []
        self._last_position_set_change_step_buffer: list[int | None] = []
        self._eval_num_good_history: list[tuple[int, float]] = []
        self._final_eval_dir = (
            Path(final_eval_dir) if final_eval_dir is not None else None
        )
        self._record_video_on_end = record_video_on_end
        self._video_folder = video_folder
        self._video_name_prefix = video_name_prefix
        self._video_fps = video_fps
        self._make_inner_eval_env_for_video = make_inner_eval_env_for_video
        self._on_eval_end = on_eval_end
        self._stopped_by_hook = False

    def _log_num_good_callback(
        self, locals_: dict[str, Any], globals_: dict[str, Any]
    ) -> None:
        """
        Callback passed to the evaluate_policy function
        to log the num_good value at the final step of each episode.

        :param locals_:
        :param globals_:
        """
        info = locals_["info"]

        if locals_["done"]:
            num_good = info.get("num_good")
            if num_good is not None:
                self._num_good_buffer.append(num_good)

    def _compute_mean_action_prob(
        self, locals_: dict[str, Any]
    ) -> float | None:
        """Get mean_action_prob from info or compute from model. Returns None on failure."""
        info = locals_.get("info")
        mean_action_prob = info.get("mean_action_prob") if info else None
        if (
            mean_action_prob is None
            and hasattr(self, "model")
            and self.model is not None
        ):
            try:
                observations = locals_.get("observations")
                actions = locals_.get("actions")
                i = locals_.get("i")
                if (
                    observations is not None
                    and actions is not None
                    and i is not None
                ):
                    obs = observations[i]
                    action = actions[i]

                    policy = self.model.policy
                    obs_tensor = th.as_tensor(obs).unsqueeze(0)
                    if hasattr(self.model, "device"):
                        obs_tensor = obs_tensor.to(self.model.device)
                    with th.no_grad():
                        dist = policy.get_distribution(obs_tensor)
                        if hasattr(dist, "distribution"):
                            dists = dist.distribution
                            if isinstance(dists, list):
                                agent_max_probs = []
                                for d in dists:
                                    if hasattr(d, "probs"):
                                        agent_max_probs.append(
                                            d.probs[0].max().item()
                                        )
                                if agent_max_probs:
                                    mean_action_prob = float(
                                        np.mean(agent_max_probs)
                                    )
                            else:
                                if hasattr(dists, "probs"):
                                    probs = dists.probs[0]
                                    action_tensor = th.as_tensor(
                                        action
                                    ).unsqueeze(0)
                                    if hasattr(self.model, "device"):
                                        action_tensor = action_tensor.to(
                                            self.model.device
                                        )
                                    mean_action_prob = float(
                                        probs[action_tensor[0]].max().item()
                                    )
                        elif hasattr(dist, "probs"):
                            probs = dist.probs[0]
                            action_tensor = th.as_tensor(action).unsqueeze(0)
                            if hasattr(self.model, "device"):
                                action_tensor = action_tensor.to(
                                    self.model.device
                                )
                            mean_action_prob = float(
                                probs[action_tensor[0]].max().item()
                            )
            except Exception:
                pass
        return mean_action_prob

    def _log_mean_action_prob_callback(
        self, locals_: dict[str, Any], globals_: dict[str, Any]
    ) -> None:
        """
        Callback passed to the evaluate_policy function
        to log the mean_action_prob value at each step.

        :param locals_:
        :param globals_:
        """
        info = locals_.get("info")
        if info is None:
            info = {}
            locals_["info"] = info
        mean_action_prob = self._compute_mean_action_prob(locals_)
        if mean_action_prob is not None:
            info["mean_action_prob"] = mean_action_prob
            self._mean_action_prob_buffer.append(mean_action_prob)

    def _log_last_move_step_callback(
        self, locals_: dict[str, Any], globals_: dict[str, Any]
    ) -> None:
        """
        On episode end, record the step index from env info when any agent
        last moved.
        """
        info = locals_.get("info")
        done = locals_.get("done", False)
        if info is None or not done:
            return
        t = info.get("last_move_step")
        if t is None or (isinstance(t, int) and t < 0):
            self._last_move_step_buffer.append(None)
        else:
            self._last_move_step_buffer.append(int(t))

    def _log_last_actual_move_step_callback(
        self, locals_: dict[str, Any], globals_: dict[str, Any]
    ) -> None:
        """
        On episode end, record the step index from env info when any agent
        last actually moved (not just attempted).
        """
        info = locals_.get("info")
        done = locals_.get("done", False)
        if info is None or not done:
            return
        t = info.get("last_actual_move_step")
        if t is None or (isinstance(t, int) and t < 0):
            self._last_actual_move_step_buffer.append(None)
        else:
            self._last_actual_move_step_buffer.append(int(t))

    def _log_last_position_set_change_step_callback(
        self, locals_: dict[str, Any], globals_: dict[str, Any]
    ) -> None:
        """
        On episode end, record the step index from env info when the agent
        position set last changed.
        """
        info = locals_.get("info")
        done = locals_.get("done", False)
        if info is None or not done:
            return
        t = info.get("last_position_set_change_step")
        if t is None or (isinstance(t, int) and t < 0):
            self._last_position_set_change_step_buffer.append(None)
        else:
            self._last_position_set_change_step_buffer.append(int(t))

    def _run_evaluation(self) -> bool:
        """
        Run evaluation and log results.
        Returns True to continue training.
        """
        # Sync training and eval env if there is VecNormalize
        if self.model.get_vec_normalize_env() is not None:
            try:
                from stable_baselines3.common.vec_env import (
                    sync_envs_normalization,
                )

                sync_envs_normalization(self.training_env, self.eval_env)
            except AttributeError as e:
                raise AssertionError(
                    "Training and eval env are not wrapped the same way, "
                    "see https://stable-baselines3.readthedocs.io/en/master/"
                    "guide/callbacks.html#evalcallback and warning above."
                ) from e

        # Reset success rate buffer
        self._is_success_buffer = []
        # Reset num_good buffer
        self._num_good_buffer = []
        # Reset mean_action_prob buffer
        self._mean_action_prob_buffer = []
        # Reset last-move-step and last-position-set-change buffers
        self._last_move_step_buffer = []
        self._last_actual_move_step_buffer = []
        self._last_position_set_change_step_buffer = []

        from stable_baselines3.common.evaluation import evaluate_policy

        # Create combined callback that logs success, num_good,
        # mean_action_prob, and last-move-step
        def combined_callback(
            locals_: dict[str, Any], globals_: dict[str, Any]
        ) -> None:
            self._log_success_callback(locals_, globals_)
            self._log_num_good_callback(locals_, globals_)
            self._log_mean_action_prob_callback(locals_, globals_)
            self._log_last_move_step_callback(locals_, globals_)
            self._log_last_actual_move_step_callback(locals_, globals_)
            self._log_last_position_set_change_step_callback(locals_, globals_)

        episode_rewards, episode_lengths = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            render=self.render,
            deterministic=self.deterministic,
            return_episode_rewards=True,
            warn=self.warn,
            callback=combined_callback,
        )

        if self.log_path is not None:
            assert isinstance(episode_rewards, list)
            assert isinstance(episode_lengths, list)
            self.evaluations_timesteps.append(self.num_timesteps)
            self.evaluations_results.append(episode_rewards)
            self.evaluations_length.append(episode_lengths)

            kwargs = {}
            # Save success log if present
            if len(self._is_success_buffer) > 0:
                self.evaluations_successes.append(self._is_success_buffer)
                kwargs = dict(successes=self.evaluations_successes)

            np.savez(
                self.log_path,
                timesteps=self.evaluations_timesteps,
                results=self.evaluations_results,
                ep_lengths=self.evaluations_length,
                **kwargs,  # type: ignore[arg-type]
            )

        mean_reward, std_reward = (
            np.mean(episode_rewards),
            np.std(episode_rewards),
        )
        mean_ep_length, std_ep_length = (
            np.mean(episode_lengths),
            np.std(episode_lengths),
        )
        self.last_mean_reward = float(mean_reward)

        if self.verbose >= 1:
            print(
                f"Eval num_timesteps={self.num_timesteps}, "
                f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
            )
            print(
                f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}"
            )
        # Add to current Logger
        self.logger.record("eval/mean_reward", float(mean_reward))
        self.logger.record("eval/mean_ep_length", mean_ep_length)

        if len(self._is_success_buffer) > 0:
            success_rate = np.mean(self._is_success_buffer)
            if self.verbose >= 1:
                print(f"Success rate: {100 * success_rate:.2f}%")
            self.logger.record("eval/success_rate", success_rate)

        # Log num_good metric
        if len(self._num_good_buffer) > 0:
            mean_num_good = np.mean(self._num_good_buffer)
            std_num_good = np.std(self._num_good_buffer)
            if self.verbose >= 1:
                msg = (
                    f"Mean num_good: {mean_num_good:.2f} "
                    f"+/- {std_num_good:.2f}"
                )
                print(msg)
            self.logger.record("eval/ep_final_step_num_good", mean_num_good)
            self._eval_num_good_history.append(
                (self.num_timesteps, float(mean_num_good))
            )

        # Log mean_action_prob metric
        if len(self._mean_action_prob_buffer) > 0:
            mean_mean_action_prob = np.mean(self._mean_action_prob_buffer)
            std_mean_action_prob = np.std(self._mean_action_prob_buffer)
            if self.verbose >= 1:
                msg = (
                    f"Mean mean_action_prob: {mean_mean_action_prob:.4f} "
                    f"+/- {std_mean_action_prob:.4f}"
                )
                print(msg)
            self.logger.record(
                "eval/Mean mean_action_prob", mean_mean_action_prob
            )

        # Log last-move-step metrics (from env info; exclude no-move from means)
        n_no_move = sum(1 for x in self._last_move_step_buffer if x is None)
        if len(self._last_move_step_buffer) > 0:
            self.logger.record("eval/last_move_step_no_move", n_no_move)
            if self.verbose >= 1:
                print(f"Episodes with no move: {n_no_move}")
        if len(self._last_move_step_buffer) > 0:
            si_vals = [x for x in self._last_move_step_buffer if x is not None]
            if si_vals:
                mean_step = float(np.mean(si_vals))
                std_step = float(np.std(si_vals))
                if self.verbose >= 1:
                    print(
                        f"Mean step index of last move: {mean_step:.2f} "
                        f"+/- {std_step:.2f} (excl. {n_no_move} no-move)"
                    )
                self.logger.record("eval/last_move_step", mean_step)

        # Log last-actual-move-step metrics (same pattern as last_move_step)
        n_no_actual_move = sum(
            1 for x in self._last_actual_move_step_buffer if x is None
        )
        if len(self._last_actual_move_step_buffer) > 0:
            self.logger.record(
                "eval/last_actual_move_step_no_move", n_no_actual_move
            )
            if self.verbose >= 1:
                print(f"Episodes with no actual move: {n_no_actual_move}")
        if len(self._last_actual_move_step_buffer) > 0:
            si_vals = [
                x for x in self._last_actual_move_step_buffer if x is not None
            ]
            if si_vals:
                mean_step = float(np.mean(si_vals))
                std_step = float(np.std(si_vals))
                if self.verbose >= 1:
                    print(
                        f"Mean step index of last actual move: {mean_step:.2f} "
                        f"+/- {std_step:.2f} (excl. {n_no_actual_move} no-actual-move)"
                    )
                self.logger.record("eval/last_actual_move_step", mean_step)

        # Log last-position-set-change-step metrics (same pattern as last_move_step)
        n_no_pos_set_change = sum(
            1 for x in self._last_position_set_change_step_buffer if x is None
        )
        if len(self._last_position_set_change_step_buffer) > 0:
            self.logger.record(
                "eval/last_position_set_change_step_no_change",
                n_no_pos_set_change,
            )
            if self.verbose >= 1:
                print(
                    f"Episodes with no position set change: {n_no_pos_set_change}"
                )
        if len(self._last_position_set_change_step_buffer) > 0:
            si_vals = [
                x
                for x in self._last_position_set_change_step_buffer
                if x is not None
            ]
            if si_vals:
                mean_step = float(np.mean(si_vals))
                std_step = float(np.std(si_vals))
                if self.verbose >= 1:
                    print(
                        f"Mean step index of last position set change: "
                        f"{mean_step:.2f} +/- {std_step:.2f} "
                        f"(excl. {n_no_pos_set_change} no-change)"
                    )
                self.logger.record(
                    "eval/last_position_set_change_step", mean_step
                )

        # Dump log so evaluation results are printed with correct timestep
        self.logger.record(
            "time/total_timesteps",
            self.num_timesteps,
            exclude="tensorboard",
        )
        self.logger.dump(self.num_timesteps)

        continue_training = True
        if mean_reward > self.best_mean_reward:
            if self.verbose >= 1:
                print("New best mean reward!")
            import os

            if self.best_model_save_path is not None:
                self.model.save(
                    os.path.join(self.best_model_save_path, "best_model")
                )
            self.best_mean_reward = float(mean_reward)
            # Trigger callback on new best model, if needed
            if self.callback_on_new_best is not None:
                continue_training = self.callback_on_new_best.on_step()

        # Trigger callback after every evaluation, if needed
        if self.callback is not None:
            continue_training = continue_training and self._on_event()

        if self._on_eval_end is not None and self._eval_num_good_history:
            step_index = len(self._eval_num_good_history) - 1
            mean_ng = self._eval_num_good_history[-1][1]
            if not self._on_eval_end(step_index, mean_ng):
                self._stopped_by_hook = True
                continue_training = False

        return continue_training

    def _on_training_start(self) -> None:
        """Run evaluation before training starts."""
        if self.verbose >= 1:
            print("Running initial evaluation before training starts...")
        self._run_evaluation()

    def _on_step(self) -> bool:
        # Call parent's _on_step, but we need to override the callback
        # So we'll need to modify the evaluate_policy call
        if self.eval_freq > 0 and self.n_calls % self.eval_freq == 0:
            return self._run_evaluation()
        return True

    def _on_training_end(self) -> None:
        """Run one final evaluation with render=True (and optional video)."""
        if self._stopped_by_hook:
            self._record_video_on_end = False
        # Sync normalization so eval_env has latest stats
        if self.model.get_vec_normalize_env() is not None:
            try:
                from stable_baselines3.common.vec_env import (
                    sync_envs_normalization,
                )

                sync_envs_normalization(self.training_env, self.eval_env)
            except AttributeError:
                pass

        from stable_baselines3.common.evaluation import evaluate_policy

        def combined_callback(
            locals_: dict[str, Any], globals_: dict[str, Any]
        ) -> None:
            self._log_success_callback(locals_, globals_)
            self._log_num_good_callback(locals_, globals_)
            self._log_mean_action_prob_callback(locals_, globals_)
            self._log_last_move_step_callback(locals_, globals_)
            self._log_last_actual_move_step_callback(locals_, globals_)
            self._log_last_position_set_change_step_callback(locals_, globals_)

        use_video_env = (
            self._record_video_on_end
            and self._video_folder is not None
            and self._make_inner_eval_env_for_video is not None
        )
        video_eval_env = None

        if use_video_env:
            folder = str(self._video_folder)
            prefix = self._video_name_prefix
            fps = self._video_fps
            n_episodes = self.n_eval_episodes
            make_inner = self._make_inner_eval_env_for_video

            def make_video_inner():
                return make_inner(folder, prefix, fps, n_episodes)

            video_vec = DummyVecEnv([make_video_inner])
            if isinstance(self.eval_env, VecNormalize):
                video_eval_env = VecNormalize(
                    video_vec,
                    training=False,
                    norm_obs=self.eval_env.norm_obs,
                    norm_reward=self.eval_env.norm_reward,
                    gamma=self.eval_env.gamma,
                    clip_obs=self.eval_env.clip_obs,
                    clip_reward=self.eval_env.clip_reward,
                )
                if hasattr(self.eval_env, "obs_rms"):
                    video_eval_env.obs_rms = self.eval_env.obs_rms
                if hasattr(self.eval_env, "ret_rms"):
                    video_eval_env.ret_rms = self.eval_env.ret_rms
            else:
                video_eval_env = video_vec
            env_to_use = video_eval_env
        else:
            env_to_use = self.eval_env

        # Reset buffers so they reflect only this final eval
        self._is_success_buffer = []
        self._num_good_buffer = []
        self._mean_action_prob_buffer = []
        self._last_move_step_buffer = []
        self._last_actual_move_step_buffer = []
        self._last_position_set_change_step_buffer = []

        if self.verbose >= 1:
            print("Running final evaluation with render=True...")
        episode_rewards, episode_lengths = evaluate_policy(
            self.model,
            env_to_use,
            n_eval_episodes=self.n_eval_episodes,
            render=True,
            deterministic=self.deterministic,
            return_episode_rewards=True,
            warn=self.warn,
            callback=combined_callback,
        )
        if video_eval_env is not None:
            video_eval_env.close()

        # Compute metrics (mirror _run_evaluation)
        assert isinstance(episode_rewards, list)
        assert isinstance(episode_lengths, list)
        mean_reward = float(np.mean(episode_rewards))
        std_reward = float(np.std(episode_rewards))
        mean_ep_length = float(np.mean(episode_lengths))
        std_ep_length = float(np.std(episode_lengths))

        # Print (same style as _run_evaluation)
        if self.verbose >= 1:
            print(
                f"Final eval num_timesteps={self.num_timesteps}, "
                f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
            )
            print(
                f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}"
            )
        success_rate = None
        if len(self._is_success_buffer) > 0:
            success_rate = float(np.mean(self._is_success_buffer))
            if self.verbose >= 1:
                print(f"Success rate: {100 * success_rate:.2f}%")
        mean_num_good = None
        std_num_good = None
        if len(self._num_good_buffer) > 0:
            mean_num_good = float(np.mean(self._num_good_buffer))
            std_num_good = float(np.std(self._num_good_buffer))
            if self.verbose >= 1:
                print(
                    f"Mean num_good: {mean_num_good:.2f} +/- {std_num_good:.2f}"
                )
        mean_mean_action_prob = None
        std_mean_action_prob = None
        if len(self._mean_action_prob_buffer) > 0:
            mean_mean_action_prob = float(
                np.mean(self._mean_action_prob_buffer)
            )
            std_mean_action_prob = float(np.std(self._mean_action_prob_buffer))
            if self.verbose >= 1:
                print(
                    f"Mean mean_action_prob: {mean_mean_action_prob:.4f} "
                    f"+/- {std_mean_action_prob:.4f}"
                )

        # Print last-move-step reports (final eval; from env info)
        n_no_move_final = sum(
            1 for x in self._last_move_step_buffer if x is None
        )
        if self.verbose >= 1 and len(self._last_move_step_buffer) > 0:
            print(f"Episodes with no move: {n_no_move_final}")
        if self.verbose >= 1 and len(self._last_move_step_buffer) > 0:
            si_vals = [x for x in self._last_move_step_buffer if x is not None]
            if si_vals:
                print(
                    f"Mean step index of last move: "
                    f"{float(np.mean(si_vals)):.2f} +/- {float(np.std(si_vals)):.2f} "
                    f"(excl. {n_no_move_final} no-move)"
                )
        # Print last-actual-move-step reports (final eval)
        n_no_actual_move_final = sum(
            1 for x in self._last_actual_move_step_buffer if x is None
        )
        if self.verbose >= 1 and len(self._last_actual_move_step_buffer) > 0:
            print(f"Episodes with no actual move: {n_no_actual_move_final}")
        if self.verbose >= 1 and len(self._last_actual_move_step_buffer) > 0:
            si_vals = [
                x for x in self._last_actual_move_step_buffer if x is not None
            ]
            if si_vals:
                print(
                    f"Mean step index of last actual move: "
                    f"{float(np.mean(si_vals)):.2f} +/- {float(np.std(si_vals)):.2f} "
                    f"(excl. {n_no_actual_move_final} no-actual-move)"
                )
        # Print last-position-set-change-step reports (final eval)
        n_no_pos_set_change_final = sum(
            1 for x in self._last_position_set_change_step_buffer if x is None
        )
        if (
            self.verbose >= 1
            and len(self._last_position_set_change_step_buffer) > 0
        ):
            print(
                f"Episodes with no position set change: {n_no_pos_set_change_final}"
            )
        if (
            self.verbose >= 1
            and len(self._last_position_set_change_step_buffer) > 0
        ):
            si_vals = [
                x
                for x in self._last_position_set_change_step_buffer
                if x is not None
            ]
            if si_vals:
                print(
                    f"Mean step index of last position set change: "
                    f"{float(np.mean(si_vals)):.2f} +/- {float(np.std(si_vals)):.2f} "
                    f"(excl. {n_no_pos_set_change_final} no-change)"
                )

        # Print histograms at end of training
        if self.verbose >= 1:
            _print_histogram(
                episode_rewards,
                "Final eval: episode_rewards (per episode)",
                num_bins=10,
            )
            _print_histogram(
                episode_lengths,
                "Final eval: episode_lengths (per episode)",
                integer_bins=True,
            )
            if len(self._num_good_buffer) > 0:
                _print_histogram(
                    self._num_good_buffer,
                    "Final eval: num_good (per episode)",
                    num_bins=10,
                )
            if len(self._mean_action_prob_buffer) > 0:
                _print_histogram(
                    self._mean_action_prob_buffer,
                    "Final eval: mean_action_prob (per step)",
                    num_bins=10,
                )
            si_vals = [x for x in self._last_move_step_buffer if x is not None]
            if si_vals:
                _print_histogram(
                    si_vals,
                    "Final eval: step index of last move (per episode, excl. no-move)",
                    integer_bins=True,
                )
            si_actual_vals = [
                x for x in self._last_actual_move_step_buffer if x is not None
            ]
            if si_actual_vals:
                _print_histogram(
                    si_actual_vals,
                    "Final eval: step index of last actual move (per episode, excl. no-actual-move)",
                    integer_bins=True,
                )
            si_psc_vals = [
                x
                for x in self._last_position_set_change_step_buffer
                if x is not None
            ]
            if si_psc_vals:
                _print_histogram(
                    si_psc_vals,
                    "Final eval: step index of last position set change (per episode, excl. no-change)",
                    integer_bins=True,
                )

        # Print table of num_good across all eval checkpoints + final eval
        if self.verbose >= 1:
            _print_num_good_table(
                self._eval_num_good_history, mean_num_good, self.num_timesteps
            )

        # Save to final_eval.json at the top level of the run directory
        if self.log_path is not None:
            data: dict[str, Any] = {}

            # num_good fields first
            if mean_num_good is not None:
                data["mean_num_good"] = mean_num_good
            if std_num_good is not None:
                data["std_num_good"] = std_num_good
            if len(self._num_good_buffer) > 0:
                data["num_good"] = [float(x) for x in self._num_good_buffer]
            data["eval_num_good_history"] = [
                {"timesteps": ts, "mean_ep_final_step_num_good": v}
                for ts, v in self._eval_num_good_history
            ]

            # Run-level summary
            data["num_timesteps"] = self.num_timesteps
            data["mean_reward"] = mean_reward
            data["std_reward"] = std_reward
            data["mean_ep_length"] = mean_ep_length
            data["std_ep_length"] = std_ep_length
            data["episode_rewards"] = [float(r) for r in episode_rewards]
            data["episode_lengths"] = [int(x) for x in episode_lengths]
            if success_rate is not None:
                data["success_rate"] = success_rate
            if mean_mean_action_prob is not None:
                data["mean_mean_action_prob"] = mean_mean_action_prob
            if std_mean_action_prob is not None:
                data["std_mean_action_prob"] = std_mean_action_prob
            # Detailed mean_action_prob (one per step across all episodes)
            if len(self._mean_action_prob_buffer) > 0:
                data["mean_action_prob"] = [
                    float(x) for x in self._mean_action_prob_buffer
                ]
            # Last-move-step (per episode from env info); None = no move in episode
            if len(self._last_move_step_buffer) > 0:
                data["n_episodes_no_move"] = n_no_move_final
            if len(self._last_move_step_buffer) > 0:
                data["last_move_step"] = list(self._last_move_step_buffer)
                si_vals = [
                    x for x in self._last_move_step_buffer if x is not None
                ]
                if si_vals:
                    data["mean_last_move_step"] = float(np.mean(si_vals))
                    data["std_last_move_step"] = float(np.std(si_vals))
            # Last-actual-move-step (per episode); None = no actual move in episode
            if len(self._last_actual_move_step_buffer) > 0:
                data["n_episodes_no_actual_move"] = n_no_actual_move_final
            if len(self._last_actual_move_step_buffer) > 0:
                data["last_actual_move_step"] = list(
                    self._last_actual_move_step_buffer
                )
                si_vals = [
                    x
                    for x in self._last_actual_move_step_buffer
                    if x is not None
                ]
                if si_vals:
                    data["mean_last_actual_move_step"] = float(
                        np.mean(si_vals)
                    )
                    data["std_last_actual_move_step"] = float(np.std(si_vals))
            # Last-position-set-change-step (per episode); None = no change in episode
            if len(self._last_position_set_change_step_buffer) > 0:
                data["n_episodes_no_position_set_change"] = (
                    n_no_pos_set_change_final
                )
            if len(self._last_position_set_change_step_buffer) > 0:
                data["last_position_set_change_step"] = list(
                    self._last_position_set_change_step_buffer
                )
                si_vals = [
                    x
                    for x in self._last_position_set_change_step_buffer
                    if x is not None
                ]
                if si_vals:
                    data["mean_last_position_set_change_step"] = float(
                        np.mean(si_vals)
                    )
                    data["std_last_position_set_change_step"] = float(
                        np.std(si_vals)
                    )
            if self._final_eval_dir is not None:
                out_path = self._final_eval_dir / "final_eval.json"
            else:
                out_path = Path(self.log_path).parent / "final_eval.json"
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2)
            if self.verbose >= 1:
                print(f"Final eval data saved to {out_path}")
