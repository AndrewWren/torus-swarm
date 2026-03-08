"""
SwarmLife-Pattern Gymnasium env for Stable-Baselines3 PPO.

- Single world per Env instance (SB3 will vectorize with VecEnv).
- Action space: MultiDiscrete([5]*N) (one move per agent).
- Observation: concatenated per-agent local patches
  (occ + tgt + optional abs pos), flattened to 1D so SB3's
  MlpPolicy works out of the box.

To train:
  env = make_vec_env(lambda: SwarmLifePatternEnv(cfg), n_envs=8)
  model = PPO("MlpPolicy", env, ...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces
from gymnasium.core import RenderFrame, Wrapper


def manhattan_dist_torus(
    pos1: np.ndarray, pos2: np.ndarray, L: int
) -> np.ndarray:
    """
    Compute Manhattan distance between positions on a torus.

    Args:
        pos1: (N, 2) array of [x, y] positions
        pos2: (N, 2) array of [x, y] positions
        L: Size of the torus (width and height)

    Returns:
        Manhattan distances, (N, N) array
    """
    dx = np.abs(pos1[:, None, 0] - pos2[None, :, 0])
    dy = np.abs(pos1[:, None, 1] - pos2[None, :, 1])
    dx = np.minimum(dx, L - dx)
    dy = np.minimum(dy, L - dy)
    return dx + dy


@dataclass(frozen=True)
class SwarmLifePatternConfig:
    L: int = 32
    N: int = 64
    r: int = 2
    T: int = 256
    include_abs_pos: bool = True
    move_penalty: float = 1e-3
    collision_penalty: float = 0.01
    seed: Optional[int] = None


class SwarmLifePatternEnv(gym.Env):
    """
    Pattern-formation swarm on a toroidal grid with hard exclusion.

    Observation:
      per-agent patch (occ, tgt[, xnorm, ynorm]) of size P x P, P=2r+1,
      concatenated over agents then flattened.

    Action:
      MultiDiscrete([5]*N), values:
        0 stay, 1 up, 2 down, 3 left, 4 right
    """

    metadata = {"render_modes": ["ansi", "rgb_array"]}

    def __init__(
        self, cfg: SwarmLifePatternConfig, render_mode: Optional[str] = None
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.render_mode = render_mode

        self.L = cfg.L
        self.N = cfg.N
        self.r = cfg.r
        self.P = 2 * cfg.r + 1
        self.t = 0

        self._rng = np.random.default_rng(cfg.seed)

        # State
        self.pos = np.zeros((self.N, 2), dtype=np.int16)  # (N,2) int x,y
        self.occ = np.zeros(
            (self.L, self.L), dtype=np.bool_
        )  # (L,L) occupancy

        # Spaces
        self.action_space = spaces.MultiDiscrete(
            np.full((self.N,), 5, dtype=np.int64)
        )

        C = 1 + (2 if cfg.include_abs_pos else 0)
        obs_dim = self.N * C * self.P * self.P
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )

        self.moves: list[np.ndarray] = []

    # ---------------- Gymnasium API ----------------

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self.t = 0
        self._last_move_step = (
            -1
        )  # 0-based step index of last step with any move
        self._last_actual_move_step = (
            -1
        )  # last step with any agent actually moving
        self._last_position_set_change_step = -1
        self._sample_positions_uniform()
        self._prev_pos_set = frozenset(tuple(p) for p in self.pos)

        self.moves = []

        obs = self._observe_flat()
        matched, num_good = self._get_reward_and_num_good()
        info = {
            "t": self.t,
            "match": float(matched),
            "num_good": num_good,
            "last_move_step": (
                self._last_move_step if self._last_move_step >= 0 else None
            ),
            "last_actual_move_step": (
                self._last_actual_move_step
                if self._last_actual_move_step >= 0
                else None
            ),
            "last_position_set_change_step": (
                self._last_position_set_change_step
                if self._last_position_set_change_step >= 0
                else None
            ),
        }
        return obs, info

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        action = np.asarray(action, dtype=np.int64)
        if action.shape != (self.N,):
            raise ValueError(
                f"Expected action shape {(self.N,)}, got {action.shape}"
            )

        moved, bounced, penalized_bounce = self._apply_moves(action)
        self.moves.append(moved.astype(np.int16))

        if (action != 0).any():
            self._last_move_step = self.t  # 0-based step we are finishing
        if moved.any():
            self._last_actual_move_step = self.t

        pos_set = frozenset(tuple(p) for p in self.pos)
        if pos_set != self._prev_pos_set:
            self._last_position_set_change_step = self.t
        self._prev_pos_set = pos_set

        matched, num_good = self._get_reward_and_num_good()
        move_cost = self.cfg.move_penalty * float((action != 0).mean())
        collision_cost = self.cfg.collision_penalty * float(
            penalized_bounce.sum()
        )
        reward = float(matched - move_cost - collision_cost)

        self.t += 1
        terminated = False
        truncated = self.t >= self.cfg.T

        obs = self._observe_flat()
        info = {
            "t": self.t,
            "match": float(matched),
            "moved_frac": float(moved.mean()),
            "num_good": num_good,
            "num_bounced": int(bounced.sum()),
            "num_penalized": int(penalized_bounce.sum()),
            "collision_cost": collision_cost,
            "last_move_step": (
                self._last_move_step if self._last_move_step >= 0 else None
            ),
            "last_actual_move_step": (
                self._last_actual_move_step
                if self._last_actual_move_step >= 0
                else None
            ),
            "last_position_set_change_step": (
                self._last_position_set_change_step
                if self._last_position_set_change_step >= 0
                else None
            ),
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> Optional[RenderFrame]:
        if self.render_mode == "ansi":
            grid = np.full((self.L, self.L), ".", dtype="<U1")
            ys = self.pos[:, 1]
            xs = self.pos[:, 0]
            grid[ys, xs] = "A"
            return "\n".join("".join(row) for row in grid)
        elif self.render_mode == "rgb_array":
            # Scale factor: each grid cell becomes scale x scale pixels
            scale = 16
            img_h, img_w = self.L * scale, self.L * scale
            img = np.zeros((img_h, img_w, 3), dtype=np.uint8)

            # Background: light gray
            img[:, :] = [240, 240, 240]

            # Draw grid lines (subtle)
            for i in range(self.L + 1):
                y = i * scale
                if y < img_h:
                    img[y, :] = [220, 220, 220]
                x = i * scale
                if x < img_w:
                    img[:, x] = [220, 220, 220]

            # Draw agents as filled circles; colour by Manhattan-1 neighbour count:
            #   0 neighbours -> red; 1-4 neighbours -> light to dark green
            _FILL_COLORS = [
                [255, 50, 50],  # 0: red
                [144, 238, 144],  # 1: light green
                [50, 205, 50],  # 2: medium green
                [0, 160, 0],  # 3: dark green
                [0, 100, 0],  # 4: very dark green
            ]
            _BORDER_COLORS = [
                [200, 0, 0],  # 0
                [100, 180, 100],  # 1
                [0, 160, 0],  # 2
                [0, 120, 0],  # 3
                [0, 60, 0],  # 4
            ]
            agent_radius = scale // 3  # Radius of agent circle
            ys = self.pos[:, 1]
            xs = self.pos[:, 0]
            distances = manhattan_dist_torus(self.pos, self.pos, self.L)
            np.fill_diagonal(distances, self.L)
            num_neighbors = (distances == 1).sum(axis=1)

            for i, (y, x) in enumerate(zip(ys, xs)):
                # Center of agent in scaled coordinates
                center_y = y * scale + scale // 2
                center_x = x * scale + scale // 2

                # Only compute mask for bounding box around agent
                y_min = max(0, center_y - agent_radius)
                y_max = min(img_h, center_y + agent_radius + 1)
                x_min = max(0, center_x - agent_radius)
                x_max = min(img_w, center_x + agent_radius + 1)

                # Create coordinate grids for this region
                y_coords, x_coords = np.ogrid[y_min:y_max, x_min:x_max]
                mask = (
                    (x_coords - center_x) ** 2 + (y_coords - center_y) ** 2
                ) <= agent_radius**2
                n = min(int(num_neighbors[i]), 4)
                fill_color = _FILL_COLORS[n]
                border_color = _BORDER_COLORS[n]
                img[y_min:y_max, x_min:x_max][mask] = fill_color

                # Add a darker border
                border_mask = (
                    (x_coords - center_x) ** 2 + (y_coords - center_y) ** 2
                ) > (agent_radius - 2) ** 2
                border_mask = (
                    border_mask
                    & ((x_coords - center_x) ** 2 + (y_coords - center_y) ** 2)
                    <= agent_radius**2
                )
                img[y_min:y_max, x_min:x_max][border_mask] = border_color

            return img
        else:
            return None

    # ---------------- Dynamics ----------------

    def _apply_moves(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Deterministic synchronous update with pre-action occupancy masking.

        Moves to cells that are occupied before this step begins are rejected
        outright (this also prevents swaps and cycles).  Of the remaining
        candidates, any two that target the same empty cell both bounce back
        and incur the collision penalty.
        """
        L = self.L

        dx = np.array([0, 0, 0, -1, +1], dtype=np.int16)
        dy = np.array([0, -1, +1, 0, 0], dtype=np.int16)

        a = action.astype(np.int16)
        prop = self.pos.copy()
        prop[:, 0] = (prop[:, 0] + dx[a]) % L
        prop[:, 1] = (prop[:, 1] + dy[a]) % L

        prop_cell = prop[:, 1].astype(np.int32) * L + prop[:, 0].astype(
            np.int32
        )
        curr_cell = self.pos[:, 1].astype(np.int32) * L + self.pos[
            :, 0
        ].astype(np.int32)

        movers_mask = a != 0

        # Reject moves whose destination is occupied pre-action (includes swaps/cycles)
        dest_occupied = self.occ.reshape(-1)[prop_cell]
        accept = movers_mask & ~dest_occupied

        # Reject same-cell collisions among remaining candidates
        counts = np.bincount(prop_cell[accept], minlength=L * L)
        multi = accept & (counts[prop_cell] > 1)
        accept = accept & ~multi

        moved = accept
        if np.any(moved):
            old_linear = curr_cell[moved]
            self.occ.flat[np.unique(old_linear)] = False
            self.pos[moved] = prop[moved]
            self.occ[self.pos[moved, 1], self.pos[moved, 0]] = True

        bounced = movers_mask & ~moved
        penalized_bounce = multi  # only same-cell collisions incur the penalty

        if self.occ.sum() != self.N:
            raise AssertionError(
                f"_apply_moves left {int(self.occ.sum())} cells occupied, "
                f"expected {self.N}"
            )
        return moved, bounced, penalized_bounce

    def action_masks(self) -> np.ndarray:
        """
        Return valid-action mask of shape (N*5,) for use with MaskablePPO.

        mask[i*5 + a] is True when action a is valid for agent i.
        Action 0 (stay) is always valid.  Actions 1-4 (up/down/left/right)
        are valid only when the destination cell is not currently occupied.
        """
        dx = np.array([0, 0, 0, -1, +1], dtype=np.int16)
        dy = np.array([0, -1, +1, 0, 0], dtype=np.int16)

        mask = np.ones((self.N, 5), dtype=np.bool_)
        for a_idx in range(1, 5):
            dest_x = (self.pos[:, 0] + dx[a_idx]) % self.L
            dest_y = (self.pos[:, 1] + dy[a_idx]) % self.L
            mask[:, a_idx] = ~self.occ[dest_y, dest_x]

        return mask.reshape(-1)

    # ---------------- Target / init ----------------

    def _sample_positions_uniform(self) -> None:
        L, N = self.L, self.N
        self.occ[:] = False
        flat = self._rng.choice(L * L, size=N, replace=False)
        y = (flat // L).astype(np.int16)
        x = (flat % L).astype(np.int16)
        self.pos[:, 0] = x
        self.pos[:, 1] = y
        self.occ[y, x] = True

    # ---------------- Obs / reward ----------------

    def _get_reward_and_num_good(self) -> tuple[np.float32, float]:
        distances = manhattan_dist_torus(self.pos, self.pos, self.L)
        # Perception zone is a square patch (Chebyshev / L∞ distance)
        dx = np.abs(self.pos[:, None, 0] - self.pos[None, :, 0])
        dy = np.abs(self.pos[:, None, 1] - self.pos[None, :, 1])
        dx = np.minimum(dx, self.L - dx)
        dy = np.minimum(dy, self.L - dy)
        in_zone = (
            np.maximum(dx, dy) <= self.r
        )  # (N, N), includes the agent itself
        np.fill_diagonal(distances, self.L)
        num_neighbors = (distances == 1).sum(axis=1)
        individual_reward = np.minimum(num_neighbors, 4).astype(np.float32) / 4
        reward = (in_zone @ individual_reward) / self.N
        return reward.mean(), float(num_neighbors.mean())

    def _observe_flat(self) -> np.ndarray:
        """
        Returns flattened float32 vector of shape (N*C*P*P,).
        """
        L, r, P, N = self.L, self.r, self.P, self.N
        occ_pad = np.pad(
            self.occ.astype(np.float32), ((r, r), (r, r)), mode="wrap"
        )

        C = 1 + (2 if self.cfg.include_abs_pos else 0)
        obs = np.zeros((N, C, P, P), dtype=np.float32)

        y0 = self.pos[:, 1].astype(np.int32) + r
        x0 = self.pos[:, 0].astype(np.int32) + r

        # Small loops are OK at N=64; can vectorize later if needed.
        for i in range(N):
            yy, xx = y0[i], x0[i]
            obs[i, 0, :, :] = occ_pad[yy - r : yy + r + 1, xx - r : xx + r + 1]

        if self.cfg.include_abs_pos:
            xnorm = (self.pos[:, 0].astype(np.float32) / (L - 1.0)).clip(0, 1)
            ynorm = (self.pos[:, 1].astype(np.float32) / (L - 1.0)).clip(0, 1)
            obs[:, 1, :, :] = xnorm[:, None, None]
            obs[:, 2, :, :] = ynorm[:, None, None]

        return obs.reshape(-1)


# ---------------- Wrapper for action probabilities ----------------
class ActionProbWrapper(Wrapper):
    """
    For use in training - not needed with EvalCallbackWithNumGood.

    Wrapper that adds mean_action_prob to the info dict for each step.

    The wrapper stores the last observation and computes action
    probabilities using the policy model. Set the model using
    set_model() before use.
    """

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.last_obs: Optional[np.ndarray] = None
        self.model = None

    def set_model(self, model) -> None:
        """Set the model to use for computing action probabilities."""
        self.model = model

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self.last_obs = obs
        return obs, info

    def step(
        self, action: np.ndarray
    ) -> Tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        # Compute action probabilities if model is available
        mean_action_prob = None
        if self.model is not None and self.last_obs is not None:
            try:
                # Get the policy
                policy = self.model.policy

                # Convert observation to tensor
                obs_tensor = th.as_tensor(self.last_obs).unsqueeze(0)
                if hasattr(self.model, "device"):
                    obs_tensor = obs_tensor.to(self.model.device)

                # Get action distribution using policy's get_distribution
                with th.no_grad():
                    dist = policy.get_distribution(obs_tensor)

                    # For MultiDiscrete, the distribution has probs attribute
                    # Shape should be (batch, N, num_actions) for MultiDiscrete
                    if hasattr(dist, "distribution"):
                        # MultiDiscrete distribution
                        dists = dist.distribution
                        if isinstance(dists, list):
                            # Get probabilities for each agent
                            agent_probs = []
                            action_tensor = th.as_tensor(action).unsqueeze(0)
                            if hasattr(self.model, "device"):
                                action_tensor = action_tensor.to(
                                    self.model.device
                                )
                            for i, d in enumerate(dists):
                                probs = d.probs[0]  # (num_actions,)
                                agent_probs.append(
                                    probs[action_tensor[0, i]].item()
                                )
                            mean_action_prob = float(np.mean(agent_probs))
                        else:
                            # Single distribution
                            probs = dists.probs[0]
                            action_tensor = th.as_tensor(action).unsqueeze(0)
                            if hasattr(self.model, "device"):
                                action_tensor = action_tensor.to(
                                    self.model.device
                                )
                            mean_action_prob = float(
                                probs[action_tensor[0]].max().item()
                            )
                    elif hasattr(dist, "probs"):
                        # Direct probs attribute
                        probs = dist.probs[0]
                        action_tensor = th.as_tensor(action).unsqueeze(0)
                        if hasattr(self.model, "device"):
                            action_tensor = action_tensor.to(self.model.device)
                        mean_action_prob = float(
                            probs[action_tensor[0]].max().item()
                        )
            except Exception:
                # If computation fails, continue without mean_action_prob
                pass

        # Call the wrapped environment's step with mean_action_prob
        obs, reward, terminated, truncated, info = self.env.step(
            action, mean_action_prob=mean_action_prob
        )

        # Update last observation
        self.last_obs = obs

        return obs, reward, terminated, truncated, info
