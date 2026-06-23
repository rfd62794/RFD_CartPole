import random
import numpy as np
import gymnasium as gym
from gymnasium import spaces

DANGER_LOW  = 0.10   # ~6 degrees in radians
DANGER_HIGH = 0.19   # ~11 degrees — near but not terminal (0.2095 = 12°)
WAYPOINT_X  = 2.0    # target distance from center (track limit is ±2.4)
WAYPOINT_THRESHOLD = 0.2   # within this distance = waypoint reached
WAYPOINT_BONUS     = 10.0  # one-time reward per waypoint reached
DANGER_BONUS       = 2.0   # per-step bonus for surviving in danger zone
RECOVERY_SCALE     = 10.0  # reward for reducing pole angle from danger
PROGRESS_SCALE     =  2.0  # reward for moving toward waypoint per step
OFFSCREEN_PENALTY  = 5.0  # per unit beyond ±2.4, per step
CORRECTION_BONUS   = 0.5   # angle-force alignment reward


class CartPoleCustomEnv(gym.Wrapper):
    """
    Stage 0 — Balance only: angle-force alignment bonus only.
               No waypoints. No overshoot survival. Standard termination.
    Stage 1 — Traverse: add waypoints + overshoot survival + progress reward.
    Stage 2 — Full: add danger zone + recovery bonus. Full complexity.

    Observation and action spaces are unchanged from CartPole-v1.
    Position-based termination is overridden (stage 1+) — the agent survives
    going off-screen if the pole is still upright, but incurs a penalty.
    """

    def __init__(self, render_mode: str | None = None, stage: int = 2):
        env = gym.make("CartPole-v1", render_mode=render_mode)
        super().__init__(env)
        assert stage in (0, 1, 2), f"stage must be 0, 1, or 2 — got {stage}"
        self.stage = stage
        self.target_x: float = -WAYPOINT_X
        self.waypoints_reached: int = 0
        self._prev_cart_pos: float = 0.0
        self._prev_pole_angle: float = 0.0
        self._last_action: int = 0

    def reset(self, **kwargs):
        self.target_x = random.choice([-WAYPOINT_X, WAYPOINT_X])
        self.waypoints_reached = 0
        self._prev_cart_pos  = 0.0
        self._prev_pole_angle = 0.0
        self._last_action = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        self._last_action = action
        obs, reward, terminated, truncated, info = self.env.step(action)
        cart_pos, _, pole_angle, _ = obs
        abs_angle = abs(pole_angle)
        prev_abs_angle = abs(self._prev_pole_angle)

        # ── Stage 0+ : angle-force alignment bonus ────────────────────────
        force_sign = 1 if action == 1 else -1
        angle_sign = 1 if pole_angle > 0 else -1
        if force_sign == angle_sign:
            reward += CORRECTION_BONUS

        # ── Stage 1+ : overshoot survival + progress + waypoints ──────────
        if self.stage >= 1:
            # Override position-based death — only die if pole falls
            if terminated and abs_angle <= 0.2095:
                terminated = False

            # Penalty proportional to distance off-screen
            overshoot = max(0.0, abs(cart_pos) - 2.4)
            if overshoot > 0:
                reward -= overshoot * OFFSCREEN_PENALTY

            # Progress toward waypoint
            prev_dist = abs(self._prev_cart_pos - self.target_x)
            curr_dist = abs(cart_pos - self.target_x)
            reward += (prev_dist - curr_dist) * PROGRESS_SCALE

            # Waypoint reached
            if curr_dist < WAYPOINT_THRESHOLD:
                reward += WAYPOINT_BONUS
                self.target_x *= -1
                self.waypoints_reached += 1
                info["waypoint_reached"] = True

        # ── Stage 2 : danger zone + recovery ──────────────────────────────
        if self.stage >= 2:
            if DANGER_LOW <= abs_angle <= DANGER_HIGH:
                reward += DANGER_BONUS
            if prev_abs_angle > DANGER_LOW and abs_angle < prev_abs_angle:
                reward += (prev_abs_angle - abs_angle) * RECOVERY_SCALE

        self._prev_cart_pos   = cart_pos
        self._prev_pole_angle = pole_angle
        info["target_x"]          = self.target_x
        info["waypoints_reached"] = self.waypoints_reached
        info["stage"]             = self.stage
        return obs, reward, terminated, truncated, info
