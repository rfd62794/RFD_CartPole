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


class CartPoleCustomEnv(gym.Wrapper):
    """
    Wraps CartPole-v1 with two reward shaping layers:

    1. PNR danger zone: bonus reward for surviving with pole
       between DANGER_LOW and DANGER_HIGH radians.

    2. Recovery bonus: reward for reducing pole angle from danger.

    3. Waypoint traversal: progress toward target per step + bonus on arrival.

    Observation and action spaces are unchanged from CartPole-v1.
    Position-based termination is overridden — the agent survives
    going off-screen if the pole is still upright, but incurs a penalty.
    """

    def __init__(self, render_mode: str | None = None):
        env = gym.make("CartPole-v1", render_mode=render_mode)
        super().__init__(env)
        self.target_x: float = -WAYPOINT_X
        self.waypoints_reached: int = 0
        self._prev_cart_pos: float = 0.0
        self._prev_pole_angle: float = 0.0

    def reset(self, **kwargs):
        self.target_x = random.choice([-WAYPOINT_X, WAYPOINT_X])
        self.waypoints_reached = 0
        self._prev_cart_pos  = 0.0
        self._prev_pole_angle = 0.0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        cart_pos, _, pole_angle, _ = obs

        abs_angle = abs(pole_angle)

        # Override position-based death — only die if pole falls
        if terminated and abs_angle <= 0.2095:
            terminated = False  # went off-screen but pole is upright — survive

        # Penalty proportional to how far off-screen
        overshoot = max(0.0, abs(cart_pos) - 2.4)
        if overshoot > 0:
            reward -= overshoot * OFFSCREEN_PENALTY

        prev_abs_angle = abs(self._prev_pole_angle)

        # 1. Danger zone survival bonus (unchanged)
        if DANGER_LOW <= abs_angle <= DANGER_HIGH:
            reward += DANGER_BONUS

        # 2. Recovery bonus — pole moving back toward vertical from danger
        if prev_abs_angle > DANGER_LOW and abs_angle < prev_abs_angle:
            reward += (prev_abs_angle - abs_angle) * RECOVERY_SCALE

        # 3. Progress toward waypoint — reward direction, not position
        prev_dist = abs(self._prev_cart_pos - self.target_x)
        curr_dist = abs(cart_pos - self.target_x)
        reward += (prev_dist - curr_dist) * PROGRESS_SCALE

        # 4. Waypoint reached (unchanged)
        if curr_dist < WAYPOINT_THRESHOLD:
            reward += WAYPOINT_BONUS
            self.target_x *= -1
            self.waypoints_reached += 1
            info["waypoint_reached"] = True

        self._prev_cart_pos   = cart_pos
        self._prev_pole_angle = pole_angle
        info["target_x"]          = self.target_x
        info["waypoints_reached"] = self.waypoints_reached
        info["overshoot"]         = overshoot
        return obs, reward, terminated, truncated, info
