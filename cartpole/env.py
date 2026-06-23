import numpy as np
import gymnasium as gym
from gymnasium import spaces

DANGER_LOW  = 0.10   # ~6 degrees in radians
DANGER_HIGH = 0.19   # ~11 degrees — near but not terminal (0.2095 = 12°)
WAYPOINT_X  = 2.0    # target distance from center (track limit is ±2.4)
WAYPOINT_THRESHOLD = 0.2   # within this distance = waypoint reached
WAYPOINT_BONUS     = 10.0  # one-time reward per waypoint reached
DANGER_BONUS       = 2.0   # per-step bonus for surviving in danger zone


class CartPoleCustomEnv(gym.Wrapper):
    """
    Wraps CartPole-v1 with two reward shaping layers:

    1. PNR danger zone: bonus reward for surviving with pole
       between DANGER_LOW and DANGER_HIGH radians.

    2. Waypoint traversal: alternating left/right targets.
       Continuous proximity reward + bonus on arrival.

    Observation and action spaces are unchanged from CartPole-v1.
    Termination conditions are unchanged from CartPole-v1.
    """

    def __init__(self, render_mode: str | None = None):
        env = gym.make("CartPole-v1", render_mode=render_mode)
        super().__init__(env)
        self.target_x: float = -WAYPOINT_X  # start by going left
        self.waypoints_reached: int = 0

    def reset(self, **kwargs):
        self.target_x = -WAYPOINT_X
        self.waypoints_reached = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        cart_pos, _, pole_angle, _ = obs

        # 1. PNR danger zone bonus
        abs_angle = abs(pole_angle)
        if DANGER_LOW <= abs_angle <= DANGER_HIGH:
            reward += DANGER_BONUS

        # 2. Waypoint traversal: proximity reward
        distance = abs(cart_pos - self.target_x)
        reward += 1.0 / (1.0 + distance)

        # 2b. Waypoint reached
        if distance < WAYPOINT_THRESHOLD:
            reward += WAYPOINT_BONUS
            self.target_x *= -1  # flip direction
            self.waypoints_reached += 1
            info["waypoint_reached"] = True

        info["target_x"] = self.target_x
        info["waypoints_reached"] = self.waypoints_reached
        return obs, reward, terminated, truncated, info
