from unittest.mock import patch
import numpy as np

from cartpole.env import CartPoleCustomEnv, DANGER_LOW, DANGER_HIGH, DANGER_BONUS


def test_custom_env_creates():
    env = CartPoleCustomEnv()
    env.close()


def test_custom_obs_space():
    env = CartPoleCustomEnv()
    assert env.observation_space.shape == (4,)
    env.close()


def test_custom_action_space():
    env = CartPoleCustomEnv()
    assert env.action_space.n == 2
    env.close()


def test_danger_bonus():
    env = CartPoleCustomEnv()
    env.reset()
    # Patch the underlying env.step to return a pole angle in the danger zone
    fake_obs = np.array([0.0, 0.0, 0.15, 0.0], dtype=np.float32)
    with patch.object(env, "env") as mock_env:
        mock_env.step.return_value = (fake_obs, 1.0, False, False, {})
        obs, reward, terminated, truncated, info = env.step(0)
    # Base reward (1.0) + danger bonus (2.0) = 3.0 minimum
    assert reward >= 1.0 + DANGER_BONUS
    env.close()


def test_waypoint_flip():
    env = CartPoleCustomEnv()
    env.reset()
    env.target_x = -2.0  # force deterministic target for test
    fake_obs = np.array([-2.0, 0.0, 0.0, 0.0], dtype=np.float32)
    with patch.object(env, "env") as mock_env:
        mock_env.step.return_value = (fake_obs, 1.0, False, False, {})
        obs, reward, terminated, truncated, info = env.step(0)
    assert info["waypoints_reached"] == 1
    assert info["target_x"] == 2.0  # flipped from -2.0
    env.close()
