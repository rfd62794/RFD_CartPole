from unittest.mock import patch
import numpy as np

from cartpole.env import CartPoleCustomEnv, CORRECTION_BONUS, DANGER_BONUS


def test_stage_0_no_waypoints():
    env = CartPoleCustomEnv(stage=0)
    env.reset()
    for _ in range(50):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            env.reset()
        assert "waypoint_reached" not in info
    env.close()


def test_stage_1_waypoints_active():
    env = CartPoleCustomEnv(stage=1)
    env.reset()
    env.target_x = -2.0
    fake_obs = np.array([-2.0, 0.0, 0.0, 0.0], dtype=np.float32)
    with patch.object(env, "env") as mock_env:
        mock_env.step.return_value = (fake_obs, 1.0, False, False, {})
        obs, reward, terminated, truncated, info = env.step(0)
    assert info.get("waypoint_reached") is True
    assert info["waypoints_reached"] == 1
    env.close()


def test_stage_2_danger_bonus():
    fake_obs = np.array([0.0, 0.0, 0.15, 0.0], dtype=np.float32)

    env1 = CartPoleCustomEnv(stage=1)
    env1.reset()
    with patch.object(env1, "env") as mock_env:
        mock_env.step.return_value = (fake_obs, 1.0, False, False, {})
        _, reward1, _, _, _ = env1.step(0)
    env1.close()

    env2 = CartPoleCustomEnv(stage=2)
    env2.reset()
    with patch.object(env2, "env") as mock_env:
        mock_env.step.return_value = (fake_obs, 1.0, False, False, {})
        _, reward2, _, _, _ = env2.step(0)
    env2.close()

    assert reward2 > reward1


def test_correction_bonus():
    fake_obs = np.array([0.0, 0.0, 0.1, 0.0], dtype=np.float32)

    env = CartPoleCustomEnv(stage=0)
    env.reset()

    # action=1 (push right), pole leaning right (angle > 0) → aligned → bonus
    with patch.object(env, "env") as mock_env:
        mock_env.step.return_value = (fake_obs, 1.0, False, False, {})
        _, reward_aligned, _, _, _ = env.step(1)

    env.reset()
    # action=0 (push left), pole leaning right (angle > 0) → misaligned → no bonus
    with patch.object(env, "env") as mock_env:
        mock_env.step.return_value = (fake_obs, 1.0, False, False, {})
        _, reward_misaligned, _, _, _ = env.step(0)

    assert reward_aligned > reward_misaligned
    env.close()
