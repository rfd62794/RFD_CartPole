import gymnasium as gym


def test_env_creates():
    env = gym.make("CartPole-v1")
    env.close()


def test_observation_space():
    env = gym.make("CartPole-v1")
    assert env.observation_space.shape == (4,)
    env.close()


def test_action_space():
    env = gym.make("CartPole-v1")
    assert env.action_space.n == 2
    env.close()
