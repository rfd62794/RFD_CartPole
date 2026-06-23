from cartpole.compare import ALGORITHMS, POLICIES, SUPPORTS_VEC
from stable_baselines3 import PPO, DQN
from sb3_contrib import RecurrentPPO
from cartpole.env import CartPoleCustomEnv


def test_algo_map_complete():
    assert set(ALGORITHMS.keys()) == {"ppo", "recurrent", "dqn"}
    assert ALGORITHMS["ppo"] is PPO
    assert ALGORITHMS["recurrent"] is RecurrentPPO
    assert ALGORITHMS["dqn"] is DQN


def test_dqn_no_ent_coef():
    env = CartPoleCustomEnv()
    model = DQN("MlpPolicy", env, verbose=0)
    env.close()
    assert model is not None


def test_recurrent_single_env():
    assert SUPPORTS_VEC["recurrent"] is False
