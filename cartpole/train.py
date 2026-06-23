import os
import sys
import gymnasium as gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import RecurrentPPO
from cartpole.logger import log_run
from cartpole.env import CartPoleCustomEnv
from cartpole.callbacks import VisualTrainingCallback

ENV_OPTIONS = {
    "cartpole": lambda: gym.make("CartPole-v1"),
    "custom":   lambda: CartPoleCustomEnv(),
}

ALGO_MAP = {
    "ppo":       (PPO,          "MlpPolicy"),
    "recurrent": (RecurrentPPO, "MlpLstmPolicy"),
    "dqn":       (DQN,          "MlpPolicy"),
}

# Algorithms incompatible with vectorized envs
NO_VEC = {"recurrent", "dqn"}


def train(total_timesteps: int = 100_000,
          model_path: str = "ppo_cartpole",
          env_id: str = "cartpole",
          visual: bool = False,
          render_every: int = 5_000,
          n_envs: int = 1,
          algo: str = "ppo") -> float:

    if env_id not in ENV_OPTIONS:
        raise ValueError(f"Unknown env_id '{env_id}'.")
    if algo not in ALGO_MAP:
        raise ValueError(f"Unknown algo '{algo}'. "
                         f"Options: {list(ALGO_MAP)}")

    AlgoClass, policy = ALGO_MAP[algo]

    # Vectorized env compatibility check
    use_vec = n_envs > 1 and algo not in NO_VEC
    if n_envs > 1 and algo in NO_VEC:
        print(f"Warning: {algo} does not support vectorized envs. "
              f"Using n_envs=1.")

    if use_vec:
        env = make_vec_env(ENV_OPTIONS[env_id], n_envs=n_envs)
    else:
        env = ENV_OPTIONS[env_id]()

    zip_path = f"{model_path}.zip"
    resumed = os.path.exists(zip_path)

    kwargs = {"verbose": 1}
    if algo != "dqn":
        kwargs["ent_coef"] = 0.05

    if resumed:
        print(f"Resuming from {zip_path}")
        model = AlgoClass.load(zip_path, env=env)
    else:
        print(f"Starting fresh [{env_id}] algo={algo} n_envs="
              f"{n_envs if use_vec else 1}")
        model = AlgoClass(policy, env, **kwargs)

    callback = None
    if visual and not use_vec:
        callback = VisualTrainingCallback(
            env_id=env_id, render_every=render_every
        )
    elif visual and use_vec:
        print("Warning: --visual ignored with vectorized envs.")

    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=not resumed,
        callback=callback,
    )
    model.save(model_path)

    eval_env = ENV_OPTIONS[env_id]()
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10
    )
    eval_env.close()
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    try:
        log_run(mean_reward, std_reward, total_timesteps, resumed)
    except Exception as e:
        print(f"Warning: logging failed: {e}")

    env.close()
    return mean_reward


if __name__ == "__main__":
    env_id  = sys.argv[1] if len(sys.argv) > 1 else "cartpole"
    algo    = next((a.split("=")[1] for a in sys.argv
                    if a.startswith("--algo=")), "ppo")
    visual  = "--visual" in sys.argv
    n_envs  = int(next((a.split("=")[1] for a in sys.argv
                        if a.startswith("--envs=")), "1"))
    model_path = f"ppo_{env_id}_{algo}"
    train(env_id=env_id, model_path=model_path,
          visual=visual, n_envs=n_envs, algo=algo)
