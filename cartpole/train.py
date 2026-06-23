import os
import sys
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from cartpole.logger import log_run
from cartpole.env import CartPoleCustomEnv

ENV_OPTIONS = {
    "cartpole": lambda render_mode=None: gym.make("CartPole-v1", render_mode=render_mode),
    "custom":   lambda render_mode=None: CartPoleCustomEnv(render_mode=render_mode),
}

def train(total_timesteps: int = 100_000,
          model_path: str = "ppo_cartpole",
          env_id: str = "cartpole",
          visual: bool = False,
          render_every: int = 5_000) -> float:

    if env_id not in ENV_OPTIONS:
        raise ValueError(f"Unknown env_id '{env_id}'. "
                         f"Options: {list(ENV_OPTIONS)}")

    render_mode = "human" if visual else None
    env = ENV_OPTIONS[env_id](render_mode=render_mode)
    zip_path = f"{model_path}.zip"
    resumed = os.path.exists(zip_path)

    if resumed:
        print(f"Resuming from {zip_path}")
        model = PPO.load(zip_path, env=env)
    else:
        print(f"Starting fresh training run [{env_id}]")
        model = PPO("MlpPolicy", env, verbose=1, ent_coef=0.05)

    if visual:
        print("Visual mode: live render every training step")

    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=not resumed,
    )
    model.save(model_path)

    mean_reward, std_reward = evaluate_policy(
        model, env, n_eval_episodes=10
    )
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    try:
        log_run(mean_reward, std_reward, total_timesteps, resumed)
    except Exception as e:
        print(f"Warning: logging failed: {e}")

    env.close()
    return mean_reward

if __name__ == "__main__":
    env_id   = sys.argv[1] if len(sys.argv) > 1 else "cartpole"
    visual   = "--visual" in sys.argv
    model_path = f"ppo_{env_id}"
    train(env_id=env_id, model_path=model_path, visual=visual)
