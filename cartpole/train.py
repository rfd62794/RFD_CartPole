import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from cartpole.logger import log_run

def train(total_timesteps: int = 50_000,
          model_path: str = "ppo_cartpole") -> float:
    env = gym.make("CartPole-v1")
    zip_path = f"{model_path}.zip"
    resumed = os.path.exists(zip_path)

    if resumed:
        print(f"Resuming from {zip_path}")
        model = PPO.load(zip_path, env=env)
    else:
        print("Starting fresh training run")
        model = PPO("MlpPolicy", env, verbose=1)

    model.learn(total_timesteps=total_timesteps,
                reset_num_timesteps=not resumed)
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
    train()
