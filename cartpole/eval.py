import sys
import gymnasium as gym
from stable_baselines3 import PPO
from cartpole.env import CartPoleCustomEnv

ENV_OPTIONS = {
    "cartpole": lambda: gym.make("CartPole-v1", render_mode="human"),
    "custom":   lambda: CartPoleCustomEnv(render_mode="human"),
}

def evaluate(model_path: str = "ppo_cartpole",
             episodes: int = 5,
             env_id: str = "cartpole") -> None:

    if env_id not in ENV_OPTIONS:
        raise ValueError(f"Unknown env_id '{env_id}'. "
                         f"Options: {list(ENV_OPTIONS)}")

    env = ENV_OPTIONS[env_id]()
    model = PPO.load(model_path)

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        waypoints = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            waypoints = info.get("waypoints_reached", 0)
        print(f"Episode {ep + 1}: reward={total_reward:.0f}  "
              f"waypoints={waypoints}")

    env.close()

if __name__ == "__main__":
    env_id = sys.argv[1] if len(sys.argv) > 1 else "cartpole"
    model_path = f"ppo_{env_id}"
    evaluate(model_path=model_path, env_id=env_id)
