import gymnasium as gym
from stable_baselines3 import PPO

def evaluate(model_path: str = "ppo_cartpole", episodes: int = 5):
    env = gym.make("CartPole-v1", render_mode="human")
    model = PPO.load(model_path)
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward
        print(f"Episode {ep + 1}: {total_reward:.0f}")
    env.close()

if __name__ == "__main__":
    evaluate()
