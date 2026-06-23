import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback
from cartpole.env import CartPoleCustomEnv

ENV_RENDER = {
    "cartpole": lambda: gym.make("CartPole-v1", render_mode="human"),
    "custom":   lambda: CartPoleCustomEnv(render_mode="human"),
}


class VisualTrainingCallback(BaseCallback):
    """
    Renders one episode in a pygame window every `render_every` steps.
    Prints step count and episode reward to terminal after each render.

    The render env is separate from the training env — training is not
    interrupted, only paused briefly for the render episode.
    """

    def __init__(self, env_id: str = "cartpole",
                 render_every: int = 5_000,
                 verbose: int = 0):
        super().__init__(verbose)
        if env_id not in ENV_RENDER:
            raise ValueError(f"Unknown env_id '{env_id}'. "
                             f"Options: {list(ENV_RENDER)}")
        self.env_id = env_id
        self.render_every = render_every
        self._render_env = None
        self.episode_rewards: list[tuple[int, float]] = []

    def _on_training_start(self) -> None:
        self._render_env = ENV_RENDER[self.env_id]()

    def _on_step(self) -> bool:
        if self.n_calls % self.render_every == 0:
            reward = self._run_rendered_episode()
            self.episode_rewards.append((self.n_calls, reward))
            print(f"  [visual] step={self.n_calls:>7}  "
                  f"reward={reward:.0f}")
        return True  # True = continue training

    def _on_training_end(self) -> None:
        if self._render_env is not None:
            self._render_env.close()
            self._render_env = None

    def _run_rendered_episode(self) -> float:
        obs, _ = self._render_env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = \
                self._render_env.step(action)
            done = terminated or truncated
            total_reward += reward
        return total_reward
