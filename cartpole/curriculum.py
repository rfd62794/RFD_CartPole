import os
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from cartpole.env import CartPoleCustomEnv
from cartpole.logger import log_run

STAGE_TIMESTEPS = {
    0: 50_000,   # balance only — fast, simple signal
    1: 150_000,  # traversal — medium complexity
    2: 300_000,  # full — high complexity
}


def make_custom_env(stage: int):
    return lambda: CartPoleCustomEnv(stage=stage)


def curriculum_train(model_path: str = "ppo_curriculum",
                     n_envs: int = 8,
                     visual: bool = False) -> float:
    """
    Runs staged curriculum training:
      Stage 0 → Stage 1 → Stage 2
    Weights carry forward between stages.
    """
    if visual:
        print("Warning: visual mode incompatible with vectorized envs — ignoring --visual")
        visual = False

    model = None
    mean_reward = 0.0

    for stage, timesteps in STAGE_TIMESTEPS.items():
        print(f"\n── Stage {stage} ── {timesteps:,} steps "
              f"({n_envs} envs) ──────────────")

        env = make_vec_env(make_custom_env(stage), n_envs=n_envs)

        if model is None:
            model = PPO("MlpPolicy", env, verbose=1,
                        ent_coef=0.05, n_steps=2048)
        else:
            model.set_env(env)

        model.learn(total_timesteps=timesteps,
                    reset_num_timesteps=(stage == 0))
        model.save(f"{model_path}_stage{stage}")
        print(f"Stage {stage} saved → {model_path}_stage{stage}.zip")

        env.close()

    # Final save and eval on stage 2
    model.save(model_path)
    eval_env = make_vec_env(make_custom_env(2), n_envs=1)
    from stable_baselines3.common.evaluation import evaluate_policy
    mean_reward, std_reward = evaluate_policy(
        model, eval_env, n_eval_episodes=10
    )
    eval_env.close()
    print(f"\nFinal mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    try:
        log_run(mean_reward, std_reward,
                sum(STAGE_TIMESTEPS.values()), False)
    except Exception as e:
        print(f"Warning: logging failed: {e}")

    return mean_reward


if __name__ == "__main__":
    import sys
    visual = "--visual" in sys.argv
    curriculum_train(visual=visual)
