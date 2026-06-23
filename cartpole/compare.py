"""
Algorithm comparison runner.
Trains PPO, RecurrentPPO, and DQN on the custom env for equal timesteps.
Prints a comparison table. Logs all runs via logger.

Usage:
  python -m cartpole.compare
  python -m cartpole.compare --timesteps=200000
"""

import sys
import time
import gymnasium as gym
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from sb3_contrib import RecurrentPPO
from cartpole.env import CartPoleCustomEnv
from cartpole.logger import log_run

ALGORITHMS = {
    "ppo": PPO,
    "recurrent": RecurrentPPO,
    "dqn": DQN,
}

POLICIES = {
    "ppo":       "MlpPolicy",
    "recurrent": "MlpLstmPolicy",
    "dqn":       "MlpPolicy",
}

# DQN is incompatible with vectorized envs — single env only
SUPPORTS_VEC = {
    "ppo":       True,
    "recurrent": False,  # RecurrentPPO requires single env
    "dqn":       False,
}


def run_comparison(timesteps: int = 200_000,
                   n_envs: int = 4) -> list[dict]:
    results = []

    for algo_name, AlgoClass in ALGORITHMS.items():
        print(f"\n── {algo_name.upper()} ── {timesteps:,} steps ──────────")
        start = time.time()

        if SUPPORTS_VEC[algo_name] and n_envs > 1:
            env = make_vec_env(lambda: CartPoleCustomEnv(), n_envs=n_envs)
        else:
            env = CartPoleCustomEnv()

        # DQN does not accept ent_coef — rebuild without it
        if algo_name == "dqn":
            model = AlgoClass(POLICIES[algo_name], env, verbose=0)
        else:
            model = AlgoClass(
                POLICIES[algo_name], env, verbose=0, ent_coef=0.05
            )

        model.learn(total_timesteps=timesteps)
        model.save(f"ppo_compare_{algo_name}")

        eval_env = CartPoleCustomEnv()
        mean_reward, std_reward = evaluate_policy(
            model, eval_env, n_eval_episodes=10
        )
        eval_env.close()
        env.close()

        elapsed = time.time() - start
        result = {
            "algo":        algo_name,
            "mean_reward": round(mean_reward, 2),
            "std_reward":  round(std_reward, 2),
            "timesteps":   timesteps,
            "elapsed_s":   round(elapsed, 1),
        }
        results.append(result)
        print(f"  reward={mean_reward:.2f} ± {std_reward:.2f} "
              f"in {elapsed:.0f}s")

        try:
            log_run(mean_reward, std_reward, timesteps, False)
        except Exception as e:
            print(f"  Warning: logging failed: {e}")

    # Print comparison table
    print("\n" + "═" * 55)
    print(f"{'Algorithm':<12} {'Mean':>8} {'Std':>8} {'Time(s)':>9}")
    print("─" * 55)
    for r in results:
        print(f"{r['algo']:<12} {r['mean_reward']:>8.2f} "
              f"{r['std_reward']:>8.2f} {r['elapsed_s']:>9.1f}")
    print("═" * 55)
    winner = max(results, key=lambda x: x["mean_reward"])
    print(f"Winner: {winner['algo'].upper()} "
          f"({winner['mean_reward']:.2f})")

    return results


if __name__ == "__main__":
    timesteps = int(next(
        (a.split("=")[1] for a in sys.argv if a.startswith("--timesteps=")),
        200_000
    ))
    run_comparison(timesteps=timesteps)
