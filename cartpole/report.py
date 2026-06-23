import csv
from pathlib import Path

LOG_PATH = Path("logs/reward_history.csv")

def print_report() -> None:
    if not LOG_PATH.exists():
        print("No run history found.")
        return
    rows = list(csv.DictReader(LOG_PATH.open()))
    if not rows:
        print("Log file is empty.")
        return
    print(f"\n{'Run':<5} {'Timestamp':<22} {'Mean':>7} "
          f"{'Std':>7} {'Steps':>8} {'Resumed'}")
    print("-" * 60)
    for i, row in enumerate(rows, 1):
        print(f"{i:<5} {row['timestamp']:<22} "
              f"{float(row['mean_reward']):>7.2f} "
              f"{float(row['std_reward']):>7.2f} "
              f"{int(row['timesteps']):>8} "
              f"{row['resumed']}")
    rewards = [float(r["mean_reward"]) for r in rows]
    print("-" * 60)
    print(f"Runs: {len(rows)}  |  "
          f"Best: {max(rewards):.2f}  |  "
          f"Latest: {rewards[-1]:.2f}")

if __name__ == "__main__":
    print_report()
