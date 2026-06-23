import csv
from datetime import datetime
from pathlib import Path

LOG_PATH = Path("logs/reward_history.csv")

def log_run(mean_reward: float, std_reward: float,
            timesteps: int, resumed: bool) -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "mean_reward",
                             "std_reward", "timesteps", "resumed"])
        writer.writerow([
            datetime.now().isoformat(timespec="seconds"),
            round(mean_reward, 2),
            round(std_reward, 2),
            timesteps,
            resumed,
        ])
