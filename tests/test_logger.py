import csv
from pathlib import Path

from cartpole.logger import log_run


def test_log_creates_file(tmp_path, monkeypatch):
    log_path = tmp_path / "reward_history.csv"
    monkeypatch.setattr("cartpole.logger.LOG_PATH", log_path)
    log_run(500.0, 0.0, 50000, False)
    assert log_path.exists()


def test_log_appends(tmp_path, monkeypatch):
    log_path = tmp_path / "reward_history.csv"
    monkeypatch.setattr("cartpole.logger.LOG_PATH", log_path)
    log_run(500.0, 0.0, 50000, False)
    log_run(480.0, 10.0, 50000, True)
    rows = list(csv.reader(log_path.open()))
    assert len(rows) == 3  # header + 2 data rows


def test_log_columns(tmp_path, monkeypatch):
    log_path = tmp_path / "reward_history.csv"
    monkeypatch.setattr("cartpole.logger.LOG_PATH", log_path)
    log_run(500.0, 0.0, 50000, False)
    with log_path.open() as f:
        reader = csv.reader(f)
        header = next(reader)
    assert header == ["timestamp", "mean_reward", "std_reward",
                      "timesteps", "resumed"]
