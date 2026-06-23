import pytest

from cartpole.callbacks import VisualTrainingCallback


def test_callback_creates():
    cb = VisualTrainingCallback(env_id="cartpole")
    assert cb.env_id == "cartpole"
    assert cb.render_every == 5_000


def test_callback_invalid_env():
    with pytest.raises(ValueError, match="Unknown env_id"):
        VisualTrainingCallback(env_id="invalid")


def test_callback_rewards_empty():
    cb = VisualTrainingCallback(env_id="cartpole")
    assert cb.episode_rewards == []
