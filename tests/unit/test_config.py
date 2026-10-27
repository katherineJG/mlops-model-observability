from pathlib import Path

import pytest

from app.config import ConfigError, load_settings


def test_defaults() -> None:
    s = load_settings({})
    assert s.model_dir == Path("models/churn-v1")
    assert (s.max_batch_size, s.log_level, s.port, s.source_revision) == (100, "INFO", 8000, None)


def test_overrides() -> None:
    s = load_settings(
        {
            "MODEL_DIR": "x",
            "MAX_BATCH_SIZE": "10",
            "LOG_LEVEL": "debug",
            "PORT": "9000",
            "SOURCE_REVISION": "abc123",
        }
    )
    assert (s.model_dir, s.max_batch_size, s.log_level, s.port, s.source_revision) == (
        Path("x"),
        10,
        "DEBUG",
        9000,
        "abc123",
    )


@pytest.mark.parametrize(
    "env",
    [
        {"MAX_BATCH_SIZE": "0"},
        {"MAX_BATCH_SIZE": "101"},
        {"MAX_BATCH_SIZE": "ten"},
        {"PORT": "70000"},
        {"LOG_LEVEL": "LOUD"},
        {"LOG_LEVEL": "NOTSET"},
    ],
)
def test_invalid_values_fail_fast(env: dict[str, str]) -> None:
    with pytest.raises(ConfigError):
        load_settings(env)
