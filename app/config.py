"""Application configuration read from environment variables and validated at startup."""

import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from app.schemas import MAX_BATCH_SIZE


class ConfigError(ValueError):
    """Raised when an environment variable is missing-but-required or has an invalid value."""


@dataclass(frozen=True)
class Settings:
    model_dir: Path
    observation_dir: Path
    max_batch_size: int
    log_level: str
    source_revision: str | None
    port: int


def _int(env: Mapping[str, str], name: str, default: int, low: int, high: int) -> int:
    raw = env.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be an integer") from None
    if not low <= value <= high:
        raise ConfigError(f"{name} must be between {low} and {high}")
    return value


def load_settings(env: Mapping[str, str] | None = None) -> Settings:
    """Build Settings from `env` (defaults to os.environ). Raises ConfigError on bad values."""
    env = os.environ if env is None else env
    level = env.get("LOG_LEVEL", "INFO").upper()
    if level not in logging.getLevelNamesMapping() or level == "NOTSET":
        raise ConfigError("LOG_LEVEL must be a standard Python logging level name")
    observation_dir = Path(env.get("OBSERVATION_DIR", "observations"))
    if observation_dir.exists() and not observation_dir.is_dir():
        raise ConfigError("OBSERVATION_DIR must be a directory")
    return Settings(
        model_dir=Path(env.get("MODEL_DIR", "models/churn-v1")),
        observation_dir=observation_dir,
        # the schema enforces MAX_BATCH_SIZE as a hard ceiling, so the setting can only lower it
        max_batch_size=_int(env, "MAX_BATCH_SIZE", MAX_BATCH_SIZE, 1, MAX_BATCH_SIZE),
        log_level=level,
        source_revision=env.get("SOURCE_REVISION") or None,
        port=_int(env, "PORT", 8000, 1, 65535),
    )
