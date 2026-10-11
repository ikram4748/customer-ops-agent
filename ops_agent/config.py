"""Settings, read from `.env` (see `.env.example`)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# The store lives in a fixed "today" so every scenario (return windows, ETAs) is reproducible.
TODAY = date(2026, 10, 1)


def _list(name: str, default: str) -> list[str]:
    return [x.strip() for x in os.getenv(name, default).split(",") if x.strip()]


@dataclass(frozen=True)
class Settings:
    api_keys: list[str] = field(default_factory=lambda: [
        k for k in (os.getenv("GEMINI_API_KEY", ""), os.getenv("GEMINI_API_KEY_2", "")) if k.strip()])
    # The agent and the simulated customer use different models: they draw on separate free-tier quotas,
    # and a customer played by the same model as the agent tends to be too cooperative.
    agent_models: list[str] = field(default_factory=lambda: _list("AGENT_MODELS", "gemini-3.5-flash-lite,gemini-3.1-flash-lite"))
    user_models: list[str] = field(default_factory=lambda: _list("USER_MODELS", "gemini-3.1-flash-lite,gemini-2.5-flash-lite"))
    requests_per_minute: int = int(os.getenv("REQUESTS_PER_MINUTE", "12"))
    cache_dir: Path = field(default_factory=lambda: Path(os.getenv("CACHE_DIR", str(ROOT / ".cache"))))
    data_dir: Path = field(default_factory=lambda: Path(os.getenv("DATA_DIR", str(ROOT / "data"))))


@lru_cache
def get_settings() -> Settings:
    return Settings()
