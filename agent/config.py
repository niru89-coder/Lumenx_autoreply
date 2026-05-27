"""Central config — loads .env once, exposes typed settings.

Import `settings` from here; do not read os.environ directly elsewhere.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env")


def _required(name: str) -> str:
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"Required env var {name!r} is not set. "
            f"Copy .env.example to .env and fill it in."
        )
    return val


class Settings:
    ANTHROPIC_API_KEY: str = _required("ANTHROPIC_API_KEY")
    LUMENX_BASE_URL: str = os.environ.get(
        "LUMENX_BASE_URL", "https://lumenx-demo.up.railway.app"
    )
    LUMENX_ADMIN_TOKEN: str = _required("LUMENX_ADMIN_TOKEN")
    LLM_CALLS_LOG: Path = REPO_ROOT / os.environ.get(
        "LLM_CALLS_LOG", "data/llm_calls.jsonl"
    )
    LUMENX_POLL_INTERVAL_SECONDS: int = int(
        os.environ.get("LUMENX_POLL_INTERVAL_SECONDS", "5")
    )


settings = Settings()
