"""Write-only management of API secrets in the project .env.

Secrets are written to .env (gitignored) and live-applied to os.environ. The
store NEVER returns secret values — only per-key `configured` booleans — so the
UI can show set/unset without exposing the value (blank input = keep existing).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv, set_key

logger = logging.getLogger(__name__)

SECRET_KEYS = ("LLM_API_KEY", "S2_API_KEY", "GITHUB_TOKEN", "SERPAPI_API_KEY")


def set_keys(env_path: Path | str, changes: dict[str, str]) -> dict[str, bool]:
    """Upsert each non-blank secret in `changes` into the .env at `env_path`.

    Blank/omitted values are left untouched ("hit enter to keep"). Only keys in
    SECRET_KEYS are accepted. Live-applies the changes (os.environ + reload).
    Returns the post-change `configured_status()`.
    """
    env_path = Path(env_path)
    if not env_path.exists():
        env_path.touch(mode=0o600)
    else:
        try:
            os.chmod(env_path, 0o600)
        except OSError as e:
            logger.warning("could not set .env permissions to 0600: %s", e)
    applied: list[str] = []
    for key, value in changes.items():
        if key not in SECRET_KEYS:
            raise ValueError(f"unknown secret key: {key!r}")
        value = (value or "").strip()
        if not value:
            continue                              # blank = keep existing
        set_key(str(env_path), key, value, quote_mode="auto")
        os.environ[key] = value
        applied.append(key)
    if applied:
        try:
            os.chmod(env_path, 0o600)
        except OSError as e:
            logger.warning("could not set .env permissions to 0600: %s", e)
        load_dotenv(env_path, override=True)
        from ndif_citations import config
        config.reload_settings()
    return configured_status()


def configured_status() -> dict[str, bool]:
    """Return {SECRET_KEY: bool} from the current environment. Never returns values."""
    return {k: bool(os.environ.get(k)) for k in SECRET_KEYS}
