"""Write-only management of API secrets in the project .env.

Secrets are written to .env (gitignored) and live-applied to os.environ. The
store NEVER returns secret values — only per-key `configured` booleans — so the
UI can show set/unset without exposing the value (blank input = keep existing).
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import dotenv_values, load_dotenv, set_key, unset_key

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
        # Strip one pasted surrounding quote pair (e.g. a copied "'ghp_…'") so the
        # quotes don't end up embedded in the stored/applied secret and break auth.
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1].strip()
        if not value:
            continue                              # blank = keep existing
        # quote_mode="never": dotenv's "auto" single-quotes any value that isn't
        # purely alphanumeric (str.isalnum() is False for '_' and '-'), so tokens
        # like github_pat_… / sk-ant-… were written as KEY='value'. Those quotes are
        # cosmetic (dotenv strips them on load), but they look alarming in .env and
        # confuse manual inspection. API keys/tokens have no spaces or shell-special
        # chars, so writing them raw is safe and round-trips cleanly.
        set_key(str(env_path), key, value, quote_mode="never")
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


def refresh_secrets_from_file(env_path: Path | str) -> dict[str, bool]:
    """Re-sync os.environ secrets to match the .env file, then return the status.

    Fixes stale "configured" status after an out-of-band .env edit: keys present
    in the file are applied to os.environ; SECRET_KEYS absent from the file are
    popped (so a removed key reflects as not-set). Scoped to SECRET_KEYS only —
    never touches model/url/email vars. (.env is the source of truth for this
    local single-user tool.)
    """
    env_path = Path(env_path)
    vals = dotenv_values(str(env_path)) if env_path.exists() else {}
    for key in SECRET_KEYS:
        if vals.get(key):
            os.environ[key] = vals[key]
        else:
            os.environ.pop(key, None)
    return configured_status()


def clear_key(env_path: Path | str, key: str) -> dict[str, bool]:
    """Remove a single secret from the .env and the live environment.

    The UI's "blank = keep existing" rule means set_keys can never unset a key;
    this is the explicit unset path. Live-applies (pop + reload) so the change
    takes effect without a restart. Returns the post-change status.
    """
    if key not in SECRET_KEYS:
        raise ValueError(f"unknown secret key: {key!r}")
    env_path = Path(env_path)
    if env_path.exists():
        unset_key(str(env_path), key)
    os.environ.pop(key, None)
    from ndif_citations import config
    config.reload_settings()
    return configured_status()


def configured_status() -> dict[str, bool]:
    """Return {SECRET_KEY: bool} from the current environment. Never returns values."""
    return {k: bool(os.environ.get(k)) for k in SECRET_KEYS}
