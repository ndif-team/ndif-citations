"""Runtime-editable settings that override config.py defaults.

This module is intentionally self-contained — it must NOT import config.py
(that would create a circular import because config imports us at module level).
"""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Canonical defaults — must stay in sync with config.py hard-coded values.
# ---------------------------------------------------------------------------
DEFAULTS: dict[str, Any] = {
    "min_paper_year": 2024,
    "shared_paper_threshold": 5,
    "excluded_github_repos": ["ndif-team/nnsight"],
    "known_course_sources": ["callummcdougall/ARENA_3.0"],
    "course_name_patterns": [
        "ARENA", "MATS", "CBAI",
        " course ", "course project", "coursework", "exercises", "capstone",
        "homework", "assignment",
    ],
    "ndif_keywords": ["nnsight", "NNsight", "NDIF", "ndif.us", "nnsight.net", "import nnsight"],
    "ndif_readme_keywords_regex": [r"\bNDIF\b"],
    "ndif_readme_keywords_substr": ["ndif.us", "NDIF cluster", "hosted on NDIF"],
    "ndif_readme_negative_patterns": [
        "NDIF Discord",
        "NDIF Pilot Program",
        "join the NDIF",
    ],
    "llm_model": "meta/llama-3.1-70b-instruct",
    "llm_base_url": "https://integrate.api.nvidia.com/v1",
    "llm_rate_limit_sleep": 12.0,
    "s2_rate_limit_sleep": 3.0,
    "github_rate_limit_sleep": 2.0,
    "publish_target": None,
}


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------

def load(path: Path | str) -> dict[str, Any]:
    """Return settings dict merged over DEFAULTS.

    If *path* does not exist, returns a deep copy of DEFAULTS unchanged.
    Unknown keys in the file are passed through (caller ignores them or not).
    """
    path = Path(path)
    result = copy.deepcopy(DEFAULTS)
    if not path.exists():
        return result
    with path.open() as fh:
        overrides: dict = json.load(fh)
    # Deep-merge: file keys win
    result.update(overrides)
    return result


def load_overrides(path: Path | str) -> dict[str, Any]:
    """Return ONLY the keys explicitly present in the settings file (no DEFAULTS fill).

    Returns ``{}`` if the file is absent.  Keys present in the file but absent
    from ``DEFAULTS`` are silently ignored (they would be unknown keys).  This
    ensures callers that only want to apply explicit overrides are not affected
    by absent-file or future schema additions.
    """
    import logging
    _log = logging.getLogger(__name__)

    path = Path(path)
    if not path.exists():
        return {}
    with path.open() as fh:
        raw: dict = json.load(fh)
    result: dict[str, Any] = {}
    for key, value in raw.items():
        if key in DEFAULTS:
            result[key] = value
        else:
            _log.warning("settings.json: unknown key %r ignored", key)
    return result


# ---------------------------------------------------------------------------
# save
# ---------------------------------------------------------------------------

def _validate_value(key: str, value: Any) -> None:
    """Raise ValueError if *value* is not an acceptable type for *key*."""
    if key not in DEFAULTS:
        raise ValueError(
            f"Unknown settings key {key!r}. "
            f"Valid keys: {sorted(DEFAULTS.keys())}"
        )
    default = DEFAULTS[key]

    # publish_target: None or str
    if key == "publish_target":
        if value is not None and not isinstance(value, str):
            raise ValueError(
                f"'publish_target' must be None or str, got {type(value).__name__!r}"
            )
        return

    # Dispatch on the default's type
    if isinstance(default, bool):
        # bool default: accept bool only
        if not isinstance(value, bool):
            raise ValueError(f"{key!r} must be bool, got {type(value).__name__!r}")
    elif isinstance(default, int):
        # int default: accept int but NOT bool
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(
                f"{key!r} must be int (not bool), got {type(value).__name__!r}"
            )
    elif isinstance(default, float):
        # float default: accept int or float (but not bool)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"{key!r} must be int or float, got {type(value).__name__!r}"
            )
    elif isinstance(default, str):
        if not isinstance(value, str):
            raise ValueError(f"{key!r} must be str, got {type(value).__name__!r}")
    elif isinstance(default, list):
        if not isinstance(value, list):
            raise ValueError(f"{key!r} must be list, got {type(value).__name__!r}")


def save(path: Path | str, partial: dict[str, Any]) -> None:
    """Validate *partial*, merge it over current on-disk settings, write JSON.

    Raises
    ------
    ValueError
        If any key is unknown or any value has the wrong type.
    """
    path = Path(path)

    # Validate first — do NOT touch disk if anything is wrong
    for key, value in partial.items():
        _validate_value(key, value)

    # Read the raw on-disk file (only previously-saved keys, NOT defaults),
    # then merge the new partial on top.  This keeps the file lean — it only
    # ever contains keys the user has explicitly set, never defaults bloat.
    current: dict[str, Any] = {}
    if path.exists():
        with path.open() as fh:
            current = json.load(fh)
    current.update(partial)

    path.write_text(json.dumps(current, indent=2))
