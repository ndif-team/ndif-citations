"""TDD tests for settings_store.py and config.reload_settings()."""
from __future__ import annotations

import json
import pytest

from ndif_citations import settings_store, config


# ---------------------------------------------------------------------------
# Snapshot/restore fixture — prevents config mutation from leaking into the
# 658-test suite.  Applied automatically to every test in this module.
# ---------------------------------------------------------------------------

_TRACKED_ATTRS = [
    "MIN_PAPER_YEAR",
    "LLM_RATE_LIMIT_SLEEP",
    "S2_RATE_LIMIT_SLEEP",
    "GITHUB_RATE_LIMIT_SLEEP",
    "EXCLUDED_GITHUB_REPOS",
    "KNOWN_COURSE_SOURCES",
    "COURSE_NAME_PATTERNS",
    "NDIF_KEYWORDS",
    "NDIF_README_KEYWORDS_REGEX",
    "NDIF_README_KEYWORDS_SUBSTR",
    "NDIF_README_NEGATIVE_PATTERNS",
    "LLM_MODEL",
    "LLM_BASE_URL",
    "SHARED_PAPER_THRESHOLD",
    "LLM_API_KEY",
    "S2_API_KEY",
    "GITHUB_TOKEN",
    "SERPAPI_API_KEY",
    "OPENALEX_EMAIL",
    "UNPAYWALL_EMAIL",
]


@pytest.fixture(autouse=True)
def restore_config():
    """Snapshot and restore config module globals after every test."""
    snapshot = {attr: getattr(config, attr, None) for attr in _TRACKED_ATTRS}
    # Also snapshot _SETTINGS_FILE so monkeypatches get cleaned up properly
    original_settings_file = getattr(config, "_SETTINGS_FILE", None)
    yield
    for attr, val in snapshot.items():
        setattr(config, attr, val)
    if original_settings_file is not None:
        config._SETTINGS_FILE = original_settings_file


# ---------------------------------------------------------------------------
# settings_store.load()
# ---------------------------------------------------------------------------


def test_load_defaults_when_missing(tmp_path):
    """load() returns a deep copy of DEFAULTS when path doesn't exist."""
    s = settings_store.load(tmp_path / "settings.json")
    assert s["min_paper_year"] == 2024
    assert "ndif-team/nnsight" in s["excluded_github_repos"]


def test_load_returns_deep_copy_of_defaults(tmp_path):
    """Mutating the returned dict must not affect DEFAULTS."""
    s = settings_store.load(tmp_path / "settings.json")
    s["ndif_keywords"].append("INJECTED")
    assert "INJECTED" not in settings_store.DEFAULTS["ndif_keywords"]


def test_load_merges_partial_override(tmp_path):
    """load() merges JSON keys over defaults; unset keys keep default values."""
    f = tmp_path / "settings.json"
    f.write_text(json.dumps({"min_paper_year": 2025}))
    s = settings_store.load(f)
    assert s["min_paper_year"] == 2025
    # Unaffected keys still hold defaults
    assert s["llm_rate_limit_sleep"] == settings_store.DEFAULTS["llm_rate_limit_sleep"]


def test_load_full_file(tmp_path):
    """load() handles a fully populated settings file."""
    f = tmp_path / "settings.json"
    data = {
        "min_paper_year": 2023,
        "llm_rate_limit_sleep": 5.0,
        "excluded_github_repos": ["a/b"],
        "known_course_sources": ["x/y"],
    }
    f.write_text(json.dumps(data))
    s = settings_store.load(f)
    assert s["min_paper_year"] == 2023
    assert s["excluded_github_repos"] == ["a/b"]
    assert s["known_course_sources"] == ["x/y"]


# ---------------------------------------------------------------------------
# settings_store.save() + config.reload_settings()
# ---------------------------------------------------------------------------


def test_save_and_reload_applies_to_config(tmp_path, monkeypatch):
    """save() + reload_settings() mutates config module attributes."""
    f = tmp_path / "settings.json"
    settings_store.save(f, {"min_paper_year": 2025, "llm_rate_limit_sleep": 2.0})
    monkeypatch.setattr(config, "_SETTINGS_FILE", f, raising=False)
    config.reload_settings()
    assert config.MIN_PAPER_YEAR == 2025
    assert config.LLM_RATE_LIMIT_SLEEP == 2.0


def test_set_typed_knobs_round_trip(tmp_path, monkeypatch):
    """set-typed config attrs (EXCLUDED_GITHUB_REPOS) stay set after round-trip."""
    f = tmp_path / "settings.json"
    # First write some integer-keyed settings, then extend with list
    settings_store.save(f, {"min_paper_year": 2025, "llm_rate_limit_sleep": 2.0})
    settings_store.save(f, {"excluded_github_repos": ["a/b", "c/d"]})
    monkeypatch.setattr(config, "_SETTINGS_FILE", f, raising=False)
    config.reload_settings()
    assert config.EXCLUDED_GITHUB_REPOS == {"a/b", "c/d"}
    assert isinstance(config.EXCLUDED_GITHUB_REPOS, set)


def test_reload_also_updates_known_course_sources_as_set(tmp_path, monkeypatch):
    """KNOWN_COURSE_SOURCES is also kept as set after reload."""
    f = tmp_path / "settings.json"
    settings_store.save(f, {"known_course_sources": ["foo/bar", "baz/qux"]})
    monkeypatch.setattr(config, "_SETTINGS_FILE", f, raising=False)
    config.reload_settings()
    assert config.KNOWN_COURSE_SOURCES == {"foo/bar", "baz/qux"}
    assert isinstance(config.KNOWN_COURSE_SOURCES, set)


# ---------------------------------------------------------------------------
# settings_store.save() — validation
# ---------------------------------------------------------------------------


def test_validate_rejects_bad_types(tmp_path):
    """save() raises ValueError for wrong type (str instead of int)."""
    with pytest.raises(ValueError):
        settings_store.save(tmp_path / "s.json", {"min_paper_year": "two thousand"})


def test_validate_rejects_unknown_key(tmp_path):
    """save() raises ValueError for unrecognized setting keys."""
    with pytest.raises(ValueError):
        settings_store.save(tmp_path / "s.json", {"nonexistent_key": 42})


def test_validate_rejects_bool_for_int(tmp_path):
    """save() rejects bool for an int-typed key (bool is subclass of int)."""
    with pytest.raises(ValueError):
        settings_store.save(tmp_path / "s.json", {"min_paper_year": True})


def test_validate_accepts_int_for_float(tmp_path):
    """save() accepts int value for float-typed keys."""
    f = tmp_path / "s.json"
    settings_store.save(f, {"llm_rate_limit_sleep": 5})  # int is fine for float
    s = settings_store.load(f)
    assert s["llm_rate_limit_sleep"] == 5


def test_validate_rejects_list_for_string(tmp_path):
    """save() rejects a list for a str-typed key."""
    with pytest.raises(ValueError):
        settings_store.save(tmp_path / "s.json", {"llm_model": ["bad"]})


def test_validate_accepts_none_for_publish_target(tmp_path):
    """publish_target accepts None (its default type)."""
    f = tmp_path / "s.json"
    settings_store.save(f, {"publish_target": None})
    s = settings_store.load(f)
    assert s["publish_target"] is None


def test_validate_accepts_str_for_publish_target(tmp_path):
    """publish_target accepts a str value."""
    f = tmp_path / "s.json"
    settings_store.save(f, {"publish_target": "s3://bucket/key"})
    s = settings_store.load(f)
    assert s["publish_target"] == "s3://bucket/key"


def test_validate_rejects_int_for_publish_target(tmp_path):
    """publish_target rejects non-str, non-None values."""
    with pytest.raises(ValueError):
        settings_store.save(tmp_path / "s.json", {"publish_target": 42})


# ---------------------------------------------------------------------------
# save() merge behaviour
# ---------------------------------------------------------------------------


def test_save_merges_over_existing(tmp_path):
    """Successive save() calls merge — earlier keys not in partial are kept."""
    f = tmp_path / "settings.json"
    settings_store.save(f, {"min_paper_year": 2025})
    settings_store.save(f, {"llm_rate_limit_sleep": 9.9})
    s = settings_store.load(f)
    assert s["min_paper_year"] == 2025  # from first save
    assert s["llm_rate_limit_sleep"] == 9.9  # from second save


# ---------------------------------------------------------------------------
# No-settings-file → behaviour identical to today
# ---------------------------------------------------------------------------


def test_import_time_apply_is_noop_when_no_settings_file(tmp_path):
    """When settings.json is absent, config values equal the hardcoded defaults."""
    # Use a guaranteed-missing path; reload to make sure
    monkeypatched_path = tmp_path / "no_such_settings.json"
    original = config._SETTINGS_FILE
    config._SETTINGS_FILE = monkeypatched_path
    try:
        config.reload_settings()
        assert config.MIN_PAPER_YEAR == 2024
        assert config.LLM_RATE_LIMIT_SLEEP == 12.0
        assert config.S2_RATE_LIMIT_SLEEP == 3.0
        assert config.GITHUB_RATE_LIMIT_SLEEP == 2.0
    finally:
        config._SETTINGS_FILE = original


# ---------------------------------------------------------------------------
# Regression test: env-derived LLM_MODEL survives absent settings.json (bug fix)
# ---------------------------------------------------------------------------


def test_env_derived_llm_model_survives_missing_settings_file(tmp_path, monkeypatch):
    """reload_settings() must NOT clobber LLM_MODEL when settings.json is absent.

    This is the regression test for the bug where load() returned all DEFAULTS
    (including llm_model) even when settings.json was absent, causing _apply_settings
    to silently overwrite an env-derived LLM_MODEL back to the static default.
    """
    # Simulate a value derived from the environment (e.g. LLM_MODEL=gpt-4o in .env)
    config.LLM_MODEL = "custom/model-xyz"

    # Point _SETTINGS_FILE at a nonexistent path so settings.json is absent
    monkeypatch.setattr(config, "_SETTINGS_FILE", tmp_path / "no_such_settings.json")

    # Prevent load_dotenv from re-reading a real .env that might contain LLM_MODEL
    monkeypatch.setattr("ndif_citations.config.load_dotenv", lambda *a, **kw: None)

    config.reload_settings()

    # The pre-set (env-derived) value must be preserved — not clobbered by the default
    assert config.LLM_MODEL == "custom/model-xyz", (
        f"Expected 'custom/model-xyz', got {config.LLM_MODEL!r}. "
        "reload_settings() must not overwrite env-derived LLM_MODEL when settings.json is absent."
    )


# ---------------------------------------------------------------------------
# load_overrides() tests
# ---------------------------------------------------------------------------


def test_load_overrides_returns_empty_when_file_missing(tmp_path):
    """load_overrides() returns {} when path does not exist."""
    result = settings_store.load_overrides(tmp_path / "nonexistent.json")
    assert result == {}


def test_load_overrides_returns_only_present_key(tmp_path):
    """load_overrides() returns only keys present in the file, not all DEFAULTS."""
    f = tmp_path / "settings.json"
    f.write_text(json.dumps({"min_paper_year": 2025}))
    result = settings_store.load_overrides(f)
    assert result == {"min_paper_year": 2025}
    # No other DEFAULTS keys should appear
    assert "llm_model" not in result
    assert "llm_rate_limit_sleep" not in result
    assert len(result) == 1


def test_load_overrides_ignores_unknown_keys(tmp_path):
    """load_overrides() silently drops keys not in DEFAULTS."""
    f = tmp_path / "settings.json"
    f.write_text(json.dumps({"min_paper_year": 2025, "totally_unknown_key": "foo"}))
    result = settings_store.load_overrides(f)
    assert "totally_unknown_key" not in result
    assert result == {"min_paper_year": 2025}


def test_load_overrides_empty_file_returns_empty(tmp_path):
    """load_overrides() returns {} for an empty JSON object on disk."""
    f = tmp_path / "settings.json"
    f.write_text("{}")
    result = settings_store.load_overrides(f)
    assert result == {}


# ---------------------------------------------------------------------------
# save() no-bloat test — file must contain ONLY explicitly-set keys
# ---------------------------------------------------------------------------


def test_save_no_bloat(tmp_path):
    """save() must write ONLY the explicitly-set keys, not all 15+ DEFAULTS keys."""
    f = tmp_path / "settings.json"
    settings_store.save(f, {"min_paper_year": 2025})
    raw = json.loads(f.read_text())
    assert set(raw.keys()) == {"min_paper_year"}, (
        f"Expected only {{'min_paper_year'}}, got {set(raw.keys())}. "
        "save() must not bloat the file with all DEFAULTS keys."
    )
    assert raw["min_paper_year"] == 2025


def test_save_accumulates_keys_without_bloat(tmp_path):
    """Successive save() calls accumulate only explicitly-set keys."""
    f = tmp_path / "settings.json"
    settings_store.save(f, {"min_paper_year": 2025})
    settings_store.save(f, {"llm_rate_limit_sleep": 5.0})
    raw = json.loads(f.read_text())
    assert set(raw.keys()) == {"min_paper_year", "llm_rate_limit_sleep"}
    assert raw["min_paper_year"] == 2025
    assert raw["llm_rate_limit_sleep"] == 5.0


# ---------------------------------------------------------------------------
# DRY sync-guard: DEFAULTS must match config hardcoded values
# ---------------------------------------------------------------------------


def test_defaults_sync_with_config_min_paper_year():
    """DEFAULTS['min_paper_year'] must equal config.MIN_PAPER_YEAR's hardcoded value."""
    assert settings_store.DEFAULTS["min_paper_year"] == 2024


def test_defaults_sync_with_config_excluded_github_repos():
    """DEFAULTS excluded_github_repos must match config.EXCLUDED_GITHUB_REPOS."""
    assert set(settings_store.DEFAULTS["excluded_github_repos"]) == {"ndif-team/nnsight"}


def test_defaults_sync_with_config_course_name_patterns():
    """DEFAULTS course_name_patterns must equal config.COURSE_NAME_PATTERNS."""
    assert settings_store.DEFAULTS["course_name_patterns"] == config.COURSE_NAME_PATTERNS


def test_defaults_sync_with_config_llm_model():
    """DEFAULTS llm_model must equal the config hardcoded LLM_MODEL default."""
    assert settings_store.DEFAULTS["llm_model"] == "meta/llama-3.1-70b-instruct"


def test_defaults_sync_with_config_llm_base_url():
    """DEFAULTS llm_base_url must equal the config hardcoded LLM_BASE_URL default."""
    assert settings_store.DEFAULTS["llm_base_url"] == "https://integrate.api.nvidia.com/v1"


def test_defaults_sync_with_config_shared_paper_threshold():
    """DEFAULTS shared_paper_threshold must equal config.SHARED_PAPER_THRESHOLD."""
    assert settings_store.DEFAULTS["shared_paper_threshold"] == config.SHARED_PAPER_THRESHOLD
