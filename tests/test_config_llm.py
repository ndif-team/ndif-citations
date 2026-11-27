"""LLM env parsing (spec §4.1): empty env = unset, {} = no extra body, kill switch."""
import pytest

from ndif_citations import config, settings_store

NVIDIA_THINKING_OFF = {"chat_template_kwargs": {"enable_thinking": False}}


def test_env_or_treats_empty_as_unset(monkeypatch):
    monkeypatch.setenv("NDIF_TEST_X", "")
    assert config._env_or("NDIF_TEST_X", "fallback") == "fallback"
    monkeypatch.setenv("NDIF_TEST_X", "set")
    assert config._env_or("NDIF_TEST_X", "fallback") == "set"


def test_extra_body_defaults_to_nvidia_thinking_off():
    assert config._parse_extra_body(None) == NVIDIA_THINKING_OFF
    assert config._parse_extra_body("") == NVIDIA_THINKING_OFF


def test_extra_body_empty_object_means_no_extra_body():
    assert config._parse_extra_body("{}") == {}


def test_extra_body_custom_object():
    assert config._parse_extra_body('{"reasoning": {"enabled": false}}') == {"reasoning": {"enabled": False}}


def test_extra_body_rejects_non_object():
    with pytest.raises(ValueError):
        config._parse_extra_body("[1, 2]")


def test_default_is_a_fresh_copy():
    a = config._parse_extra_body(None)
    a["chat_template_kwargs"]["enable_thinking"] = True
    assert config._parse_extra_body(None) == NVIDIA_THINKING_OFF


@pytest.mark.parametrize("raw,expected", [
    (None, True), ("", True), ("1", True), ("true", True),
    ("0", False), ("false", False), ("No", False), ("off", False),
])
def test_parse_bool_env(raw, expected):
    assert config._parse_bool_env(raw, True) is expected


def test_live_model_default():
    assert config.DEFAULT_LLM_MODEL == "nvidia/nemotron-3-super-120b-a12b"
    assert settings_store.DEFAULTS["llm_model"] == config.DEFAULT_LLM_MODEL


def test_settings_store_accepts_dict_extra_body(tmp_path):
    path = tmp_path / "settings.json"
    settings_store.save(path, {"llm_extra_body": {}})
    assert settings_store.load_overrides(path) == {"llm_extra_body": {}}


def test_settings_store_rejects_non_dict_extra_body(tmp_path):
    with pytest.raises(ValueError):
        settings_store.save(tmp_path / "settings.json", {"llm_extra_body": "nope"})


_TUNABLES = ("LLM_TIMEOUT", "LLM_MAX_RETRIES", "LLM_RATE_LIMIT_SLEEP")


@pytest.fixture
def reload_config(monkeypatch):
    import importlib
    for k in _TUNABLES:
        monkeypatch.delenv(k, raising=False)
    yield lambda: importlib.reload(config)
    monkeypatch.undo()
    importlib.reload(config)


def test_llm_tunables_from_env(monkeypatch, reload_config):
    monkeypatch.setenv("LLM_TIMEOUT", "45")
    monkeypatch.setenv("LLM_MAX_RETRIES", "0")
    monkeypatch.setenv("LLM_RATE_LIMIT_SLEEP", "3")
    c = reload_config()
    assert (c.LLM_TIMEOUT, c.LLM_MAX_RETRIES, c.LLM_RATE_LIMIT_SLEEP) == (45.0, 0, 3.0)


def test_llm_tunables_empty_env_uses_defaults(monkeypatch, reload_config):
    for k in _TUNABLES:
        monkeypatch.setenv(k, "")
    c = reload_config()
    assert (c.LLM_TIMEOUT, c.LLM_MAX_RETRIES, c.LLM_RATE_LIMIT_SLEEP) == (120.0, 1, 12.0)


def test_zero_pacing_does_not_sleep(monkeypatch):
    from ndif_citations import utils
    calls = []
    monkeypatch.setattr(utils.time, "sleep", calls.append)
    utils.rate_limit_sleep(0.0, "LLM summary")
    assert calls == []
    utils.rate_limit_sleep(12.0, "LLM summary")
    assert calls == [12.0]
