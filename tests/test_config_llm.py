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
