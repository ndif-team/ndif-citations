"""Tests for _select_classification_prompt (US-D7)."""
import pytest

from ndif_citations.process import (
    INFRASTRUCTURE_PROMPT,
    LIBRARY_PROMPT,
    UNIFIED_PROMPT,
    _select_classification_prompt,
)


class TestSelectClassificationPrompt:
    def test_library_only_context_returns_library_prompt(self):
        context = "import nnsight; model = nnsight.LanguageModel('gpt2')"
        assert _select_classification_prompt(context) is LIBRARY_PROMPT

    def test_infrastructure_only_context_returns_infra_prompt(self):
        context = "We ran all experiments on the NDIF cluster using hosted models."
        assert _select_classification_prompt(context) is INFRASTRUCTURE_PROMPT

    def test_mixed_context_returns_unified_prompt(self):
        context = "We use nnsight on the NDIF cluster for all experiments."
        assert _select_classification_prompt(context) is UNIFIED_PROMPT

    def test_empty_context_returns_unified_prompt(self):
        assert _select_classification_prompt("") is UNIFIED_PROMPT

    def test_no_direct_mentions_returns_unified_prompt(self):
        context = "No direct mentions of NDIF or nnsight found in the paper text."
        assert _select_classification_prompt(context) is UNIFIED_PROMPT

    def test_abstract_only_no_keywords_returns_unified(self):
        context = "We study how transformers learn attention patterns."
        assert _select_classification_prompt(context) is UNIFIED_PROMPT

    def test_library_prompt_does_not_contain_uses_ndif(self):
        assert "uses_ndif" not in LIBRARY_PROMPT

    def test_infra_prompt_does_not_contain_uses_nnsight(self):
        assert "uses_nnsight" not in INFRASTRUCTURE_PROMPT

    def test_unified_prompt_contains_all_four_replies(self):
        for reply in ("uses_ndif", "uses_nnsight", "referencing", "unclassified"):
            assert reply in UNIFIED_PROMPT

    def test_nnsight_net_url_triggers_library_prompt(self):
        context = "For our experiments we used the nnsight.net API."
        assert _select_classification_prompt(context) is LIBRARY_PROMPT

    def test_ndif_us_url_triggers_infra_prompt(self):
        context = "Inference was hosted at ndif.us for this study."
        assert _select_classification_prompt(context) is INFRASTRUCTURE_PROMPT
