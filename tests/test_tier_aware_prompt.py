"""Tests for _augment_prompt_with_tier and classify_category tier integration (US-T4)."""
import pytest

from ndif_citations.process import (
    INFRASTRUCTURE_PROMPT,
    LIBRARY_PROMPT,
    UNIFIED_PROMPT,
    _augment_prompt_with_tier,
)


class TestAugmentPromptWithTier:
    def test_tier1_appends_bibtex_block(self):
        augmented = _augment_prompt_with_tier(LIBRARY_PROMPT, 1)
        assert "TIER 1 CROSS-LINK EVIDENCE" in augmented
        assert "BibTeX block in README" in augmented
        assert augmented != LIBRARY_PROMPT

    def test_tier2_appends_citation_section_block(self):
        augmented = _augment_prompt_with_tier(INFRASTRUCTURE_PROMPT, 2)
        assert "TIER 2 CROSS-LINK EVIDENCE" in augmented
        assert "Citation/Paper section heading" in augmented
        assert augmented != INFRASTRUCTURE_PROMPT

    def test_tier3_returns_prompt_unchanged(self):
        augmented = _augment_prompt_with_tier(UNIFIED_PROMPT, 3)
        assert augmented == UNIFIED_PROMPT

    def test_tier4_returns_prompt_unchanged(self):
        augmented = _augment_prompt_with_tier(LIBRARY_PROMPT, 4)
        assert augmented == LIBRARY_PROMPT

    def test_none_tier_returns_prompt_unchanged(self):
        augmented = _augment_prompt_with_tier(UNIFIED_PROMPT, None)
        assert augmented == UNIFIED_PROMPT

    def test_augmentation_composes_with_library_prompt(self):
        augmented = _augment_prompt_with_tier(LIBRARY_PROMPT, 1)
        # Must preserve the original library prompt content AND add tier block
        assert "uses_nnsight" in augmented
        assert "BibTeX block" in augmented
        # The library prompt base must still be at the start
        assert augmented.startswith(LIBRARY_PROMPT)

    def test_augmentation_composes_with_infra_prompt(self):
        augmented = _augment_prompt_with_tier(INFRASTRUCTURE_PROMPT, 2)
        assert "uses_ndif" in augmented
        assert "Citation/Paper section heading" in augmented


class TestClassifyCategoryTierIntegration:
    def test_tier1_cross_link_sends_augmented_system_prompt(self, monkeypatch, tmp_path):
        from pathlib import Path
        from ndif_citations.models import Category
        from ndif_citations.process import classify_category
        from tests.conftest import make_paper
        from tests.helpers.llm import MockLLMClient

        mock = MockLLMClient()
        mock.expect("uses_nnsight")
        monkeypatch.setattr("ndif_citations.process._get_llm_client", lambda: mock)

        context = "import nnsight; model = nnsight.LanguageModel('gpt2')"
        pdf = tmp_path / "paper.pdf"
        pdf.write_bytes(b"%PDF-1.4 fake")
        monkeypatch.setattr(
            "ndif_citations.process.extract_ndif_context",
            lambda path, window=500: context,
        )

        paper = make_paper(abstract="We use nnsight.")
        paper.linked_paper_tier = 1

        cat, _conf, _band = classify_category(paper, Path("/tmp"), pdf_path=pdf)
        assert cat == Category.USES_NNSIGHT

        calls = mock.record_calls()
        assert len(calls) == 1
        system_msg = calls[0][0]["content"]  # first message is the system message
        assert "TIER 1 CROSS-LINK" in system_msg
        assert "BibTeX block" in system_msg
