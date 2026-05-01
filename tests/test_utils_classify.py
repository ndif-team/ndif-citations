"""Unit tests for classification pre-filter helper functions (US-Q1, US-Q2, US-Q3)."""

import pytest

from ndif_citations.process import (
    _context_is_comparison_table,
    _has_negative_evidence,
    _is_ack_only_thank_you,
)


# ---------------------------------------------------------------------------
# US-Q1: _has_negative_evidence
# ---------------------------------------------------------------------------

class TestHasNegativeEvidence:
    def test_removing_dependency_on_nnsight(self):
        assert _has_negative_evidence(
            "Removing the dependency on nnsight (only used for MLP activations) "
            "and replacing it with vanilla torch hooks."
        )

    def test_removed_nnsight(self):
        assert _has_negative_evidence("We removed the nnsight dependency from our codebase.")

    def test_rather_than_nnsight(self):
        assert _has_negative_evidence(
            "We used standard PyTorch hooks rather than nnsight for speed."
        )

    def test_instead_of_nnsight(self):
        assert _has_negative_evidence("We implemented hooks instead of nnsight.")

    def test_without_using_nnsight(self):
        assert _has_negative_evidence("Our system operates without using nnsight.")

    def test_no_longer_uses_nnsight(self):
        assert _has_negative_evidence("The final version no longer uses nnsight.")

    def test_alternative_to_ndif(self):
        assert _has_negative_evidence("TDHook is an alternative to NDIF for distributed inference.")

    def test_compared_to_nnsight(self):
        assert _has_negative_evidence("Our approach compared to alternatives like nnsight.")

    def test_case_insensitive(self):
        assert _has_negative_evidence("REMOVING THE DEPENDENCY ON NNSIGHT.")

    def test_positive_usage_no_match(self):
        assert not _has_negative_evidence(
            "We use nnsight to implement our activation patching experiments."
        )

    def test_citation_mention_no_match(self):
        assert not _has_negative_evidence("NNsight [Smith et al., 2024] introduced this API.")

    def test_empty_string(self):
        assert not _has_negative_evidence("")


# ---------------------------------------------------------------------------
# US-Q2: _context_is_comparison_table
# ---------------------------------------------------------------------------

class TestContextIsComparisonTable:
    def test_three_checkmarks_detected(self):
        assert _context_is_comparison_table(
            "captum ✓ ✗ ✗ ∼ 35+ nnsight ✓ ✓ ✓ ✗ 0"
        )

    def test_only_checkmark_unicode(self):
        assert _context_is_comparison_table("✓ ✓ ✓")

    def test_mixed_table_chars(self):
        assert _context_is_comparison_table("✓ ✗ ∼ other text here")

    def test_two_checkmarks_not_enough(self):
        assert not _context_is_comparison_table("Framework ✓ ✗ description")

    def test_one_checkmark_not_enough(self):
        assert not _context_is_comparison_table("We verified ✓ the results.")

    def test_plain_prose_no_match(self):
        assert not _context_is_comparison_table(
            "We use nnsight to implement our activation patching experiments."
        )

    def test_pipe_table_not_matched(self):
        # Pipe-style tables should NOT trigger (pipe excluded by design)
        assert not _context_is_comparison_table(
            "| Framework | Supported | License |\n"
            "| NNsight | Yes | MIT |"
        )

    def test_empty_string(self):
        assert not _context_is_comparison_table("")


# ---------------------------------------------------------------------------
# US-Q3: _is_ack_only_thank_you
# ---------------------------------------------------------------------------

class TestIsAckOnlyThankYou:
    def test_nsf_ndif_platform_phrase(self):
        assert _is_ack_only_thank_you(
            "We thank NSF NDIF for providing a platform for reproducible experiments."
        )

    def test_thank_ndif_compute_resources(self):
        assert _is_ack_only_thank_you(
            "We thank NDIF for providing compute resources for our preliminary experiments."
        )

    def test_acknowledge_nnsight_team(self):
        assert _is_ack_only_thank_you(
            "We acknowledge the nnsight team for their open-source contribution."
        )

    def test_supported_by_ndif(self):
        assert _is_ack_only_thank_you(
            "This work was supported by NDIF infrastructure grants."
        )

    def test_pattern_a_impl_confirm_blocks_filter(self):
        # "conducted using NDIF" → Pattern A fires → not filtered
        assert not _is_ack_only_thank_you(
            "We thank NDIF. All experiments were conducted using NDIF cluster."
        )

    def test_pattern_b_keyword_first_blocks_filter(self):
        # "NNsight ... is the package we used" → Pattern B fires → not filtered
        assert not _is_ack_only_thank_you(
            "We thank the team working on NNsight which is the python package we used "
            "to implement all our experiments."
        )

    def test_pattern_c_to_implement_blocks_filter(self):
        # "NNsight ... to implement" → Pattern C fires → not filtered
        assert not _is_ack_only_thank_you(
            "We thank NDIF and used NNsight to implement our interventions."
        )

    def test_cross_sentence_guard(self):
        # Period between thank and impl-confirm → _IMPL_CONFIRM_RE shouldn't bridge it
        # "We thank NDIF." ends the sentence. Next sentence uses TransformerLens, not NDIF.
        assert _is_ack_only_thank_you(
            "We thank NDIF. Our experiments use TransformerLens."
        )

    def test_pattern_a_impl_uses_nnsight(self):
        # "using nnsight" → Pattern A match → not filtered
        assert not _is_ack_only_thank_you(
            "We thank the NDIF team. Experiments are implemented using nnsight."
        )

    def test_no_ack_keyword_no_match(self):
        # No acknowledgment word present → not an ack window
        assert not _is_ack_only_thank_you(
            "We use nnsight to extract activations from GPT-2."
        )

    def test_empty_string(self):
        assert not _is_ack_only_thank_you("")
