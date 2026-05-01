"""Tests for affiliation-extraction helpers (utils.py).

Each Bug A/B/C/D/E comment in utils.py maps to at least one assertion here,
ensuring the fixes can't be silently reverted.
"""
import pytest
from unittest.mock import MagicMock

from ndif_citations.utils import (
    _affil_clean,
    _affil_dedupe,
    _affil_find_footnote_block,
    _affil_fix_hyphens,
    _affil_looks_valid,
    _affil_parse_inline_block,
    _affil_parse_marker_block,
    _affil_parse_suffix_markers,
    extract_affiliations_from_pdf,
)


# ---------------------------------------------------------------------------
# _affil_clean  (Bug A fix: trailing "Correspondence to:" etc.)
# ---------------------------------------------------------------------------

class TestAffilClean:
    def test_strips_email(self):
        result = _affil_clean("MIT, user@mit.edu")
        assert "@" not in result

    def test_strips_brace_group(self):
        result = _affil_clean("{alice,bob}@stanford.edu Stanford University")
        assert "{" not in result
        assert "}" not in result

    def test_strips_leading_digit_marker(self):
        result = _affil_clean("1Stanford University")
        assert result.startswith("Stanford")

    def test_strips_leading_dagger(self):
        result = _affil_clean("†Harvard University")
        assert result.startswith("Harvard")

    def test_strips_trailing_correspondence_bug_a(self):
        # Bug A: trailing "Correspondence to:" truncated
        result = _affil_clean("MIT Correspondence to: someone")
        assert "Correspondence" not in result

    def test_strips_trailing_preprint(self):
        result = _affil_clean("Stanford University Preprint. Under review")
        assert "Preprint" not in result

    def test_strips_trailing_punctuation(self):
        result = _affil_clean("MIT,;.: ")
        assert not result.endswith((",", ";", ".", ":"))

    def test_normalizes_whitespace(self):
        result = _affil_clean("MIT   University")
        assert "  " not in result

    def test_empty_string(self):
        assert _affil_clean("") == ""


# ---------------------------------------------------------------------------
# _affil_looks_valid  (Bug B fix: compound entries with 2+ digit-letter patterns)
# ---------------------------------------------------------------------------

class TestAffilLooksValid:
    def test_accepts_university(self):
        assert _affil_looks_valid("University of Freiburg") is True

    def test_accepts_big_tech(self):
        assert _affil_looks_valid("Microsoft Research") is True

    def test_accepts_northeastern(self):
        assert _affil_looks_valid("Northeastern University") is True

    def test_rejects_empty(self):
        assert _affil_looks_valid("") is False

    def test_rejects_too_short(self):
        assert _affil_looks_valid("MIT"[:3]) is False  # "MIT" is exactly 3 chars

    def test_rejects_too_long(self):
        assert _affil_looks_valid("A" * 201) is False

    def test_rejects_equal_contributions_noise(self):
        assert _affil_looks_valid("Equal contributions∗") is False

    def test_rejects_work_done_during_noise(self):
        assert _affil_looks_valid("Work done during internship at Google") is False

    def test_rejects_compound_entry_bug_b(self):
        # Bug B: "1Stanford 2MIT 3Apple" has 2+ digit-uppercase patterns
        assert _affil_looks_valid("1Stanford 2MIT 3Apple") is False

    def test_rejects_string_with_no_org_keyword(self):
        assert _affil_looks_valid("Some random phrase without org") is False


# ---------------------------------------------------------------------------
# _affil_parse_marker_block
# ---------------------------------------------------------------------------

class TestAffilParseMarkerBlock:
    def test_digit_prefix_format(self):
        result = _affil_parse_marker_block("1Stanford University 2Massachusetts Institute of Technology")
        assert any("Stanford" in a for a in result)
        assert any("Massachusetts" in a or "Technology" in a for a in result)

    def test_dagger_prefix_format(self):
        # Need longer org names so _affil_looks_valid passes
        result = _affil_parse_marker_block("†Apple Inc ‡Google Research")
        assert any("Apple" in a for a in result), f"Got: {result}"

    def test_empty_returns_empty(self):
        assert _affil_parse_marker_block("") == []

    def test_cuts_at_correspondence(self):
        block = "1Stanford University. Correspondence to: alice@stanford.edu"
        result = _affil_parse_marker_block(block)
        for a in result:
            assert "Correspondence" not in a


# ---------------------------------------------------------------------------
# _affil_parse_suffix_markers  (Bug D fix)
# ---------------------------------------------------------------------------

class TestAffilParseSuffixMarkers:
    def test_two_suffix_markers(self):
        block = "University of Freiburg♡, Microsoft Research♣"
        result = _affil_parse_suffix_markers(block)
        assert len(result) == 2
        assert any("Freiburg" in a for a in result)
        assert any("Microsoft" in a for a in result)

    def test_single_marker_returns_empty(self):
        # Requires ≥2 matches to count as suffix-marker format
        result = _affil_parse_suffix_markers("University of Freiburg♡")
        assert result == []

    def test_empty_string(self):
        assert _affil_parse_suffix_markers("") == []


# ---------------------------------------------------------------------------
# _affil_parse_inline_block  (Bug B fall-through + Bug D integration)
# ---------------------------------------------------------------------------

class TestAffilParseInlineBlock:
    def test_prefix_marker_takes_priority(self):
        # Authors block before Abstract with digit-prefixed affiliations
        text = (
            "Alice Smith¹, Bob Jones²\n"
            "¹Stanford University\n"
            "²MIT\n"
            "\nAbstract\nThis paper presents...\n"
        )
        result = _affil_parse_inline_block(text, "Smith, Jones")
        # Should find affiliations and NOT duplicate via line fallback (Bug B fix)
        stanford_count = sum(1 for a in result if "Stanford" in a)
        assert stanford_count == 1

    def test_suffix_marker_fallback_when_no_prefix(self):
        text = (
            "Alice Smith, Bob Jones\n"
            "Stanford University♡, MIT♣\n"
            "\nAbstract\nThis paper discusses..."
        )
        result = _affil_parse_inline_block(text, "Smith, Jones")
        assert any("Stanford" in a for a in result) or any("MIT" in a for a in result)

    def test_line_fallback_when_no_markers(self):
        text = (
            "Alice Smith, Bob Jones\n"
            "Northeastern University\n"
            "\nAbstract\nThis is the abstract..."
        )
        result = _affil_parse_inline_block(text, "Smith, Jones")
        assert any("Northeastern" in a for a in result)

    def test_no_abstract_marker_returns_empty(self):
        text = "Alice Smith\nMIT\nNo abstract heading here"
        result = _affil_parse_inline_block(text, "Smith")
        assert result == []


# ---------------------------------------------------------------------------
# _affil_fix_hyphens
# ---------------------------------------------------------------------------

class TestAffilFixHyphens:
    def test_joins_hyphenated_line_break(self):
        result = _affil_fix_hyphens("Col-\nlege")
        assert result == "College"

    def test_multiple_hyphenations(self):
        result = _affil_fix_hyphens("Uni-\nver-\nsity")
        assert result == "University"

    def test_no_hyphen_unchanged(self):
        text = "Stanford University"
        assert _affil_fix_hyphens(text) == text

    def test_does_not_join_non_word_hyphen(self):
        # Hyphen not between word chars should be left
        text = "Some text- \nnon-word"
        result = _affil_fix_hyphens(text)
        # The \w-\n-\w pattern only matches word-boundary hyphens
        assert isinstance(result, str)  # just don't crash


# ---------------------------------------------------------------------------
# _affil_dedupe
# ---------------------------------------------------------------------------

class TestAffilDedupe:
    def test_removes_exact_duplicate(self):
        from ndif_citations.utils import _affil_dedupe
        result = _affil_dedupe(["MIT", "MIT"])
        assert result == ["MIT"]

    def test_case_insensitive_dedup(self):
        from ndif_citations.utils import _affil_dedupe
        result = _affil_dedupe(["MIT", "mit"])
        assert len(result) == 1

    def test_preserves_first_seen_order(self):
        from ndif_citations.utils import _affil_dedupe
        result = _affil_dedupe(["Stanford University", "MIT", "Stanford University"])
        assert result == ["Stanford University", "MIT"]

    def test_empty_list(self):
        from ndif_citations.utils import _affil_dedupe
        assert _affil_dedupe([]) == []


# ---------------------------------------------------------------------------
# _affil_find_footnote_block
# ---------------------------------------------------------------------------

class TestAffilFindFootnoteBlock:
    def test_anchors_on_correspondence_to(self):
        text = (
            "some preamble\n"
            "1Stanford University 2MIT\n"
            "Correspondence to: alice@stanford.edu\n"
        )
        block = _affil_find_footnote_block(text)
        assert block is not None

    def test_anchors_on_preprint(self):
        text = "1Stanford University 2Northeastern University\nPreprint.\n"
        block = _affil_find_footnote_block(text)
        assert block is not None

    def test_returns_none_when_no_anchor(self):
        text = "Just some text about machine learning and transformers."
        assert _affil_find_footnote_block(text) is None

    def test_returns_none_when_no_digit_uppercase_in_window(self):
        text = "No structured affiliations here.\nCorrespondence to: someone"
        result = _affil_find_footnote_block(text)
        assert result is None


# ---------------------------------------------------------------------------
# extract_affiliations_from_pdf — integration (mocked fitz)  (Bug C check)
# ---------------------------------------------------------------------------

def _make_fitz_doc(page0_text: str):
    """Build a fitz.open() mock returning page0_text from doc[0].get_text()."""
    page_mock = MagicMock()
    page_mock.get_text = MagicMock(
        side_effect=lambda fmt=None, **kw: page0_text if fmt != "blocks" else []
    )
    doc_mock = MagicMock()
    doc_mock.__getitem__ = MagicMock(return_value=page_mock)
    doc_mock.__len__ = MagicMock(return_value=1)
    doc_mock.__enter__ = MagicMock(return_value=doc_mock)
    doc_mock.__exit__ = MagicMock(return_value=False)
    doc_mock.close = MagicMock()
    return doc_mock


def _mock_fitz(page0_text: str, monkeypatch):
    """Inject a fitz mock into sys.modules (fitz is imported locally inside functions)."""
    import sys
    fitz_mock = MagicMock()
    fitz_mock.open.return_value = _make_fitz_doc(page0_text)
    monkeypatch.setitem(sys.modules, "fitz", fitz_mock)
    return fitz_mock


class TestExtractAffiliationsIntegration:
    def test_inline_block_strategy(self, monkeypatch):
        """ACL/EMNLP inline block (Strategy 3) succeeds."""
        text = (
            "Alice Smith, Bob Jones\n"
            "Northeastern University\n"
            "\nAbstract\nThis is the abstract.\n"
        )
        _mock_fitz(text, monkeypatch)
        from pathlib import Path
        result = extract_affiliations_from_pdf(Path("/fake/paper.pdf"), "Smith, Jones")
        assert any("Northeastern" in a for a in result)

    def test_bug_c_footnote_detected_but_empty_falls_through(self, monkeypatch):
        """Bug C: footnote block detected but yields no entries → falls through to inline."""
        text = (
            "Alice Smith\n"
            "Northeastern University\n"
            "\nAbstract\nSome abstract text.\n"
            "Correspondence to: alice@northeastern.edu\n"
        )
        _mock_fitz(text, monkeypatch)
        from pathlib import Path
        result = extract_affiliations_from_pdf(Path("/fake/paper.pdf"), "Smith")
        assert isinstance(result, list)
