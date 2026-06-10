"""Tests for ndif_citations.utils helpers."""


def test_fuzzy_title_match_subtitle_unrelated_and_empty():
    from ndif_citations.utils import fuzzy_title_match
    full = "Not Just a Piece of Cake: Cross-Lingual Fine-Tuning for Idiom Identification"
    # subtitle/prefix of an existing title should match (token_set_ratio = 88)
    assert fuzzy_title_match("Not Just a Piece of Cake", full) is True
    # unrelated title must not match
    assert fuzzy_title_match("Totally Unrelated Title", full) is False
    # empty/None guards
    assert fuzzy_title_match("", full) is False
    assert fuzzy_title_match(full, "") is False
