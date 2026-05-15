"""Unit tests for src/ndif_citations/venue.py — venue resolution + normalization."""
from __future__ import annotations

import pytest

from ndif_citations.venue import (
    decode_doi_prefix,
    has_known_venue_token,
    is_preprint_sentinel,
    normalize_venue,
    resolve_venue,
    _parse_arxiv_comment,
)
from tests.conftest import make_paper


# ---------------------------------------------------------------------------
# decode_doi_prefix
# ---------------------------------------------------------------------------

class TestDoiPrefixDecode:
    @pytest.mark.parametrize("doi,expected", [
        ("10.18653/v1/2025.acl-long.1536",        "ACL 2025"),
        ("10.18653/v1/2024.naacl-long.16",        "NAACL 2024"),
        ("10.18653/v1/2024.emnlp-main.54",        "EMNLP 2024"),
        ("10.18653/v1/2026.eacl-long.225",        "EACL 2026"),
        ("10.18653/v1/2026.eacl-demo.16",         "EACL 2026"),
        ("10.18653/v1/2025.findings-emnlp.123",   "Findings of EMNLP 2025"),
        ("10.18653/v1/2024.findings-acl.99",      "Findings of ACL 2024"),
        ("10.18653/v1/2025.blackboxnlp-1.14",     "BlackboxNLP 2025"),
        ("10.18653/v1/2024.blackboxnlp-1",        "BlackboxNLP 2024"),
        ("10.1145/3715275.3732201",               "FAccT 2025"),
        ("10.48550/arXiv.2502.00873",             "ArXiv"),
        ("10.48550/arxiv.2406.00877",             "ArXiv"),
        ("https://doi.org/10.18653/v1/2025.acl-long.1", "ACL 2025"),
        # Non-matching
        ("10.1109/TPAMI.2024.12345",              ""),
        ("10.1038/s41592-025-02836-7",            ""),
        ("",                                      ""),
    ])
    def test_doi_decodes_correctly(self, doi, expected):
        assert decode_doi_prefix(doi) == expected


# ---------------------------------------------------------------------------
# is_preprint_sentinel
# ---------------------------------------------------------------------------

class TestPreprintSentinel:
    @pytest.mark.parametrize("venue,expected", [
        ("ArXiv",                                  True),
        ("arxiv 2025",                             True),
        ("ArXiv 2026",                             True),
        ("arXiv preprint arXiv … 2026",            True),
        ("arXiv e-prints 2025",                    True),
        ("arXiv e-print",                          True),
        ("CoRR 2026",                              True),
        ("CoRR",                                   True),
        ("BiorXiv",                                True),
        ("SSRN 6486258",                           True),
        ("Available at SSRN 6486258 2026",         True),
        ("UvA-DARE (University of Amsterdam) 2025", True),
        ("openreview.net",                         True),
        ("https://arxiv.org/abs/2501.12345",       True),
        ("ArXiv.org",                              True),
        ("arXiv.org 2025",                         True),
        ("arxiv.org",                              True),
        ("",                                       True),
        ("   ",                                    True),
        # Truncation residue (year-only, dangling-preposition tails, bare shells)
        ("2025",                                   True),
        ("of the Extended Abstracts of the 2026",  True),
        ("International Conference on 2024",       True),
        ("Conference on 2025",                     True),
        ("URL https://arxiv. org/abs 2024",        True),
        # Mangled normalize output (must be detected so cascade falls through)
        ("39th Conference on NeurIPS (NeurIPS 2025", True),  # unbalanced parens
        ("Foo (Bar 2024",                          True),  # unbalanced parens
        ("39th Conference on NeurIPS 2025",        True),  # leading ordinal
        ("19th 2026",                              True),
        ("14th 2025",                              True),
        # Real venues with prepositions in the middle — NOT sentinels
        ("ICML 2025",                              False),
        ("NeurIPS 2024",                           False),
        ("Nature Methods 2024",                    False),
        ("BlackboxNLP 2025",                       False),
        ("Workshop on Scientific Methods 2024",    False),
        ("the mechanistic interpretability workshop of NeurIPS 2025", False),
    ])
    def test_sentinel_detection(self, venue, expected):
        assert is_preprint_sentinel(venue) is expected


# ---------------------------------------------------------------------------
# has_known_venue_token
# ---------------------------------------------------------------------------

class TestKnownVenueToken:
    @pytest.mark.parametrize("venue,expected", [
        # Conference acronyms (word-boundary match)
        ("ICLR 2025",                              True),
        ("NeurIPS 2024",                           True),
        ("EMNLP 2024",                             True),
        ("FAccT 2025",                             True),
        ("BlackboxNLP 2024",                       True),
        ("Findings of EMNLP 2025",                 True),
        # Acronym embedded in workshop string
        ("Workshop at NeurIPS 2025",               True),
        ("ICLR 2026 Workshop on LLM Reasoning",    True),
        ("CogInterp @ NeurIPS 2025",               True),
        # Acronym with hyphen
        ("ACL-SRW 2025",                           True),
        # Journals (substring match)
        ("Nature Methods 2024",                    True),
        ("Computational Linguistics 2024",         True),
        ("AI 2026",                                True),  # AI in journals list
        # Untruncated unknowns — NOT known
        ("Advances in Neural 2025",                False),  # truncated
        ("Handbook of Human 2025",                 False),
        ("Open Learning: The Journal of Open 2025", False),
        ("Conference on 2025",                     False),  # truncated shell
        ("ICASSP- IEEE 2026",                      False),  # ICASSP not in known list
        ("",                                       False),
    ])
    def test_known_token_detection(self, venue, expected):
        assert has_known_venue_token(venue) is expected


# ---------------------------------------------------------------------------
# normalize_venue
# ---------------------------------------------------------------------------

class TestNormalizeVenue:
    """Each input is from the actual mess survey of research-papers-full.json."""

    @pytest.mark.parametrize("raw,year,expected", [
        # Long-form → acronym
        ("International Conference on Machine Learning 2024", 2024, "ICML 2024"),
        ("International Conference on Machine Learning",      2025, "ICML 2025"),
        ("Neural Information Processing Systems 2024",        2024, "NeurIPS 2024"),
        ("North American Chapter of the Association for Computational Linguistics 2024",
                                                              2024, "NAACL 2024"),
        ("Conference of the European Chapter of the Association for Computational Linguistics 2025",
                                                              2025, "EACL 2025"),
        ("International Conference on Learning Representations 2024", 2024, "ICLR 2024"),
        ("Annual Meeting of the Association for Computational Linguistics 2025",
                                                              2025, "ACL 2025"),
        ("Conference on Fairness, Accountability and Transparency 2025", 2025, "FAccT 2025"),

        # Proceedings prefix stripping + Volume tail stripping
        ("Proceedings of the 19th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers) 2025",
                                                              2025, "EACL 2025"),
        ("Proceedings of the 8th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP 2025",
                                                              2025, "BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP 2025"),
        ("Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers) 2025",
                                                              2025, "ACL 2025"),

        # Poster/Spotlight/Oral suffix stripping (per spec — Workshop is preserved)
        ("ICLR 2026 Poster",                        2026, "ICLR 2026"),
        ("ICLR 2026 Conference Desk Rejected Submission", 2026, "ICLR 2026"),
        ("ACL-SRW 2025 Poster",                     2025, "ACL-SRW 2025"),
        ("CogInterp @ NeurIPS 2025 Poster",         2025, "CogInterp @ NeurIPS 2025"),
        ("CODEML@ICML25 Spotlight 2025",            2025, "CODEML@ICML25 2025"),
        # Workshop preserved
        ("ICLR 2026 Workshop LLM Reasoning",        2026, "ICLR 2026 Workshop LLM Reasoning"),
        ("BlackboxNLP 2025",                        2025, "BlackboxNLP 2025"),

        # Ellipsis truncation
        ("ICASSP- IEEE … 2026",                     2026, "ICASSP- IEEE 2026"),
        ("ICML Workshop … 2024",                    2024, "ICML Workshop 2024"),
        ("Workshop on Latent {\\& … 2026",          2026, "Workshop on Latent {\\& 2026"),

        # Cornell University parenthetical
        ("arXiv (Cornell University) 2025",         2025, ""),  # → preprint sentinel after strip

        # Repeated year collapse
        ("EMNLP 2025 2025",                         2025, "EMNLP 2025"),

        # Comma-before-year (from arXiv journal_ref like "ICLR, 2025")
        ("ICLR, 2025",                              2025, "ICLR 2025"),
        ("International Conference on Learning Representations, 2025", 2025, "ICLR 2025"),

        # Leading article ("the ICLR 2026 Workshop ..." → "ICLR 2026 Workshop ...")
        ("the ICLR 2026 Workshop on LLM Reasoning", 2026, "ICLR 2026 Workshop on LLM Reasoning"),
        ("the ICML 2025",                           2025, "ICML 2025"),
        ("an EMNLP 2024 paper",                     2024, "EMNLP 2024 paper"),

        # Year append when missing
        ("Nature Methods",                          2024, "Nature Methods 2024"),
        ("ICML",                                    2025, "ICML 2025"),

        # Already clean — no-op
        ("ICML 2025",                               2025, "ICML 2025"),
        ("Nature Methods 2024",                     2024, "Nature Methods 2024"),

        # Preprint sentinels normalize to empty (callers fall back to ArXiv {year})
        ("CoRR 2026",                               2026, ""),
        ("UvA-DARE (University of Amsterdam) 2025", 2025, ""),
        ("arXiv preprint arXiv … 2026",             2026, ""),
        ("openreview.net",                          0,    ""),
        ("",                                        2024, ""),
    ])
    def test_normalize(self, raw, year, expected):
        assert normalize_venue(raw, year) == expected


# ---------------------------------------------------------------------------
# _parse_arxiv_comment
# ---------------------------------------------------------------------------

class TestParseArxivComment:
    @pytest.mark.parametrize("comment,expected", [
        ("Accepted at ICLR 2025",                       "ICLR 2025"),
        ("Presented at NeurIPS 2024 main track",         "NeurIPS 2024"),
        ("Accepted to NeurIPS 2024",                    "NeurIPS 2024"),
        ("Published in Nature Methods 2024",            "Nature Methods 2024"),
        ("To appear in ACL 2025",                       "ACL 2025"),
        ("Appeared at ICML 2024",                       "ICML 2024"),
        ("13 pages, accepted at NeurIPS 2025 — code at github.com/x", "NeurIPS 2025"),
        # Trailing comma / period after venue
        ("Accepted to ICLR 2025, camera-ready version", "ICLR 2025"),
        # No match
        ("13 pages, 5 figures",                         ""),
        ("",                                            ""),
    ])
    def test_parse(self, comment, expected):
        assert _parse_arxiv_comment(comment) == expected


# ---------------------------------------------------------------------------
# resolve_venue (integration)
# ---------------------------------------------------------------------------

class TestResolveVenue:
    def test_doi_prefix_wins_over_other_sources(self):
        # DOI says ACL 2025, S2 says some long form
        paper = make_paper(
            title="X",
            doi="10.18653/v1/2025.acl-long.42",
            arxiv_id="2505.16612",
            venue="Conference of the European Chapter…",
            year=2025,
        )
        sources = {"s2": "Annual Meeting of the Association for Computational Linguistics"}
        assert resolve_venue(paper, sources) == "ACL 2025"

    def test_arxiv_comment_acceptance(self):
        # No DOI; arXiv comment names the venue
        paper = make_paper(
            title="X", doi=None, arxiv_id="2401.00001",
            venue="arXiv 2024", year=2024,
        )
        sources = {"arxiv_comment": "10 pages, accepted at NeurIPS 2024"}
        assert resolve_venue(paper, sources) == "NeurIPS 2024"

    def test_openalex_long_form_normalized(self):
        # OpenAlex returns the long form; we normalize to acronym
        paper = make_paper(
            title="X", doi=None, arxiv_id="2410.17194",
            venue="", year=2024,
        )
        sources = {"openalex": "International Conference on Machine Learning"}
        assert resolve_venue(paper, sources) == "ICML 2024"

    def test_existing_venue_used_if_clean(self):
        paper = make_paper(
            title="X", doi=None, arxiv_id=None,
            venue="ICML 2025", year=2025,
        )
        assert resolve_venue(paper, {}) == "ICML 2025"

    def test_fallback_to_arxiv_when_all_empty(self):
        paper = make_paper(
            title="X", doi=None, arxiv_id="2401.00001",
            venue="arXiv preprint arXiv … 2026", year=2026,
        )
        # No external sources, existing venue is a sentinel after normalization
        assert resolve_venue(paper, {}) == "ArXiv 2026"

    def test_fallback_to_arxiv_when_no_year(self):
        paper = make_paper(
            title="X", doi=None, arxiv_id=None,
            venue="openreview.net", year=0,
        )
        assert resolve_venue(paper, {}) == "ArXiv"

    def test_title_search_only_when_no_identifiers(self):
        # No DOI, no arxiv_id, no openalex_id, and existing venue is a sentinel
        # (so steps 1-8 produce nothing and step 9 — title_search — fires)
        paper = make_paper(
            title="Some Truncated Paper",
            doi=None, arxiv_id=None, openalex_id=None,
            venue="openreview.net", year=2024,
        )
        called = []

        def fake_title_search(title):
            called.append(title)
            return "NeurIPS 2024 Workshop on Foo"

        result = resolve_venue(paper, {}, title_search_fn=fake_title_search)
        assert called == ["Some Truncated Paper"]
        assert result == "NeurIPS 2024 Workshop on Foo"

    def test_title_search_skipped_when_arxiv_id_present(self):
        # arxiv_id present → don't bother with title search; fall back to ArXiv
        paper = make_paper(
            title="X", arxiv_id="2401.00001", doi=None, openalex_id=None,
            venue="…", year=2024,
        )
        called = []
        result = resolve_venue(
            paper, {}, title_search_fn=lambda t: called.append(t) or "X 2024"
        )
        assert called == []
        assert result == "ArXiv 2024"

    def test_priority_order_doi_beats_arxiv_comment(self):
        # DOI prefix → ACL 2025; arxiv comment says NeurIPS 2025
        paper = make_paper(
            title="X",
            doi="10.18653/v1/2025.acl-long.1",
            arxiv_id="2401.00001",
            venue="", year=2025,
        )
        sources = {"arxiv_comment": "Accepted at NeurIPS 2025"}
        assert resolve_venue(paper, sources) == "ACL 2025"

    def test_repository_leak_falls_back(self):
        # Existing venue is "UvA-DARE…" (institutional repo); paper has DOI → NAACL
        paper = make_paper(
            title="X",
            doi="10.18653/v1/2025.naacl-long.16",
            arxiv_id="2412.05353",
            venue="UvA-DARE (University of Amsterdam) 2025", year=2025,
        )
        assert resolve_venue(paper, {}) == "NAACL 2025"

    def test_journal_ref_freeform_text_rejected(self):
        # arXiv journal_ref is often free-form bibliographic text. Without a
        # "Accepted at X YYYY" structured form, we MUST NOT use it as a venue.
        # The pipeline should fall through to the next source (or ArXiv fallback)
        # rather than emit a mangled string.
        paper = make_paper(
            title="X", arxiv_id="2510.X", doi=None,
            venue="Workshop at NeurIPS 2025", year=2025,
        )
        sources = {
            "arxiv_journal_ref": "39th Conference on Neural Information Processing Systems (NeurIPS 2025)"
        }
        # Should fall through journal_ref (no "Accepted at" prefix) and use existing
        # paper.venue, which normalizes cleanly.
        result = resolve_venue(paper, sources)
        assert result == "Workshop at NeurIPS 2025"

    def test_mangled_normalize_output_falls_through(self):
        # OpenAlex returns "39th Conference on Neural Information Processing Systems"
        # which after acronym substitution becomes "39th Conference on NeurIPS" —
        # mangled (leading ordinal). resolve_venue must reject it and fall through.
        paper = make_paper(
            title="X", arxiv_id="2510.X", doi=None,
            venue="", year=2025,
        )
        sources = {
            "openalex": "39th Conference on Neural Information Processing Systems"
        }
        result = resolve_venue(paper, sources)
        # Fell through every source; should fall back to ArXiv 2025.
        assert result == "ArXiv 2025"

    def test_truncated_existing_venue_rejected(self):
        # Scholar-truncated existing venue ("Advances in Neural… 2025" → after
        # ellipsis strip → "Advances in Neural 2025") doesn't contain any known
        # acronym/journal. resolve_venue must reject it and fall back to ArXiv.
        paper = make_paper(
            title="X", arxiv_id="2510.X", doi=None,
            venue="Advances in Neural 2025", year=2025,
        )
        result = resolve_venue(paper, sources={})
        assert result == "ArXiv 2025"

    def test_unknown_venue_with_year_rejected(self):
        # Random unrecognized venue string falls through, regardless of having a year.
        paper = make_paper(
            title="X", arxiv_id="2510.X", doi=None,
            venue="Handbook of Human 2025", year=2025,
        )
        assert resolve_venue(paper, sources={}) == "ArXiv 2025"

    def test_known_workshop_kept_via_parent_acronym(self):
        # Workshop named after a parent venue ("Workshop at NeurIPS 2025") passes
        # the confidence gate because it contains a known acronym (NeurIPS).
        paper = make_paper(
            title="X", arxiv_id="2510.X", doi=None,
            venue="Workshop at NeurIPS 2025", year=2025,
        )
        assert resolve_venue(paper, sources={}) == "Workshop at NeurIPS 2025"

    def test_known_journal_kept(self):
        # "Nature Methods 2024" passes because "Nature" is in the known journals list.
        paper = make_paper(
            title="X", arxiv_id="2510.X", doi="10.1038/something",
            venue="Nature Methods 2024", year=2024,
        )
        assert resolve_venue(paper, sources={}) == "Nature Methods 2024"

    def test_double_year_collapse(self):
        paper = make_paper(
            title="X", doi="10.18653/v1/2025.emnlp-main.56",
            arxiv_id=None, venue="2025 2025", year=2025,
        )
        # DOI prefix wins anyway, but exercise the path
        assert resolve_venue(paper, {}) == "EMNLP 2025"
