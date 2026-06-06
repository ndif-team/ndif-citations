import pytest
from ndif_citations.enrichment import is_broken, reconcile_field, Candidate


@pytest.mark.parametrize("value,expected", [
    ("", True),
    ("Short snippet about a model …", True),          # ellipsis
    ("Short snippet about a model ...", True),         # ascii ellipsis
    ("x" * 200, True),                                  # under 280-char floor
    ("A full abstract. " * 30, False),                  # long, complete
])
def test_is_broken_abstract(value, expected):
    assert is_broken("abstract", value) is expected


@pytest.mark.parametrize("value,expected", [
    ("", True),
    ("J. Smith, A. …", True),
    ("Jane Smith et al", True),                         # truncated author list
    ("Jane Smith et al.", True),                        # with trailing period
    ("Jane Smith, Alan Turing", False),
])
def test_is_broken_authors(value, expected):
    assert is_broken("authors", value) is expected


def test_is_broken_venue_and_affiliations_and_year():
    assert is_broken("venue", "") is True
    assert is_broken("venue", "arXiv") is True          # weak (via venue._WEAK_VENUE_RE)
    assert is_broken("venue", "NeurIPS 2024") is False
    assert is_broken("affiliations", "") is True
    assert is_broken("affiliations", "MIT") is False
    assert is_broken("year", 0) is True
    assert is_broken("year", 2024) is False


def _c(value, source): return Candidate(value=value, source=source)


def test_reconcile_replaces_broken_with_valid():
    r = reconcile_field("abstract", _c("snippet …", "scholar"),
                         [_c("A" * 600, "openalex")])
    assert r.changed and r.value == "A" * 600 and r.source == "openalex"


def test_reconcile_keeps_good_over_lower_trust():
    good = "A clean full abstract. " * 30
    r = reconcile_field("abstract", _c(good, "openalex"),
                         [_c("Another full abstract. " * 30, "scholar")])
    assert r.changed is False and r.source == "openalex"


def test_reconcile_tie_on_validity_prefers_most_trusted():
    a = "Full abstract alpha. " * 30
    b = "Full abstract bravo. " * 30  # same length tier, different source
    r = reconcile_field("abstract", _c(a, "s2"), [_c(b, "openalex")])
    assert r.value == b and r.source == "openalex"


def test_reconcile_no_candidates_unchanged():
    r = reconcile_field("authors", _c("Jane Smith, Alan Turing", "s2"), [])
    assert r.changed is False


def test_reconcile_low_confidence_flag_propagates():
    r = reconcile_field("abstract", _c("snippet …", "scholar"),
                        [_c("A" * 600, "openalex")], low_confidence_sources={"openalex"})
    assert r.changed and r.low_confidence is True
