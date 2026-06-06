import pytest
from ndif_citations.enrichment import is_broken


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
