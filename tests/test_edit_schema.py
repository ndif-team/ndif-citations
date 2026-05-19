"""Smoke tests for the EDITABLE_FIELDS schema."""
from ndif_citations.edit_schema import EDITABLE_FIELDS, get_field
from ndif_citations.models import Bucket, Category


def test_schema_is_non_empty():
    assert len(EDITABLE_FIELDS) >= 10


def test_no_id_fields_in_schema():
    names = {f.name for f in EDITABLE_FIELDS}
    for forbidden in ("arxiv_id", "doi", "openalex_id", "s2_paper_id",
                      "content_hash", "has_summary", "processing_bucket"):
        assert forbidden not in names, f"{forbidden!r} should not be editable"


def test_category_parses_enum():
    field = get_field("category")
    assert field.parse("uses_ndif") == Category.USES_NDIF


def test_bucket_parses_enum():
    field = get_field("bucket")
    assert field.parse("pending") == Bucket.PENDING


def test_year_parses_int():
    field = get_field("year")
    assert field.parse("2025") == 2025


def test_unknown_field_returns_none():
    assert get_field("nonexistent") is None


def test_optional_str_empty_clears():
    field = get_field("project_url")
    assert field.parse("") is None
    assert field.parse("https://example.com") == "https://example.com"


def test_bool_parser_handles_yes_no():
    field = get_field("peer_reviewed")
    assert field.parse("yes") is True
    assert field.parse("no") is False
    assert field.parse("1") is True
    assert field.parse("0") is False


def test_reason_parser_handles_none():
    field = get_field("reason")
    assert field.parse("") is None
    assert field.parse("none") is None
