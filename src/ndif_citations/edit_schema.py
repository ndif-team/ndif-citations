"""Field schema for the `edit` CLI command.

Curated list of fields a curator can override interactively. Each entry
declares the field name, the Python type for parsing, and a short
human-readable description.

Identifiers (arxiv_id, doi, s2_paper_id, openalex_id) and computed
fields (content_hash, has_*, processing_bucket, date_discovered,
classification_signal, unclassified_reason) are intentionally excluded
— editing them risks silent dedup failures.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

from ndif_citations.models import Bucket, Category, PaperReason


@dataclass(frozen=True)
class EditableField:
    name: str
    label: str          # display name in the prompt
    parser: Callable[[str], Any]   # str → typed value
    description: str    # short hint

    def parse(self, raw: str) -> Any:
        return self.parser(raw)


def _parse_int(s: str) -> int:
    return int(s.strip())


def _parse_optional_str(s: str) -> Optional[str]:
    s = s.strip()
    return s if s else None


def _parse_bool(s: str) -> Optional[bool]:
    s = s.strip().lower()
    if s in ("y", "yes", "true", "t", "1"):
        return True
    if s in ("n", "no", "false", "f", "0"):
        return False
    if s == "":
        return None
    raise ValueError(f"Expected yes/no, got {s!r}")


def _parse_category(s: str) -> Category:
    return Category(s.strip().lower())


def _parse_bucket(s: str) -> Bucket:
    return Bucket(s.strip().lower())


def _parse_reason(s: str) -> Optional[PaperReason]:
    s = s.strip().lower()
    if not s or s == "none":
        return None
    return PaperReason(s)


EDITABLE_FIELDS: list[EditableField] = [
    EditableField("title",          "Title",          str,                 "Paper title"),
    EditableField("authors",        "Authors",        str,                 "Comma-separated author list"),
    EditableField("affiliations",   "Affiliations",   str,                 "Comma-separated institutions"),
    EditableField("venue",          "Venue",          str,                 "Conference / journal / 'ArXiv YYYY'"),
    EditableField("year",           "Year",           _parse_int,          "Publication year"),
    EditableField("category",       "Category",       _parse_category,     "uses_ndif | uses_nnsight | referencing | unclassified"),
    EditableField("description",    "Description",    str,                 "1-3 sentence website summary"),
    EditableField("url",            "URL",            str,                 "Landing page URL"),
    EditableField("pdf_url",        "PDF URL",        _parse_optional_str, "Direct PDF link (or empty to clear)"),
    EditableField("project_url",    "Project URL",    _parse_optional_str, "GitHub or project page (or empty to clear)"),
    EditableField("image",          "Image path",     _parse_optional_str, "e.g. /images/Slug.png (or empty to clear)"),
    EditableField("bucket",         "Bucket",         _parse_bucket,       "pending | verified | discarded"),
    EditableField("reason",         "Reason",         _parse_reason,       "PaperReason value or empty for none"),
    EditableField("reason_detail",  "Reason detail",  _parse_optional_str, "Free-text supplement"),
    EditableField("peer_reviewed",  "Peer reviewed",  _parse_bool,         "yes/no"),
    EditableField("abstract",       "Abstract",       _parse_optional_str, "Full abstract text"),
]


def get_field(name: str) -> Optional[EditableField]:
    """Lookup field by name (case-insensitive). Returns None if not editable."""
    name_lower = name.lower()
    for f in EDITABLE_FIELDS:
        if f.name.lower() == name_lower:
            return f
    return None
