"""Authoritative-source metadata reconciliation engine + orchestration.

Pure helpers (is_broken / reconcile_field / reconcile_paper) take already-fetched
values and decide the best one; the fetch/orchestration helpers reuse existing
query functions. See docs/superpowers/specs/2026-06-06-robust-enrichment-design.md.
"""
from __future__ import annotations

from ndif_citations.venue import _WEAK_VENUE_RE

_ELLIPSIS = ("…", "...")
_ABSTRACT_MIN = 280


def is_broken(field: str, value) -> bool:
    """True if `value` for `field` is empty/truncated/low-quality."""
    if field == "year":
        return not value
    text = (value or "").strip() if isinstance(value, str) else ""
    if not text:
        return True
    if field == "abstract":
        return text.endswith(_ELLIPSIS) or len(text) < _ABSTRACT_MIN
    if field == "authors":
        stripped = text.rstrip(". ")
        return text.endswith(_ELLIPSIS) or "…" in text or stripped.endswith("et al")
    if field == "venue":
        return bool(_WEAK_VENUE_RE.match(text))
    if field == "affiliations":
        return False  # non-empty affiliations are acceptable
    return False
