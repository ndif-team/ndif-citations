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


from dataclasses import dataclass


SOURCE_TRUST: dict[str, int] = {
    "openalex": 4, "crossref": 3, "arxiv": 3, "s2": 3,
    "manual_add": 2, "scholar": 1, "unknown": 0,
}


@dataclass(frozen=True)
class Candidate:
    value: object
    source: str


@dataclass(frozen=True)
class Resolution:
    value: object
    source: str
    changed: bool
    low_confidence: bool


def _trust(source: str) -> int:
    return SOURCE_TRUST.get(source, 0)


def _completeness(field: str, value) -> int:
    if field in ("abstract", "authors") and isinstance(value, str):
        return len(value)
    return 1


def _score(field: str, c: Candidate):
    # higher tuple wins: valid first, then trust, then completeness
    return (0 if is_broken(field, c.value) else 1, _trust(c.source), _completeness(field, c.value))


def reconcile_field(field, current: Candidate, candidates: list[Candidate],
                    low_confidence_sources: set[str] | None = None) -> Resolution:
    low_confidence_sources = low_confidence_sources or set()
    pool = [current] + [c for c in candidates
                        if (c.value not in (None, "", 0))]
    winner = max(pool, key=lambda c: _score(field, c))
    # Regression guard: never downgrade a non-broken current to an equal-or-worse value.
    if not is_broken(field, current.value) and _score(field, winner) <= _score(field, current):
        winner = current
    changed = winner.value != current.value
    low_conf = changed and winner.source in low_confidence_sources
    return Resolution(value=winner.value, source=winner.source, changed=changed, low_confidence=low_conf)
