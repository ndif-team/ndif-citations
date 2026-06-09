"""Data-quality — strip raw LaTeX markup from abstracts during enrichment.

Some abstracts render literal LaTeX (e.g. ADAG: "\\textbf{circuit tracing}",
"\\textit{...}"). strip_latex() unwraps text-formatting commands and de-escapes
LaTeX-escaped characters; enrich_paper applies it to the final abstract for
non-locked papers (curator-locked papers are left untouched).
"""
from __future__ import annotations

import pytest

from ndif_citations import enrichment
from ndif_citations.enrichment import Record, enrich_paper
from ndif_citations.utils import strip_latex

from tests.conftest import make_paper


# ---------------------------------------------------------------------------
# strip_latex helper
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [
    (r"We propose \textbf{circuit tracing} for models.", "We propose circuit tracing for models."),
    (r"A \textit{novel} method.", "A novel method."),
    (r"\emph{Emphasis} and \texttt{code}.", "Emphasis and code."),
    (r"Nested \textbf{\textit{deep}} markup.", "Nested deep markup."),
    (r"Escaped 50\% gain, A \& B, x\_y.", "Escaped 50% gain, A & B, x_y."),
    ("Plain text with no markup.", "Plain text with no markup."),
    ("", ""),
])
def test_strip_latex_cases(raw, expected):
    assert strip_latex(raw) == expected


def test_strip_latex_is_idempotent():
    raw = r"\textbf{a} \textit{b} 10\% \emph{c}"
    once = strip_latex(raw)
    assert strip_latex(once) == once


def test_strip_latex_leaves_math_delimiters_alone():
    # Conservative: we don't try to render math, only unwrap text formatting.
    assert strip_latex(r"The loss $L = \sum x_i$ is minimized.") == r"The loss $L = \sum x_i$ is minimized."


# ---------------------------------------------------------------------------
# enrich_paper integration
# ---------------------------------------------------------------------------

def _no_network(monkeypatch):
    monkeypatch.setattr(enrichment, "resolve_identifiers", lambda paper: enrichment.ResolveResult(resolved=False))
    monkeypatch.setattr(enrichment, "fetch_records", lambda paper: [])


def test_enrich_strips_latex_from_existing_abstract(monkeypatch):
    # A legacy, non-broken abstract (repair-only would NOT otherwise touch it) that
    # carries raw LaTeX still gets normalized.
    _no_network(monkeypatch)
    p = make_paper(
        arxiv_id="2501.1",
        abstract=r"We introduce \textbf{circuit tracing}, a \textit{novel} approach.",
        manual_override=False,
    )
    enrich_paper(p)
    assert p.abstract == "We introduce circuit tracing, a novel approach."


def test_enrich_does_not_touch_latex_in_locked_paper(monkeypatch):
    _no_network(monkeypatch)
    raw = r"Curated \textbf{abstract} kept verbatim."
    p = make_paper(arxiv_id="2501.2", abstract=raw, manual_override=True)
    enrich_paper(p)
    assert p.abstract == raw  # curator lock wins — no cosmetic edits
