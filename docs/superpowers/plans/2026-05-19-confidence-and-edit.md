# Confidence Bands + Interactive Edit + Fill-Gaps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace single-float `category_confidence` with a categorical `Confidence` band tied to evidence type, add an interactive `edit <id>` CLI for per-paper field overrides, and let the router fill empty fields on `manual_override=True` papers without overwriting curated values.

**Architecture:** Add `Confidence` enum to `models.py` (CERTAIN/HIGH/MEDIUM/LOW/NONE) and persist alongside the existing float for backwards-compat. Compute the band deterministically from a pure helper that reads existing classification metadata (linked_paper_tier, surviving window count, context source, signal). Add a new `MEDIUM_CONFIDENCE` `PaperReason`. Change `router.py` so `manual_override=True` papers with empty `has_*` flags route to `FILL_GAPS` instead of `PROTECTED`, and guard `process.py`'s FILL_GAPS path to skip writes when the field is already non-empty on a manually-overridden paper. Add a new `edit` click command that walks a curated field schema interactively (menu-driven), auto-sets `manual_override=True`, re-derives `has_*` flags, and re-runs `_decide_bucket`.

**Tech Stack:** Python 3.10+, Click, Pydantic v2, Rich (already in deps), pytest + pytest-mock (dev deps).

---

## File Structure

**Modify:**
- `src/ndif_citations/models.py` — add `Confidence` enum, `MEDIUM_CONFIDENCE` reason, `category_confidence_band` field, migration in `model_post_init`
- `src/ndif_citations/process.py` — add `_compute_confidence_band` helper, rewire `classify_category` to set band, update `_decide_bucket` to gate on band, add FILL_GAPS-protect guard
- `src/ndif_citations/router.py` — change `_route_single_paper` so `manual_override=True` with empty fields routes to `FILL_GAPS`
- `src/ndif_citations/cli.py` — add `edit` command (and `edit --new <url>` to replace the broken `add`)
- `src/ndif_citations/output.py` — update low-confidence report to use bands

**Create:**
- `tests/test_confidence_band.py` — pure-function tests for `_compute_confidence_band`
- `tests/test_classify_band_integration.py` — integration tests confirming `classify_category` emits expected bands
- `tests/test_router_protected_fill_gaps.py` — router promotes PROTECTED → FILL_GAPS when fields empty
- `tests/test_process_fill_gaps_protect.py` — process.py guard: don't overwrite non-empty fields on manual_override
- `tests/test_cli_edit.py` — end-to-end interactive edit, --new mode, --dry-run, --field flags

**Note:** No deletions in this PR. The dead-code cleanup (`_write_csv`, `calculate_image_score`, etc.) and the `add` command bug are deferred to a follow-up PR.

---

## Task 1: Add `Confidence` enum + new fields to `DiscoveredPaper`

**Files:**
- Modify: `src/ndif_citations/models.py:1-30` (imports + new Enum)
- Modify: `src/ndif_citations/models.py:96-135` (DiscoveredPaper fields)
- Modify: `src/ndif_citations/models.py:143-155` (model_post_init migration)
- Test: `tests/test_confidence_band.py` (new file — for the migration unit test only)

- [ ] **Step 1: Write the failing test for migration from legacy float**

Create `tests/test_confidence_band.py`:

```python
"""Tests for the Confidence enum, band ↔ float migration, and the
_compute_confidence_band rule.
"""
from __future__ import annotations

import pytest

from ndif_citations.models import (
    Confidence,
    Category,
    DiscoveredPaper,
)


class TestConfidenceFloatBridge:
    """Round-trip between the float (legacy) and the band (new)."""

    def test_band_to_float_mapping(self):
        from ndif_citations.models import _BAND_TO_FLOAT
        assert _BAND_TO_FLOAT[Confidence.CERTAIN] == 1.0
        assert _BAND_TO_FLOAT[Confidence.HIGH] == 0.85
        assert _BAND_TO_FLOAT[Confidence.MEDIUM] == 0.55
        assert _BAND_TO_FLOAT[Confidence.LOW] == 0.30
        assert _BAND_TO_FLOAT[Confidence.NONE] == 0.0

    def test_legacy_float_load_derives_band_high(self):
        """Loading an old paper with category_confidence=0.85 → band HIGH."""
        paper = DiscoveredPaper(
            title="X", abstract="Y",
            category=Category.USES_NDIF,
            category_confidence=0.85,
        )
        assert paper.category_confidence_band == Confidence.HIGH

    def test_legacy_float_load_derives_band_medium(self):
        paper = DiscoveredPaper(
            title="X", abstract="Y",
            category=Category.USES_NNSIGHT,
            category_confidence=0.55,
        )
        assert paper.category_confidence_band == Confidence.MEDIUM

    def test_legacy_float_load_derives_band_low(self):
        paper = DiscoveredPaper(
            title="X", abstract="Y",
            category_confidence=0.40,
        )
        assert paper.category_confidence_band == Confidence.LOW

    def test_legacy_float_load_derives_band_none(self):
        paper = DiscoveredPaper(
            title="X", abstract="Y",
            category=Category.UNCLASSIFIED,
            category_confidence=0.0,
        )
        assert paper.category_confidence_band == Confidence.NONE

    def test_manual_override_with_any_confidence_promotes_to_certain(self):
        """A paper with manual_override=True should be CERTAIN regardless of stored float."""
        paper = DiscoveredPaper(
            title="X", abstract="Y",
            category=Category.USES_NDIF,
            category_confidence=0.40,
            manual_override=True,
        )
        assert paper.category_confidence_band == Confidence.CERTAIN
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_confidence_band.py::TestConfidenceFloatBridge -v`
Expected: FAIL (ImportError for `Confidence`)

- [ ] **Step 3: Add `Confidence` enum + field + migration**

Edit `src/ndif_citations/models.py`. Add the enum right after `Category` (around line 19):

```python
class Confidence(str, Enum):
    """Categorical confidence band tied to evidence type (replaces the
    informal float thresholds). Each band carries a backwards-compat
    float in _BAND_TO_FLOAT for any caller still reading the legacy
    category_confidence field.
    """
    CERTAIN = "certain"  # manual_override OR pre_filter:negative_evidence
    HIGH    = "high"     # LLM verdict on rich evidence
    MEDIUM  = "medium"   # LLM verdict on thin evidence
    LOW     = "low"      # keyword fallback (no LLM)
    NONE    = "none"     # UNCLASSIFIED


_BAND_TO_FLOAT: dict[Confidence, float] = {
    Confidence.CERTAIN: 1.00,
    Confidence.HIGH:    0.85,
    Confidence.MEDIUM:  0.55,
    Confidence.LOW:     0.30,
    Confidence.NONE:    0.00,
}


def _float_to_band(value: float, *, manual_override: bool = False) -> Confidence:
    """Derive a band from a legacy float. Used to migrate existing JSON state
    that predates the band field. Bucket thresholds chosen to match the
    historical 0.0 / 0.4 / 0.55 / 0.85 / 1.0 values produced by classify_category.
    """
    if manual_override:
        return Confidence.CERTAIN
    if value >= 0.95:
        return Confidence.CERTAIN
    if value >= 0.70:
        return Confidence.HIGH
    if value >= 0.50:
        return Confidence.MEDIUM
    if value > 0.0:
        return Confidence.LOW
    return Confidence.NONE
```

Add the new field to `DiscoveredPaper` (around line 97, alongside `category_confidence`):

```python
    category_confidence: float = 0.0
    category_confidence_band: Confidence = Confidence.NONE
```

Extend `model_post_init` (around line 143) to migrate legacy data:

```python
    def model_post_init(self, __context):
        """Auto-compute hash on creation if not set."""
        if not self.content_hash:
            self.content_hash = self.compute_hash()
        # Migrate legacy float → band when band is at its default but float isn't,
        # OR when manual_override is set (band should be CERTAIN).
        if self.manual_override and self.category_confidence_band != Confidence.CERTAIN:
            self.category_confidence_band = Confidence.CERTAIN
        elif (
            self.category_confidence_band == Confidence.NONE
            and self.category_confidence > 0.0
        ):
            self.category_confidence_band = _float_to_band(self.category_confidence)
        if not hasattr(self, 'has_summary') or self.has_summary is None:
            self.has_summary = bool(self.description)
        if not hasattr(self, 'has_classification') or self.has_classification is None:
            self.has_classification = self.category != Category.REFERENCING or self.category_confidence > 0
        if not hasattr(self, 'has_thumbnail') or self.has_thumbnail is None:
            self.has_thumbnail = bool(self.image)
        if not hasattr(self, 'has_affiliations') or self.has_affiliations is None:
            self.has_affiliations = bool(self.affiliations)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_confidence_band.py::TestConfidenceFloatBridge -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Run full test suite to confirm no regression**

Run: `pytest tests/ -x -q`
Expected: All previously-passing tests still pass.

- [ ] **Step 6: Commit**

```bash
git add src/ndif_citations/models.py tests/test_confidence_band.py
git commit -m "feat(models): add Confidence enum and band field

Categorical band replaces the implicit thresholds on the
category_confidence float. The float is preserved for backwards
compat and auto-migrated to a band on load via _float_to_band.
manual_override=True always implies CERTAIN."
```

---

## Task 2: Add `PaperReason.MEDIUM_CONFIDENCE`

**Files:**
- Modify: `src/ndif_citations/models.py:28-41` (PaperReason enum)
- Test: `tests/test_confidence_band.py` (extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_confidence_band.py`:

```python
class TestNewPaperReason:
    def test_medium_confidence_reason_exists(self):
        from ndif_citations.models import PaperReason
        assert PaperReason.MEDIUM_CONFIDENCE.value == "medium_confidence"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_confidence_band.py::TestNewPaperReason -v`
Expected: FAIL (AttributeError on MEDIUM_CONFIDENCE)

- [ ] **Step 3: Add reason value**

Edit `src/ndif_citations/models.py:30-41` to add `MEDIUM_CONFIDENCE` after `LOW_CONFIDENCE`:

```python
class PaperReason(str, Enum):
    """Reason a paper was placed in pending or discarded."""
    # Pending reasons
    OPENALEX_SOURCE = "openalex_source"
    LOW_CONFIDENCE = "low_confidence"
    MEDIUM_CONFIDENCE = "medium_confidence"
    UNCLASSIFIED_NO_KEYWORDS = "unclassified_no_keywords"
    UNCLASSIFIED_LLM = "unclassified_llm"
    STUB_METADATA = "stub_metadata"
    # Discarded reasons
    ZERO_PDF_HITS = "zero_pdf_hits"
    MANUAL_DISCARD = "manual_discard"
    # Manual curator override
    MANUAL_DEMOTE = "manual_demote"
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_confidence_band.py::TestNewPaperReason -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/ndif_citations/models.py tests/test_confidence_band.py
git commit -m "feat(models): add PaperReason.MEDIUM_CONFIDENCE

New pending reason for papers whose evidence is real but thin
(single-window LLM verdict or abstract-only context)."
```

---

## Task 3: Implement `_compute_confidence_band` pure helper

**Files:**
- Create: helper at top of `src/ndif_citations/process.py` after the LLM client section
- Test: `tests/test_confidence_band.py` (extend with `TestComputeBand`)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_confidence_band.py`:

```python
class TestComputeBand:
    """Pure-function rule: signal + tier + window_count + context_source -> band.

    These tests pin down the band assignment rules described in
    docs/superpowers/plans/2026-05-19-confidence-and-edit.md.
    """

    def test_pre_filter_negative_is_certain(self):
        from ndif_citations.process import _compute_confidence_band
        band = _compute_confidence_band(
            signal="pre_filter:negative_evidence",
            linked_paper_tier=None,
            surviving_window_count=0,
            context_source="pdf",
            category=Category.REFERENCING,
        )
        assert band == Confidence.CERTAIN

    def test_pre_filter_table_or_acks_is_medium(self):
        from ndif_citations.process import _compute_confidence_band
        for signal in ("pre_filter:comparison_table", "pre_filter:acks_only_thank_you"):
            band = _compute_confidence_band(
                signal=signal,
                linked_paper_tier=None,
                surviving_window_count=0,
                context_source="pdf",
                category=Category.REFERENCING,
            )
            assert band == Confidence.MEDIUM, f"{signal!r} → {band}"

    def test_llm_with_tier_1_or_2_cross_link_is_high(self):
        from ndif_citations.process import _compute_confidence_band
        for tier in (1, 2):
            band = _compute_confidence_band(
                signal="llm",
                linked_paper_tier=tier,
                surviving_window_count=1,
                context_source="abstract",
                category=Category.USES_NNSIGHT,
            )
            assert band == Confidence.HIGH, f"tier={tier} → {band}"

    def test_llm_with_multi_window_pdf_is_high(self):
        from ndif_citations.process import _compute_confidence_band
        band = _compute_confidence_band(
            signal="llm",
            linked_paper_tier=None,
            surviving_window_count=3,
            context_source="pdf",
            category=Category.USES_NDIF,
        )
        assert band == Confidence.HIGH

    def test_llm_with_single_window_pdf_is_medium(self):
        from ndif_citations.process import _compute_confidence_band
        band = _compute_confidence_band(
            signal="llm",
            linked_paper_tier=None,
            surviving_window_count=1,
            context_source="pdf",
            category=Category.USES_NDIF,
        )
        assert band == Confidence.MEDIUM

    def test_llm_with_abstract_only_is_medium(self):
        from ndif_citations.process import _compute_confidence_band
        band = _compute_confidence_band(
            signal="llm",
            linked_paper_tier=None,
            surviving_window_count=1,
            context_source="abstract",
            category=Category.USES_NDIF,
        )
        assert band == Confidence.MEDIUM

    def test_keyword_fallback_is_low(self):
        from ndif_citations.process import _compute_confidence_band
        band = _compute_confidence_band(
            signal="keyword_fallback",
            linked_paper_tier=None,
            surviving_window_count=0,
            context_source="pdf",
            category=Category.USES_NDIF,
        )
        assert band == Confidence.LOW

    def test_unclassified_is_none(self):
        from ndif_citations.process import _compute_confidence_band
        band = _compute_confidence_band(
            signal="llm",
            linked_paper_tier=None,
            surviving_window_count=0,
            context_source="none",
            category=Category.UNCLASSIFIED,
        )
        assert band == Confidence.NONE
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_confidence_band.py::TestComputeBand -v`
Expected: FAIL (ImportError for `_compute_confidence_band`)

- [ ] **Step 3: Add the pure helper**

Edit `src/ndif_citations/process.py`. After the LLM client section (around line 40) and before `# 3a. Summary generation`, add:

```python
# ---------------------------------------------------------------------------
# Confidence band derivation (pure function)
# ---------------------------------------------------------------------------

def _compute_confidence_band(
    *,
    signal: str,
    linked_paper_tier: Optional[int],
    surviving_window_count: int,
    context_source: str,
    category: Category,
) -> Confidence:
    """Map classification metadata to a Confidence band.

    Rules (first match wins):
      1. UNCLASSIFIED                                          -> NONE
      2. signal == pre_filter:negative_evidence                -> CERTAIN
      3. signal in (pre_filter:comparison_table,
                    pre_filter:acks_only_thank_you)            -> MEDIUM
      4. signal == keyword_fallback                            -> LOW
      5. signal == llm AND (linked_paper_tier <= 2
                            OR surviving_window_count >= 2)    -> HIGH
      6. signal == llm AND (single window OR abstract-only)    -> MEDIUM
      7. anything else                                          -> LOW (defensive)
    """
    if category == Category.UNCLASSIFIED:
        return Confidence.NONE
    if signal == "pre_filter:negative_evidence":
        return Confidence.CERTAIN
    if signal in (
        "pre_filter:comparison_table",
        "pre_filter:acks_only_thank_you",
    ):
        return Confidence.MEDIUM
    if signal == "keyword_fallback":
        return Confidence.LOW
    if signal == "llm":
        if (linked_paper_tier is not None and linked_paper_tier <= 2):
            return Confidence.HIGH
        if surviving_window_count >= 2:
            return Confidence.HIGH
        return Confidence.MEDIUM
    return Confidence.LOW
```

Update imports at the top of `process.py` to include `Confidence`:

```python
from ndif_citations.models import (
    Bucket, Category, Confidence, DiscoveredPaper, DiscoveredRepo, PaperReason,
)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_confidence_band.py::TestComputeBand -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ndif_citations/process.py tests/test_confidence_band.py
git commit -m "feat(process): add _compute_confidence_band pure helper

Deterministic mapping from (signal, tier, window_count, context_source,
category) to a Confidence band. Used by classify_category to set
paper.category_confidence_band on each run."
```

---

## Task 4: Wire band into `classify_category`

**Files:**
- Modify: `src/ndif_citations/process.py:277-407` (classify_category function)
- Test: `tests/test_classify_band_integration.py` (new file)

This task changes `classify_category` from `(Category, float)` to `(Category, float, Confidence)` and threads the band through callers. It also captures the new metadata (window_count, context_source, signal) inline rather than persisting them.

- [ ] **Step 1: Write integration tests**

Create `tests/test_classify_band_integration.py`:

```python
"""Integration tests: classify_category emits the correct Confidence band
across all decision paths in process.py."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ndif_citations.models import Category, Confidence
from ndif_citations.process import classify_category
from tests.conftest import make_paper


@pytest.fixture
def llm_client_uses_ndif():
    """Mock LLM client returning 'uses_ndif'."""
    client = MagicMock()
    client.chat.completions.create.return_value = MagicMock(
        choices=[MagicMock(message=MagicMock(content="uses_ndif"))]
    )
    return client


class TestClassifyBandPath:
    """Each path through classify_category should produce a known band."""

    def test_no_pdf_no_keywords_in_abstract_is_none(self, tmp_path):
        paper = make_paper(abstract="Totally unrelated abstract about cats.")
        cat, conf, band = classify_category(paper, tmp_path, pdf_path=None)
        assert cat == Category.UNCLASSIFIED
        assert band == Confidence.NONE

    def test_no_pdf_no_abstract_is_none(self, tmp_path):
        paper = make_paper(abstract=None)
        cat, conf, band = classify_category(paper, tmp_path, pdf_path=None)
        assert cat == Category.UNCLASSIFIED
        assert band == Confidence.NONE

    def test_pre_filter_negative_evidence_is_certain(self, tmp_path, monkeypatch):
        # Patch extract_ndif_context to return text with negative evidence
        monkeypatch.setattr(
            "ndif_citations.process.extract_ndif_context",
            lambda *a, **kw: "Our approach is an alternative to NDIF and avoids it entirely.",
        )
        pdf = tmp_path / "fake.pdf"
        pdf.write_bytes(b"%PDF")  # exists()
        paper = make_paper()
        cat, conf, band = classify_category(paper, tmp_path, pdf_path=pdf)
        assert cat == Category.REFERENCING
        assert band == Confidence.CERTAIN

    def test_llm_with_tier_2_link_is_high(self, tmp_path, monkeypatch, llm_client_uses_ndif):
        monkeypatch.setattr(
            "ndif_citations.process.extract_ndif_context",
            lambda *a, **kw: "We use nnsight to investigate model internals.",
        )
        monkeypatch.setattr(
            "ndif_citations.process._get_llm_client",
            lambda: llm_client_uses_ndif,
        )
        pdf = tmp_path / "fake.pdf"
        pdf.write_bytes(b"%PDF")
        paper = make_paper(linked_paper_tier=2)
        cat, conf, band = classify_category(paper, tmp_path, pdf_path=pdf)
        assert cat == Category.USES_NDIF
        assert band == Confidence.HIGH

    def test_llm_with_single_window_pdf_is_medium(self, tmp_path, monkeypatch, llm_client_uses_ndif):
        monkeypatch.setattr(
            "ndif_citations.process.extract_ndif_context",
            lambda *a, **kw: "We mention nnsight once in passing.",
        )
        monkeypatch.setattr(
            "ndif_citations.process._get_llm_client",
            lambda: llm_client_uses_ndif,
        )
        pdf = tmp_path / "fake.pdf"
        pdf.write_bytes(b"%PDF")
        paper = make_paper(linked_paper_tier=None)
        cat, conf, band = classify_category(paper, tmp_path, pdf_path=pdf)
        # Single window, no tier -> MEDIUM
        assert band == Confidence.MEDIUM

    def test_keyword_fallback_when_llm_unavailable_is_low(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "ndif_citations.process.extract_ndif_context",
            lambda *a, **kw: "import nnsight is used in our pipeline.",
        )
        monkeypatch.setattr(
            "ndif_citations.process._get_llm_client",
            lambda: None,  # No client → fallback
        )
        pdf = tmp_path / "fake.pdf"
        pdf.write_bytes(b"%PDF")
        paper = make_paper()
        cat, conf, band = classify_category(paper, tmp_path, pdf_path=pdf)
        assert band == Confidence.LOW
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_classify_band_integration.py -v`
Expected: FAIL — `classify_category` still returns a 2-tuple.

- [ ] **Step 3: Refactor `classify_category` to return `(Category, float, Confidence)`**

Replace the body of `classify_category` in `src/ndif_citations/process.py:277-407`. The new logic emits a `signal` label at each decision, captures `surviving_window_count` and `context_source`, then derives the band via `_compute_confidence_band`. Float is derived from the band via `_BAND_TO_FLOAT`.

Edit `src/ndif_citations/process.py` — replace the existing function:

```python
def classify_category(
    paper: DiscoveredPaper, output_dir: Path,
    pdf_path: Path | None = None,
) -> tuple[Category, float, Confidence]:
    """Classify a paper's relationship to NDIF/NNsight.

    Returns (category, float_confidence, band). float_confidence is derived
    from the band via _BAND_TO_FLOAT — preserved for backwards-compat callers.
    Sets paper.unclassified_reason on UNCLASSIFIED outcomes and
    paper.classification_signal to the label used by _compute_confidence_band.
    """
    from ndif_citations.models import _BAND_TO_FLOAT  # local import to avoid cycle

    context = ""
    context_source = "none"
    signal: str = "llm"

    if pdf_path and pdf_path.exists():
        context = extract_ndif_context(pdf_path, window=config.CONTEXT_WINDOW)
        logger.debug(f"PDF text extracted for '{paper.title}': context length={len(context)}")

    if not context or "No direct mentions" in context:
        if paper.abstract:
            abstract_lower = paper.abstract.lower()
            has_mention = any(kw.lower() in abstract_lower for kw in config.NDIF_KEYWORDS)
            if has_mention:
                context = paper.abstract
                context_source = "abstract"
            else:
                logger.info(f"No NDIF/nnsight mentions found for '{paper.title}' — UNCLASSIFIED")
                paper.unclassified_reason = "no_keywords_anywhere"
                paper.classification_signal = "no_keywords"
                band = Confidence.NONE
                return Category.UNCLASSIFIED, _BAND_TO_FLOAT[band], band
        else:
            logger.info(f"No extractable evidence for '{paper.title}' — UNCLASSIFIED")
            paper.unclassified_reason = "no_evidence_extractable"
            paper.classification_signal = "no_evidence"
            band = Confidence.NONE
            return Category.UNCLASSIFIED, _BAND_TO_FLOAT[band], band
    else:
        context_source = "pdf"

    # Pre-filters
    windows = context.split("\n---\n")
    surviving_windows, prefilter_signal = _apply_prefilters(windows, paper)
    surviving_window_count = len(surviving_windows)

    if not surviving_windows:
        # All windows eliminated by pre-filter — classify without LLM
        paper.classification_signal = prefilter_signal
        band = _compute_confidence_band(
            signal=prefilter_signal or "pre_filter:unknown",
            linked_paper_tier=paper.linked_paper_tier,
            surviving_window_count=0,
            context_source=context_source,
            category=Category.REFERENCING,
        )
        return Category.REFERENCING, _BAND_TO_FLOAT[band], band

    # Rebuild context for LLM
    context = "\n---\n".join(surviving_windows)

    client = _get_llm_client()
    if not client:
        logger.debug(f"No LLM client — keyword fallback for '{paper.title}'")
        cat = _fallback_classification(context)
        paper.classification_signal = "keyword_fallback"
        band = _compute_confidence_band(
            signal="keyword_fallback",
            linked_paper_tier=paper.linked_paper_tier,
            surviving_window_count=surviving_window_count,
            context_source=context_source,
            category=cat,
        )
        return cat, _BAND_TO_FLOAT[band], band

    try:
        system_prompt = _augment_prompt_with_tier(
            _select_classification_prompt(context),
            paper.linked_paper_tier,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": (
                f"Paper title: {paper.title}\n\n"
                f"Evidence of NDIF/nnsight usage:\n{context}"
            )},
        ]
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=messages,
            temperature=0.1,
            max_tokens=20,
        )
        raw = response.choices[0].message.content.strip().lower()
        logger.debug(f"LLM raw reply for '{paper.title}': {raw!r}")
        rate_limit_sleep(config.LLM_RATE_LIMIT_SLEEP, "LLM classify")

        if "uses_ndif" in raw:
            cat = Category.USES_NDIF
        elif "uses_nnsight" in raw:
            cat = Category.USES_NNSIGHT
        elif "referencing" in raw:
            cat = Category.REFERENCING
        elif "unclassified" in raw:
            paper.unclassified_reason = "llm_returned_unclassified"
            paper.classification_signal = "llm"
            band = Confidence.NONE
            return Category.UNCLASSIFIED, _BAND_TO_FLOAT[band], band
        else:
            logger.warning(f"Unexpected LLM classification '{raw}' for '{paper.title}'")
            paper.unclassified_reason = "llm_unparseable"
            paper.classification_signal = "llm"
            band = Confidence.NONE
            return Category.UNCLASSIFIED, _BAND_TO_FLOAT[band], band

        paper.classification_signal = "llm"
        band = _compute_confidence_band(
            signal="llm",
            linked_paper_tier=paper.linked_paper_tier,
            surviving_window_count=surviving_window_count,
            context_source=context_source,
            category=cat,
        )
        return cat, _BAND_TO_FLOAT[band], band

    except Exception as e:
        logger.warning(f"LLM classification failed for '{paper.title}': {e}")
        cat = _fallback_classification(context)
        paper.classification_signal = "keyword_fallback"
        band = _compute_confidence_band(
            signal="keyword_fallback",
            linked_paper_tier=paper.linked_paper_tier,
            surviving_window_count=surviving_window_count,
            context_source=context_source,
            category=cat,
        )
        return cat, _BAND_TO_FLOAT[band], band
```

- [ ] **Step 4: Update callers of `classify_category` to handle the 3-tuple**

There are 2 callers:
- `process.py:751` inside `process_papers`
- `cli.py:683` inside `reclassify`

Edit `src/ndif_citations/process.py:749-757`:

```python
        # Category classification
        if needs.get("classify") and not skip_llm:
            effective_pdf = pdf_path if pdf_path and pdf_path.exists() else None
            category, confidence, band = classify_category(paper, output_dir, pdf_path=effective_pdf)
            paper.category = category
            paper.category_confidence = confidence
            paper.category_confidence_band = band
            paper.has_classification = category != Category.UNCLASSIFIED

            # Bucket decision after classification (US-D3)
            paper.bucket, paper.reason = _decide_bucket(paper)
```

Edit `src/ndif_citations/cli.py:682-691`:

```python
        new_cat, new_conf, new_band = classify_category(paper, out, pdf_path=pdf_path)

        if new_cat != paper.category or new_band != paper.category_confidence_band:
            signal = paper.classification_signal
            changes.append((paper.title, old_cat, new_cat.value, signal))
            if not dry_run:
                paper.category = new_cat
                paper.category_confidence = new_conf
                paper.category_confidence_band = new_band
                paper.has_classification = new_cat != Category.UNCLASSIFIED
```

- [ ] **Step 5: Run tests**

Run: `pytest tests/test_classify_band_integration.py -v`
Expected: PASS (6 tests)

Run: `pytest tests/ -x -q`
Expected: All tests pass. (If `tests/test_classify_category.py` or `tests/test_cli_reclassify.py` fail because they expect 2-tuple, update them in this step — they're in scope for this task.)

- [ ] **Step 6: Commit**

```bash
git add src/ndif_citations/process.py src/ndif_citations/cli.py \
        tests/test_classify_band_integration.py tests/test_classify_category.py \
        tests/test_cli_reclassify.py
git commit -m "feat(process): classify_category emits Confidence band

Returns (category, float, band). Band is derived from the new
_compute_confidence_band helper using signal label, linked_paper_tier,
surviving window count, and context source. Float is preserved for
backwards compat. Callers (process_papers, reclassify) updated."
```

---

## Task 5: Update `_decide_bucket` to gate on bands

**Files:**
- Modify: `src/ndif_citations/process.py:607-642` (_decide_bucket)
- Test: `tests/test_decide_bucket.py` (extend)

- [ ] **Step 1: Write failing test for MEDIUM band routing**

Append to `tests/test_decide_bucket.py`:

```python
class TestDecideBucketBands:
    """_decide_bucket should gate on Confidence band, not the legacy float."""

    def test_high_band_goes_to_verified(self):
        from ndif_citations.models import Bucket, Confidence
        from ndif_citations.process import _decide_bucket
        from tests.conftest import make_paper
        paper = make_paper(
            year=2024,
            abstract="A real abstract.",
            arxiv_id="2407.14561",
            category=Category.USES_NDIF,
            category_confidence_band=Confidence.HIGH,
        )
        bucket, reason = _decide_bucket(paper)
        assert bucket == Bucket.VERIFIED
        assert reason is None

    def test_medium_band_goes_to_pending_with_medium_reason(self):
        from ndif_citations.models import Bucket, Confidence, PaperReason
        from ndif_citations.process import _decide_bucket
        from tests.conftest import make_paper
        paper = make_paper(
            year=2024,
            abstract="A real abstract.",
            arxiv_id="2407.14561",
            category=Category.USES_NDIF,
            category_confidence_band=Confidence.MEDIUM,
        )
        bucket, reason = _decide_bucket(paper)
        assert bucket == Bucket.PENDING
        assert reason == PaperReason.MEDIUM_CONFIDENCE

    def test_low_band_goes_to_pending_low_confidence(self):
        from ndif_citations.models import Bucket, Confidence, PaperReason
        from ndif_citations.process import _decide_bucket
        from tests.conftest import make_paper
        paper = make_paper(
            year=2024,
            abstract="A real abstract.",
            arxiv_id="2407.14561",
            category=Category.USES_NDIF,
            category_confidence_band=Confidence.LOW,
        )
        bucket, reason = _decide_bucket(paper)
        assert bucket == Bucket.PENDING
        assert reason == PaperReason.LOW_CONFIDENCE

    def test_certain_band_goes_to_verified(self):
        from ndif_citations.models import Bucket, Confidence
        from ndif_citations.process import _decide_bucket
        from tests.conftest import make_paper
        paper = make_paper(
            year=2024,
            abstract="A real abstract.",
            arxiv_id="2407.14561",
            category=Category.REFERENCING,
            category_confidence_band=Confidence.CERTAIN,
        )
        bucket, reason = _decide_bucket(paper)
        assert bucket == Bucket.VERIFIED

# Note: existing tests for stub_metadata / unclassified should still pass —
# they take precedence over the confidence rule.
```

(Add `from ndif_citations.models import Category` to the test file's imports if not already present.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_decide_bucket.py::TestDecideBucketBands -v`
Expected: FAIL — `_decide_bucket` still uses the float threshold.

- [ ] **Step 3: Update `_decide_bucket`**

Edit `src/ndif_citations/process.py:607-642`. Replace the function:

```python
def _decide_bucket(paper: DiscoveredPaper) -> tuple[Bucket, Optional[PaperReason]]:
    """Return (bucket, reason) for a paper based on ordered demotion rules.

    Rules apply IN ORDER; first match wins:
    1. stub_metadata     — year==0 OR no usable abstract OR no identifier
    2. unclassified_*    — Category.UNCLASSIFIED
    3. medium_confidence — band == MEDIUM
    4. low_confidence    — band == LOW

    Papers with band CERTAIN or HIGH pass through to VERIFIED.
    """
    has_any_identifier = bool(
        paper.arxiv_id or paper.doi or paper.s2_paper_id
        or paper.openalex_id or paper.url
    )
    if (
        paper.year == 0
        or not _has_usable_abstract(paper)
        or not has_any_identifier
    ):
        return Bucket.PENDING, PaperReason.STUB_METADATA

    if paper.category == Category.UNCLASSIFIED:
        if paper.unclassified_reason in ("no_keywords_anywhere", "no_evidence_extractable"):
            return Bucket.PENDING, PaperReason.UNCLASSIFIED_NO_KEYWORDS
        return Bucket.PENDING, PaperReason.UNCLASSIFIED_LLM

    if paper.category_confidence_band == Confidence.MEDIUM:
        return Bucket.PENDING, PaperReason.MEDIUM_CONFIDENCE

    if paper.category_confidence_band == Confidence.LOW:
        return Bucket.PENDING, PaperReason.LOW_CONFIDENCE

    return Bucket.VERIFIED, None
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_decide_bucket.py -v`
Expected: All pass — both new tests and the existing ones (stub_metadata, unclassified take precedence).

Run: `pytest tests/ -x -q`
Expected: All tests pass.

- [ ] **Step 5: Commit**

```bash
git add src/ndif_citations/process.py tests/test_decide_bucket.py
git commit -m "feat(process): _decide_bucket gates on Confidence band

Replaces the 0.7 magic threshold. MEDIUM band gets its own pending
reason (medium_confidence) so curators can distinguish 'real but
thin evidence' from 'LLM unavailable, keyword guess'."
```

---

## Task 6: Router — PROTECTED with empty fields → FILL_GAPS

**Files:**
- Modify: `src/ndif_citations/router.py:138-170` (_route_single_paper)
- Test: `tests/test_router_protected_fill_gaps.py` (new file)

- [ ] **Step 1: Write failing tests**

Create `tests/test_router_protected_fill_gaps.py`:

```python
"""Tests for the manual_override → FILL_GAPS routing path (PRD §3)."""
from __future__ import annotations

from ndif_citations.models import Category, DiscoveredPaper
from ndif_citations.router import ProcessingBucket, route_papers
from tests.conftest import make_paper


class TestProtectedFillGaps:
    """manual_override=True papers with empty has_* flags should route
    to FILL_GAPS (not PROTECTED) so the pipeline can fill description,
    thumbnail, affiliations without touching curated values.
    """

    def test_protected_with_no_empty_fields_stays_protected(self):
        existing = make_paper(
            arxiv_id="2407.14561",
            manual_override=True,
            has_summary=True,
            has_classification=True,
            has_thumbnail=True,
            has_affiliations=True,
        )
        discovered = make_paper(arxiv_id="2407.14561")
        decisions = route_papers([discovered], [existing])
        assert decisions[0].bucket == ProcessingBucket.PROTECTED
        assert all(v is False for v in decisions[0].processing_needed.values())

    def test_protected_with_empty_description_routes_to_fill_gaps(self):
        existing = make_paper(
            arxiv_id="2407.14561",
            manual_override=True,
            has_summary=False,  # description is empty
            has_classification=True,
            has_thumbnail=True,
            has_affiliations=True,
        )
        discovered = make_paper(arxiv_id="2407.14561")
        decisions = route_papers([discovered], [existing])
        assert decisions[0].bucket == ProcessingBucket.FILL_GAPS
        assert decisions[0].processing_needed["summary"] is True
        assert decisions[0].processing_needed["classify"] is False
        assert decisions[0].processing_needed["thumbnail"] is False
        assert decisions[0].processing_needed["affiliations"] is False

    def test_protected_with_multiple_empty_fields(self):
        existing = make_paper(
            arxiv_id="2407.14561",
            manual_override=True,
            has_summary=False,
            has_classification=False,
            has_thumbnail=False,
            has_affiliations=False,
        )
        discovered = make_paper(arxiv_id="2407.14561")
        decisions = route_papers([discovered], [existing])
        assert decisions[0].bucket == ProcessingBucket.FILL_GAPS
        assert all(
            decisions[0].processing_needed[f] is True
            for f in ("summary", "classify", "thumbnail", "affiliations")
        )
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_router_protected_fill_gaps.py -v`
Expected: 2 FAIL (the second and third — they expect FILL_GAPS but get PROTECTED). 1 PASS.

- [ ] **Step 3: Update `_route_single_paper`**

Edit `src/ndif_citations/router.py:138-142` (the PROTECTED branch). Replace:

```python
    # PROTECTED: Manual override — but allow fill-gaps for empty fields
    if existing.manual_override:
        needs = {
            "summary":      not existing.has_summary,
            "classify":     not existing.has_classification,
            "thumbnail":    not existing.has_thumbnail,
            "affiliations": not existing.has_affiliations,
        }
        if any(needs.values()):
            logger.debug(f"PROTECTED→FILL_GAPS: {paper.title[:50]}... (needs: {needs})")
            return RoutingDecision(paper, ProcessingBucket.FILL_GAPS, existing, needs)
        logger.debug(f"PROTECTED: {paper.title[:50]}...")
        return RoutingDecision(paper, ProcessingBucket.PROTECTED, existing, _all_false())
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_router_protected_fill_gaps.py -v`
Expected: PASS (3 tests).

Run: `pytest tests/test_router.py -v`
Expected: All pass. (If an existing test expects PROTECTED for a manual_override paper with empty fields, update it to expect FILL_GAPS.)

Run: `pytest tests/ -x -q`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/ndif_citations/router.py tests/test_router_protected_fill_gaps.py tests/test_router.py
git commit -m "feat(router): allow fill-gaps on manual_override papers

Previously manual_override=True always routed to PROTECTED, meaning
empty description/thumbnail/affiliations stayed empty forever. Now
routes to FILL_GAPS when any has_* flag is False, letting the
pipeline fill gaps while still protecting curated fields (the guard
lives in process.py — next task)."
```

---

## Task 7: Process — guard FILL_GAPS to skip non-empty fields on manual_override

**Files:**
- Modify: `src/ndif_citations/process.py:704-784` (process_papers main loop)
- Test: `tests/test_process_fill_gaps_protect.py` (new file)

- [ ] **Step 1: Write failing test**

Create `tests/test_process_fill_gaps_protect.py`:

```python
"""Process.py guard: when manual_override=True and routing is FILL_GAPS,
only write to fields that are currently empty.

This is the second half of the PROTECTED → FILL_GAPS feature.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from ndif_citations.models import Bucket, Category, Confidence, DiscoveredPaper
from ndif_citations.process import process_papers
from ndif_citations.router import ProcessingBucket, RoutingDecision
from tests.conftest import make_paper


class TestFillGapsProtectsManualFields:
    """On a manual_override paper routed to FILL_GAPS, the pipeline must
    only write to fields where the existing value is empty.
    """

    def test_description_is_preserved_when_present(self, tmp_path, monkeypatch):
        # Set up a manual_override paper that has a description already but
        # missing thumbnail.
        existing = make_paper(
            arxiv_id="2407.14561",
            description="Emma's curated summary that must NOT be overwritten.",
            manual_override=True,
            has_summary=True,        # description is set
            has_classification=True, # category is set
            has_thumbnail=False,     # missing → should be filled
            has_affiliations=True,
        )
        decision = RoutingDecision(
            paper=existing,
            bucket=ProcessingBucket.FILL_GAPS,
            existing_paper=existing,
            processing_needed={"summary": False, "classify": False,
                               "thumbnail": True, "affiliations": False},
        )
        # Patch the heavy bits
        monkeypatch.setattr("ndif_citations.process.get_cached_pdf",
                            lambda p, o: None)  # no PDF, thumbnail no-op
        monkeypatch.setattr("ndif_citations.process.generate_summary",
                            lambda p: "PIPELINE-GENERATED summary (must not appear)")
        result = process_papers([decision], tmp_path)
        assert len(result) == 1
        assert result[0].description == "Emma's curated summary that must NOT be overwritten."

    def test_summary_is_filled_when_empty_on_manual_override(self, tmp_path, monkeypatch):
        existing = make_paper(
            arxiv_id="2407.14561",
            description="",          # empty
            manual_override=True,
            has_summary=False,
            has_classification=True,
            has_thumbnail=True,
            has_affiliations=True,
        )
        decision = RoutingDecision(
            paper=existing,
            bucket=ProcessingBucket.FILL_GAPS,
            existing_paper=existing,
            processing_needed={"summary": True, "classify": False,
                               "thumbnail": False, "affiliations": False},
        )
        monkeypatch.setattr("ndif_citations.process.get_cached_pdf",
                            lambda p, o: None)
        monkeypatch.setattr("ndif_citations.process.generate_summary",
                            lambda p: "NEW summary from pipeline")
        result = process_papers([decision], tmp_path)
        assert result[0].description == "NEW summary from pipeline"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_process_fill_gaps_protect.py -v`
Expected: FAIL — `process_papers` writes regardless of manual_override.

- [ ] **Step 3: Add the guard in `process_papers`**

Edit `src/ndif_citations/process.py:704-784`. Add a guard inside each write site that checks `is_protected_manual = (decision.existing_paper is not None and decision.existing_paper.manual_override)` and skips when the existing field is non-empty. Concretely, modify the summary write (line ~737) and similar paths:

```python
        is_protected_manual = (
            decision.existing_paper is not None
            and decision.existing_paper.manual_override
        )

        # Summary generation
        if needs.get("summary") and not skip_llm:
            if is_protected_manual and decision.existing_paper.description:
                pass  # don't overwrite curated description
            else:
                paper.description = generate_summary(paper)
                paper.has_summary = bool(paper.description)

        # Discard check ...
        # (unchanged)

        # Category classification
        if needs.get("classify") and not skip_llm:
            if is_protected_manual and decision.existing_paper.category != Category.UNCLASSIFIED:
                pass  # curated category stays
            else:
                effective_pdf = pdf_path if pdf_path and pdf_path.exists() else None
                category, confidence, band = classify_category(paper, output_dir, pdf_path=effective_pdf)
                paper.category = category
                paper.category_confidence = confidence
                paper.category_confidence_band = band
                paper.has_classification = category != Category.UNCLASSIFIED
                paper.bucket, paper.reason = _decide_bucket(paper)

        # Thumbnail extraction
        if needs.get("thumbnail"):
            if is_protected_manual and decision.existing_paper.image:
                paper.image = decision.existing_paper.image  # keep curated
            elif pdf_path and pdf_path.exists():
                image_filename = f"{slugify(paper.title)}.png"
                image_path = output_dir / "images" / image_filename
                if not image_path.exists():
                    extracted = extract_thumbnail(paper, output_dir, pdf_path=pdf_path)
                    if extracted:
                        paper.image = extracted
            paper.has_thumbnail = bool(paper.image)

        # Affiliation extraction
        if needs.get("affiliations") and not paper.affiliations and pdf_path and pdf_path.exists():
            if is_protected_manual and decision.existing_paper.affiliations:
                paper.affiliations = decision.existing_paper.affiliations
            else:
                from ndif_citations.utils import extract_affiliations_from_pdf
                try:
                    affs = extract_affiliations_from_pdf(pdf_path, paper.authors or "")
                    if affs:
                        paper.affiliations = ", ".join(affs)
                except Exception as e:
                    logger.debug(f"Affiliation extraction failed: {e}")
        paper.has_affiliations = bool(paper.affiliations)
```

Also: when `is_protected_manual=True`, the existing paper's `bucket`, `category`, `description`, etc. need to seed `paper` at the START of the loop iteration — otherwise the routing-decision's discovered `paper` (which may have minimal fields) overwrites curated values. Add at the top of the loop body (before the bucket SKIP/PROTECTED branch around line 711):

```python
        # When fill-gaps on a manual_override paper, hydrate the discovered
        # paper with the curated existing state so we preserve everything we
        # don't explicitly refill below.
        if (
            bucket == ProcessingBucket.FILL_GAPS
            and decision.existing_paper is not None
            and decision.existing_paper.manual_override
        ):
            existing = decision.existing_paper
            paper.title = existing.title or paper.title
            paper.authors = existing.authors or paper.authors
            paper.affiliations = existing.affiliations or paper.affiliations
            paper.venue = existing.venue or paper.venue
            paper.year = existing.year or paper.year
            paper.description = existing.description or paper.description
            paper.category = existing.category
            paper.category_confidence = existing.category_confidence
            paper.category_confidence_band = existing.category_confidence_band
            paper.bucket = existing.bucket
            paper.reason = existing.reason
            paper.reason_detail = existing.reason_detail
            paper.image = existing.image or paper.image
            paper.project_url = existing.project_url or paper.project_url
            paper.url = existing.url or paper.url
            paper.pdf_url = existing.pdf_url or paper.pdf_url
            paper.manual_override = True
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_process_fill_gaps_protect.py -v`
Expected: PASS (2 tests).

Run: `pytest tests/ -x -q`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/ndif_citations/process.py tests/test_process_fill_gaps_protect.py
git commit -m "feat(process): protect curated fields in FILL_GAPS on manual_override

When the router routes a manual_override paper to FILL_GAPS, each
write site now checks the existing value and skips when non-empty.
Also hydrates the discovered paper with curated state at the top of
the loop so untouched fields persist across runs."
```

---

## Task 8: Define `_EDITABLE_FIELDS` schema for the `edit` command

**Files:**
- Create: `src/ndif_citations/edit_schema.py` (new file — keeps cli.py thin)

- [ ] **Step 1: Add the schema module**

Create `src/ndif_citations/edit_schema.py`:

```python
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
    EditableField("title",          "Title",          str,                "Paper title"),
    EditableField("authors",        "Authors",        str,                "Comma-separated author list"),
    EditableField("affiliations",   "Affiliations",   str,                "Comma-separated institutions"),
    EditableField("venue",          "Venue",          str,                "Conference / journal / 'ArXiv YYYY'"),
    EditableField("year",           "Year",           _parse_int,         "Publication year"),
    EditableField("category",       "Category",       _parse_category,    "uses_ndif | uses_nnsight | referencing | unclassified"),
    EditableField("description",    "Description",    str,                "1-3 sentence website summary"),
    EditableField("url",            "URL",            str,                "Landing page URL"),
    EditableField("pdf_url",        "PDF URL",        _parse_optional_str,"Direct PDF link (or empty to clear)"),
    EditableField("project_url",    "Project URL",    _parse_optional_str,"GitHub or project page (or empty to clear)"),
    EditableField("image",          "Image path",     _parse_optional_str,"e.g. /images/Slug.png (or empty to clear)"),
    EditableField("bucket",         "Bucket",         _parse_bucket,      "pending | verified | discarded"),
    EditableField("reason",         "Reason",         _parse_reason,      "PaperReason value or empty for none"),
    EditableField("reason_detail",  "Reason detail",  _parse_optional_str,"Free-text supplement"),
    EditableField("peer_reviewed",  "Peer reviewed",  _parse_bool,        "yes/no"),
    EditableField("abstract",       "Abstract",       _parse_optional_str,"Full abstract text"),
]


def get_field(name: str) -> Optional[EditableField]:
    """Lookup field by name (case-insensitive). Returns None if not editable."""
    name_lower = name.lower()
    for f in EDITABLE_FIELDS:
        if f.name.lower() == name_lower:
            return f
    return None
```

- [ ] **Step 2: Smoke test the schema**

Append to `tests/test_cli_edit.py` (file will be created in Task 9, but this is fine as a smoke import test in a new file):

Create `tests/test_edit_schema.py`:

```python
"""Smoke tests for the EDITABLE_FIELDS schema."""
from ndif_citations.edit_schema import EDITABLE_FIELDS, get_field
from ndif_citations.models import Bucket, Category, PaperReason


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
```

Run: `pytest tests/test_edit_schema.py -v`
Expected: PASS (6 tests).

- [ ] **Step 3: Commit**

```bash
git add src/ndif_citations/edit_schema.py tests/test_edit_schema.py
git commit -m "feat(edit): add EDITABLE_FIELDS schema

Curated list of 16 fields a curator can override via the new edit
command. Identifiers and computed fields are intentionally excluded
to prevent dedup-breaking edits."
```

---

## Task 9: Implement `edit <id>` CLI command

**Files:**
- Modify: `src/ndif_citations/cli.py` (add new `edit` command after `discard`)
- Test: `tests/test_cli_edit.py` (new file)

This task adds the menu-driven interactive command. UX:

```
$ ndif-citations edit 2407.14561

Paper: NNsight and NDIF: Democratizing Access...

  [1] Title           NNsight and NDIF: Democratizing Access...
  [2] Authors         Jaden Fiotto-Kaufman, ...
  [3] Affiliations    Northeastern University, MIT, ...
  ...

Choose field number (q to save+quit, a to abort, ? for help): 4
Current Venue: ICLR 2025
New value (empty=keep): ICLR 2025 (Spotlight)

[updated] Venue: ICLR 2025 → ICLR 2025 (Spotlight)

Choose field number ...: q

Summary of changes:
  venue: ICLR 2025 → ICLR 2025 (Spotlight)

manual_override will be set to True.
Save? [Y/n]: y
[OK] Saved.
```

- [ ] **Step 1: Write the failing tests**

Create `tests/test_cli_edit.py`:

```python
"""Tests for the `edit` CLI command."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from ndif_citations.cli import cli
from ndif_citations.models import Bucket, Category, DiscoveredPaper
from tests.conftest import make_paper


def _write_state(output_dir: Path, papers: list[DiscoveredPaper]) -> None:
    """Write a minimal research-papers-full.json with the 3-bucket structure."""
    state = {"pending": [], "verified": [], "discarded": []}
    for p in papers:
        state[p.bucket.value].append(p.to_full_dict())
    (output_dir / "research-papers-full.json").write_text(
        json.dumps(state, default=str)
    )


class TestEditCommand:
    """Cover the main interactive paths."""

    def test_edit_single_field_via_menu(self, tmp_path):
        paper = make_paper(
            title="Original",
            arxiv_id="2407.14561",
            venue="ICLR 2025",
            bucket=Bucket.VERIFIED,
        )
        _write_state(tmp_path, [paper])

        # Stdin: choose field 4 (Venue), input new value, q to save, y to confirm
        # Field numbering: 1=title 2=authors 3=affiliations 4=venue 5=year ...
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["edit", "2407.14561", "--output-dir", str(tmp_path)],
            input="4\nICLR 2025 (Spotlight)\nq\ny\n",
        )
        assert result.exit_code == 0, result.output

        # Reload and verify
        state = json.loads((tmp_path / "research-papers-full.json").read_text())
        verified = state["verified"]
        assert len(verified) == 1
        assert verified[0]["venue"] == "ICLR 2025 (Spotlight)"
        assert verified[0]["manual_override"] is True

    def test_edit_abort_does_not_save(self, tmp_path):
        paper = make_paper(
            title="Original",
            arxiv_id="2407.14561",
            venue="ICLR 2025",
            bucket=Bucket.VERIFIED,
        )
        _write_state(tmp_path, [paper])

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["edit", "2407.14561", "--output-dir", str(tmp_path)],
            input="4\nICLR 2025 (Spotlight)\na\n",
        )
        assert result.exit_code == 0

        state = json.loads((tmp_path / "research-papers-full.json").read_text())
        assert state["verified"][0]["venue"] == "ICLR 2025"
        assert state["verified"][0]["manual_override"] is False

    def test_edit_paper_not_found(self, tmp_path):
        _write_state(tmp_path, [])

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["edit", "9999.99999", "--output-dir", str(tmp_path)],
            input="",
        )
        # Should exit cleanly with a warning message, not crash
        assert "not found" in result.output.lower()

    def test_edit_with_field_flag_one_shot(self, tmp_path):
        """edit <id> --set venue=ICML\\ 2025 should bypass interactive prompt."""
        paper = make_paper(
            title="Original",
            arxiv_id="2407.14561",
            venue="ICLR 2025",
            bucket=Bucket.VERIFIED,
        )
        _write_state(tmp_path, [paper])

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["edit", "2407.14561",
             "--set", "venue=ICML 2025",
             "--set", "category=uses_nnsight",
             "--output-dir", str(tmp_path),
             "--yes"],  # confirm flag for non-interactive
        )
        assert result.exit_code == 0, result.output

        state = json.loads((tmp_path / "research-papers-full.json").read_text())
        # Paper should now be in verified with updated fields
        all_papers = state["verified"] + state["pending"] + state["discarded"]
        target = [p for p in all_papers if p.get("arxiv_id") == "2407.14561"][0]
        assert target["venue"] == "ICML 2025"
        assert target["category"] == "uses_nnsight"
        assert target["manual_override"] is True

    def test_edit_dry_run_does_not_write(self, tmp_path):
        paper = make_paper(
            title="Original",
            arxiv_id="2407.14561",
            venue="ICLR 2025",
            bucket=Bucket.VERIFIED,
        )
        _write_state(tmp_path, [paper])

        runner = CliRunner()
        result = runner.invoke(
            cli,
            ["edit", "2407.14561",
             "--set", "venue=ICML 2025",
             "--dry-run",
             "--output-dir", str(tmp_path),
             "--yes"],
        )
        assert result.exit_code == 0

        state = json.loads((tmp_path / "research-papers-full.json").read_text())
        assert state["verified"][0]["venue"] == "ICLR 2025"  # unchanged
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_cli_edit.py -v`
Expected: FAIL — `edit` command doesn't exist yet.

- [ ] **Step 3: Add the `edit` CLI command**

Edit `src/ndif_citations/cli.py`. After the existing `discard` command (line ~842) and before `if __name__ == "__main__":` (line ~844), add:

```python
@cli.command()
@click.argument("paper_id")
@click.option("--output-dir", "-o", default=None, help="Custom output directory")
@click.option("--set", "set_pairs", multiple=True,
              help="One-shot field=value (repeatable). Bypasses the interactive prompt.")
@click.option("--dry-run", is_flag=True, help="Preview changes without writing files")
@click.option("--yes", "auto_confirm", is_flag=True,
              help="Skip the final 'Save? [Y/n]' confirmation (for scripting)")
def edit(paper_id: str, output_dir: str | None,
         set_pairs: tuple[str, ...], dry_run: bool, auto_confirm: bool) -> None:
    """Interactively edit a paper's fields. Sets manual_override=True on save.

    Two modes:
      Interactive: `edit <id>` — menu-driven prompt loop.
      One-shot:    `edit <id> --set field=value --set field2=value2 --yes`
    """
    from ndif_citations.edit_schema import EDITABLE_FIELDS, get_field
    from ndif_citations.models import PipelineRun
    from ndif_citations.output import load_existing_papers, write_outputs
    from ndif_citations.process import _decide_bucket

    out = config.get_output_dir(output_dir)
    papers = load_existing_papers(out)
    idx, paper = _resolve_paper(papers, paper_id)
    if paper is None:
        console.print(f"[bold yellow]WARNING:[/bold yellow] Paper not found for ID {paper_id!r}")
        return

    console.print(f"\n[bold cyan]Edit paper:[/bold cyan] {paper.title[:80]}\n")

    changes: list[tuple[str, str, str]] = []  # (field_name, old, new)

    if set_pairs:
        # One-shot mode: parse --set entries
        for pair in set_pairs:
            if "=" not in pair:
                console.print(f"  [red]Bad --set value (need 'field=value'): {pair!r}[/red]")
                raise SystemExit(1)
            name, _, raw_value = pair.partition("=")
            field = get_field(name.strip())
            if field is None:
                console.print(f"  [red]Unknown field: {name!r}[/red]")
                raise SystemExit(1)
            try:
                new_value = field.parse(raw_value)
            except Exception as e:
                console.print(f"  [red]Failed to parse {name}={raw_value!r}: {e}[/red]")
                raise SystemExit(1)
            old_value = getattr(paper, field.name)
            if new_value != old_value:
                setattr(paper, field.name, new_value)
                changes.append((field.name, str(old_value), str(new_value)))
    else:
        # Interactive mode
        while True:
            console.print("\nEditable fields:")
            for i, f in enumerate(EDITABLE_FIELDS, 1):
                current = getattr(paper, f.name)
                display = str(current) if current not in (None, "", 0) else "[dim](none)[/dim]"
                if len(display) > 60:
                    display = display[:57] + "..."
                console.print(f"  [{i:2}] {f.label:<16} {display}")
            choice = click.prompt(
                "\nChoose field number (q=save+quit, a=abort)",
                default="q", show_default=False,
            )
            choice = choice.strip().lower()
            if choice == "a":
                console.print("[yellow]Aborted, no changes saved.[/yellow]")
                return
            if choice == "q":
                break
            try:
                n = int(choice)
                if not 1 <= n <= len(EDITABLE_FIELDS):
                    raise ValueError
            except ValueError:
                console.print("  [red]Bad choice — pick a number, q, or a.[/red]")
                continue
            field = EDITABLE_FIELDS[n - 1]
            current = getattr(paper, field.name)
            console.print(f"\n  Current {field.label}: [cyan]{current!r}[/cyan]")
            console.print(f"  Hint: {field.description}")
            raw = click.prompt("  New value (empty=keep)", default="", show_default=False)
            if not raw:
                continue
            try:
                new_value = field.parse(raw)
            except Exception as e:
                console.print(f"  [red]Failed to parse: {e}[/red]")
                continue
            old_value = current
            setattr(paper, field.name, new_value)
            changes.append((field.name, str(old_value), str(new_value)))
            console.print(f"  [green][updated] {field.label}: {old_value} -> {new_value}[/green]")

    if not changes:
        console.print("\n[dim]No changes made.[/dim]")
        return

    # Re-derive has_* flags
    paper.has_summary = bool(paper.description)
    paper.has_classification = paper.category != Category.UNCLASSIFIED
    paper.has_thumbnail = bool(paper.image)
    paper.has_affiliations = bool(paper.affiliations)
    paper.manual_override = True

    # Re-run _decide_bucket if user didn't explicitly set bucket
    if not any(name == "bucket" for name, _, _ in changes):
        new_bucket, new_reason = _decide_bucket(paper)
        if new_bucket != paper.bucket:
            paper.bucket = new_bucket
            paper.reason = new_reason

    # Print diff
    console.print("\n[bold]Summary of changes:[/bold]")
    for name, old, new in changes:
        console.print(f"  {name}: [yellow]{old}[/yellow] -> [green]{new}[/green]")
    console.print(f"\nmanual_override will be set to True.")

    if dry_run:
        console.print("\n[dim]Dry run — no files written.[/dim]")
        return

    if not auto_confirm:
        if not click.confirm("Save?", default=True):
            console.print("[yellow]Aborted, no changes saved.[/yellow]")
            return

    run = PipelineRun()
    write_outputs(papers, out, run)
    console.print("\n[green][OK] Saved.[/green]")
```

Place this after the `discard` command (around line 842) and before the `if __name__ == "__main__":` block.

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_cli_edit.py -v`
Expected: PASS (5 tests).

Run: `pytest tests/ -x -q`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/ndif_citations/cli.py tests/test_cli_edit.py
git commit -m "feat(cli): add edit <id> interactive command

Menu-driven curator UX for overriding any of 16 curated fields on
a single paper. Auto-sets manual_override=True, re-derives has_*
flags, re-runs _decide_bucket. Supports --set field=value one-shot
mode and --dry-run preview."
```

---

## Task 10: Update low-confidence report to use bands

**Files:**
- Modify: `src/ndif_citations/output.py:375-379`
- Test: `tests/test_output_report.py` (extend)

The existing low_confidence report uses `< 0.7`. Update it to enumerate MEDIUM + LOW bands separately, since they now have distinct meanings.

- [ ] **Step 1: Write failing test**

Append to `tests/test_output_report.py`:

```python
class TestLowConfidenceReportUsesBands:
    """The low_confidence list in PipelineRun should now distinguish
    MEDIUM and LOW bands (was: any confidence < 0.7).
    """

    def test_medium_band_appears_in_low_confidence_list(self, tmp_path):
        from ndif_citations.models import (
            Bucket, Category, Confidence, DiscoveredPaper, PipelineRun
        )
        from ndif_citations.output import write_outputs
        from tests.conftest import make_paper

        paper = make_paper(
            arxiv_id="2407.14561",
            year=2024,
            category=Category.USES_NDIF,
            category_confidence_band=Confidence.MEDIUM,
            bucket=Bucket.PENDING,  # MEDIUM rolls to pending
        )
        run = PipelineRun()
        (tmp_path / "images").mkdir(parents=True, exist_ok=True)
        write_outputs([paper], tmp_path, run)
        # Should show up labeled medium
        assert any("medium" in line.lower() for line in run.low_confidence)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_output_report.py::TestLowConfidenceReportUsesBands -v`
Expected: FAIL.

- [ ] **Step 3: Update `write_outputs`**

Edit `src/ndif_citations/output.py:375-379`:

```python
    from ndif_citations.models import Category, Confidence
    run.low_confidence = [
        f'"{p.title}" -- classified as "{p.category.value}" (band: {p.category_confidence_band.value})'
        for p in papers
        if p.category_confidence_band in (Confidence.MEDIUM, Confidence.LOW)
        and p.category != Category.UNCLASSIFIED
    ]
```

- [ ] **Step 4: Run tests**

Run: `pytest tests/test_output_report.py -v`
Expected: All pass.

Run: `pytest tests/ -x -q`
Expected: All pass.

- [ ] **Step 5: Commit**

```bash
git add src/ndif_citations/output.py tests/test_output_report.py
git commit -m "fix(output): low-confidence report uses Confidence bands

Replaces the implicit < 0.7 threshold with explicit MEDIUM/LOW band
membership. Report labels each entry with its band so curators can
sort review priority."
```

---

## Task 11: Update CLI documentation strings + AGENTS.md (if exists)

**Files:**
- Modify: `README.md` (CLI table near line 30)

- [ ] **Step 1: Update README CLI table**

Edit `README.md`. In the commands table, add a row for `edit`:

```markdown
| `python -m ndif_citations edit <id>` | Interactively override fields on a single paper (sets manual_override) |
| `python -m ndif_citations edit <id> --set field=value` | One-shot field edit, scriptable |
```

Also note in the README's "Output" section that `category_confidence_band` is the new band field; the float `category_confidence` is kept for backwards compat.

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: document the new edit command and confidence band field"
```

---

## Self-Review

**Spec coverage:**
- ✅ Feature 2 (Confidence rework, categorical bands): Tasks 1-5, 10
- ✅ Feature 1b (Router fill-gaps for protected): Tasks 6-7
- ✅ Feature 1a (Interactive edit command): Tasks 8-9
- ✅ Backwards-compat for legacy float: Task 1 migration logic
- ✅ MEDIUM_CONFIDENCE new bucket reason: Task 2
- ✅ Curated editable field schema (no IDs): Task 8
- ✅ Auto manual_override=True on edit: Task 9
- ✅ Re-derive has_* flags + re-run _decide_bucket: Task 9
- ✅ --set one-shot and --dry-run modes: Task 9
- ✅ Documentation update: Task 11

**No placeholders:** All steps have concrete code blocks, expected outputs, exact commands. No TBDs.

**Type consistency:** `_compute_confidence_band` keyword signature is consistent across the test stub (Task 3) and implementation (Task 3). `classify_category` returns `(Category, float, Confidence)` consistently in Tasks 3-4 and at every caller updated in Task 4. `EditableField` dataclass is defined in Task 8 and consumed in Task 9.

**Out of scope (deferred to next PR):**
- Fix the broken `add <url>` command — separate PR; the `edit` command's `--new` mode could replace it but adding `--new` to `edit` plus deleting `add` cleanly is its own PR with migration messaging
- Dead code cleanup (`_write_csv`, `calculate_image_score`, `download_pdf` legacy, `process_papers_legacy`)
- Section-aware classification prompts (feature 4 from the user's roadmap)
- Better LLM for summary generation (feature 4)
- README/CODEBASE_CONTEXT.md CSV claim correction
- `__version__` mismatch

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-19-confidence-and-edit.md`. Two execution options:

1. **Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration
2. **Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints

Which approach?
