# Paper Curation Workspace UX Implementation Plan (Plan B)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement task-by-task. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Turn the Paper detail Sheet into a self-contained verify-and-fix workspace: see what's missing, view the figure (lightbox) and cached PDF, read the abstract + the classifier's evidence, and step through the queue — without leaving the app.

**Architecture:** Small backend additions (cached-PDF path helper + `/api/papers/{id}/pdf` route; `get_paper` enriched with `missing`+`has_pdf`; persisted evidence: model fields + `compute_context` helper in process.py + `backfill-evidence` CLI) feed frontend additions in `PaperSheet`/`Papers` that ride the existing `GET /papers/{id}` detail fetch.

**Tech Stack:** FastAPI + Pydantic v2; Click CLI; pytest+monkeypatch. Frontend: Vite/React/TS, TanStack Query, shadcn (Dialog/Sheet), Tailwind. Verify FE via `cd web && bun run build` (tsc) + cmux browser.

**Spec:** `docs/superpowers/specs/2026-06-06-paper-curation-ux-design.md`. **Prereq:** Plan A merged (enrichment) — done. Baseline: 893 tests green, branch `feat/citations-web-app`.

**Key facts (verified):** `PaperSheet({paperId, onClose})` at `web/src/components/papers/PaperSheet.tsx:527`; image `<img src={/api/images/${basename}}>` @617 with "No thumbnail" @626; abstract already shown (clamped) @722; external links row @734. `papers_svc.get_paper` returns `to_full_dict()`; `_compute_missing` exists. `pdf_cache.get_cached_pdf` builds `out/pdfs/{arxiv-ID|doi-slug|slug}.pdf` (105 cached). `process.py` classify (~line 339-377) computes `surviving_windows`+`context_source`. Images route `server/routers/images.py` has the path-traversal guard to mirror. `PaperRow.missing` already exists; `PaperDetail` has no `id` (use arxiv/doi/merge_key). authors/affiliations are comma-strings.

---

### Task 1: Model — evidence fields

**Files:** Modify `src/ndif_citations/models.py` (DiscoveredPaper); Test `tests/test_evidence_model.py`

- [ ] **Step 1: failing test**
```python
# tests/test_evidence_model.py
from ndif_citations.models import DiscoveredPaper

def test_evidence_fields_default_and_roundtrip():
    p = DiscoveredPaper(title="X")
    assert p.ndif_context_windows == [] and p.context_source == "none"
    p.ndif_context_windows = ["...nnsight..."]; p.context_source = "pdf"
    d = p.model_dump(mode="json")
    assert d["ndif_context_windows"] == ["...nnsight..."] and d["context_source"] == "pdf"
    assert DiscoveredPaper.model_validate(d).context_source == "pdf"
```
- [ ] **Step 2: run** `pytest tests/test_evidence_model.py -q` → FAIL.
- [ ] **Step 3: add fields** in `DiscoveredPaper` (near `enrichment_provenance`):
```python
    # NDIF evidence the classifier used (persisted at classify time for the UI).
    ndif_context_windows: list[str] = Field(default_factory=list)
    context_source: str = "none"  # "pdf" | "abstract" | "none"
```
- [ ] **Step 4: run** `pytest tests/test_evidence_model.py -q` → PASS.
- [ ] **Step 5: full suite** `pytest tests/ -q` → all pass (update `tests/fixtures/golden/research-papers-full.json` ONLY if a strict parity test requires the new keys; add `"ndif_context_windows": [], "context_source": "none"` to each golden paper if so — verify that's the only change).
- [ ] **Step 6: commit** `git commit -am "feat(model): persist ndif_context_windows + context_source"`

---

### Task 2: `compute_context` helper + classify persists it (behavior-preserving)

**Files:** Modify `src/ndif_citations/process.py`; Test `tests/test_compute_context.py`

First READ `process.py` ~lines 335-380: it builds `context = extract_ndif_context(pdf_path, ...)` or abstract fallback, sets `context_source` ("none"/"abstract"/"pdf"), `windows = context.split("\n---\n")`, `surviving_windows, prefilter_signal = _apply_prefilters(windows, paper)`, `surviving_window_count = len(surviving_windows)`.

- [ ] **Step 1: failing test**
```python
# tests/test_compute_context.py
from ndif_citations import process
from tests.conftest import make_paper

def test_compute_context_returns_windows_source_signal(monkeypatch, tmp_path):
    paper = make_paper(abstract="we use nnsight to trace activations. " * 5)
    monkeypatch.setattr(process, "get_cached_pdf", lambda p, o: None)  # force abstract path
    windows, source, signal = process.compute_context(paper, tmp_path)
    assert isinstance(windows, list) and source in ("pdf", "abstract", "none")
```
- [ ] **Step 2: run** → FAIL (`compute_context` missing).
- [ ] **Step 3: refactor** — extract the context-building block (PDF resolve → extract_ndif_context → split → `_apply_prefilters`) from the classify function into:
```python
def compute_context(paper, out) -> tuple[list[str], str, "Optional[str]"]:
    """Return (surviving_windows, context_source, prefilter_signal). No LLM.
    Mirrors the classify path so backfill and classify can't diverge."""
    # ... moved verbatim from classify: resolve cached pdf, extract_ndif_context,
    # set context_source, split windows, _apply_prefilters ...
    return surviving_windows, context_source, prefilter_signal
```
Then in the classify function, REPLACE the moved block with `surviving_windows, context_source, prefilter_signal = compute_context(paper, out)` and **persist**: `paper.ndif_context_windows = surviving_windows; paper.context_source = context_source`. Keep `surviving_window_count = len(surviving_windows)` and all downstream band logic IDENTICAL.
- [ ] **Step 4: run** `pytest tests/test_compute_context.py tests/ -q` → PASS (the existing classify/band tests are the parity guard — they must stay green, proving the refactor is behavior-preserving).
- [ ] **Step 5: commit** `git commit -am "refactor(process): extract compute_context; classify persists evidence windows"`

---

### Task 3: `backfill-evidence` CLI

**Files:** Modify `src/ndif_citations/cli.py`; Test `tests/test_cli_backfill_evidence.py`

- [ ] **Step 1: failing test**
```python
# tests/test_cli_backfill_evidence.py
import json
from click.testing import CliRunner
from ndif_citations.cli import cli
from ndif_citations import cli as cli_mod

def _cat(out, papers):
    out.mkdir(parents=True, exist_ok=True)
    (out / "research-papers-full.json").write_text(json.dumps({"verified": papers, "pending": [], "discarded": []}))

def test_backfill_evidence_populates(tmp_path, monkeypatch):
    out = tmp_path / "output"
    _cat(out, [{"title": "P", "arxiv_id": "2401.1", "abstract": "we use nnsight"}])
    # stub compute_context so no PDF/network needed
    monkeypatch.setattr("ndif_citations.process.compute_context",
                        lambda paper, o: (["window about nnsight"], "abstract", None))
    res = CliRunner().invoke(cli, ["backfill-evidence", "-o", str(out)])
    assert res.exit_code == 0, res.output
    data = json.loads((out / "research-papers-full.json").read_text())
    assert data["verified"][0]["ndif_context_windows"] == ["window about nnsight"]
    assert data["verified"][0]["context_source"] == "abstract"
```
- [ ] **Step 2: run** → FAIL.
- [ ] **Step 3: implement** (mirror `re-enrich`/`reclassify` structure; no LLM):
```python
@cli.command(name="backfill-evidence")
@click.option("--ids", default=None)
@click.option("--output-dir", "-o", default=None)
@click.option("--dry-run", is_flag=True)
def backfill_evidence(ids, output_dir, dry_run):
    """Populate ndif_context_windows/context_source for existing papers (no LLM)."""
    from ndif_citations.output import load_existing_papers, write_outputs
    from ndif_citations import config as cfg, process
    from ndif_citations.models import PipelineRun
    _setup_logging(verbose=True)
    out = cfg.get_output_dir(output_dir)
    papers = load_existing_papers(out)
    targets = papers
    if ids:
        wanted = {x.strip() for x in ids.split(",") if x.strip()}
        targets = [p for p in papers if p.arxiv_id in wanted or p.doi in wanted or p.url in wanted]
    n = 0
    for p in targets:
        try:
            windows, source, _ = process.compute_context(p, out)
        except Exception as e:
            console.print(f"  [red]ERROR[/red] {p.title[:50]!r}: {e}"); continue
        if windows or source != "none":
            if not dry_run:
                p.ndif_context_windows = windows; p.context_source = source
            n += 1
    console.print(f"{n} papers with evidence" + (" (dry-run)" if dry_run else ""))
    if not dry_run and n:
        write_outputs(papers, out, PipelineRun()); console.print("[green]Catalog written.[/green]")
```
- [ ] **Step 4: run** `pytest tests/test_cli_backfill_evidence.py tests/ -q` → PASS.
- [ ] **Step 5: commit** `git commit -am "feat(cli): backfill-evidence command"`

---

### Task 4: cached-PDF path helper + `GET /api/papers/{id}/pdf`

**Files:** Modify `src/ndif_citations/pdf_cache.py`, `src/ndif_citations/server/routers/papers.py`; Test `tests/test_pdf_cache_path.py`, `tests/test_api_pdf.py`

- [ ] **Step 1: failing tests** — `cached_pdf_path` (hit→path, miss→None, no download) + the route (200 `application/pdf` for a fixture file under `out/pdfs/`, 404 when absent, traversal id rejected). Write both test files; use a `tmp_path/output/pdfs/arxiv-2401.1.pdf` fixture written with `%PDF` bytes and a fixture catalog so `resolve(out, id)` finds the paper. (Mirror `tests/` patterns for the API client/`TestClient`.)
- [ ] **Step 2: run** → FAIL.
- [ ] **Step 3a:** in `pdf_cache.py` add:
```python
def cached_pdf_path(paper, output_dir) -> "Path | None":
    """Path to the already-cached PDF, or None. No mkdir, no download."""
    cache_dir = output_dir / "pdfs"
    if paper.arxiv_id:
        name = f"arxiv-{paper.arxiv_id}.pdf"
    elif paper.doi:
        name = f"doi-{slugify(paper.doi)}.pdf"
    else:
        name = f"{slugify(paper.title[:50])}.pdf"
    p = cache_dir / name
    return p if p.exists() else None
```
Refactor `get_cached_pdf` to compute its `cache_path` from the same naming (call a shared private namer) so they can't drift.
- [ ] **Step 3b:** in `papers.py` add (mirror images.py guard):
```python
@router.get("/papers/{paper_id:path}/pdf")
def get_paper_pdf(paper_id: str, out: Path = Depends(deps.get_output_dir)) -> FileResponse:
    from ndif_citations.pdf_cache import cached_pdf_path
    from ndif_citations.server.services import papers_svc
    paper = papers_svc.resolve(out, paper_id)
    if paper is None:
        raise HTTPException(404, "Paper not found")
    path = cached_pdf_path(paper, out)
    if path is None:
        raise HTTPException(404, "No cached PDF")
    resolved, pdfs_dir = path.resolve(), (out / "pdfs").resolve()
    if resolved.parent != pdfs_dir:
        raise HTTPException(404, "Invalid PDF path")
    return FileResponse(str(resolved), media_type="application/pdf")
```
(Confirm `papers_svc.resolve` exists + the router's existing imports for `FileResponse`/`HTTPException`/`deps`.)
- [ ] **Step 4: run** the two test files + `pytest tests/ -q` → PASS.
- [ ] **Step 5: commit** `git commit -am "feat(api): GET /papers/{id}/pdf serves cached PDF; cached_pdf_path helper"`

---

### Task 5: `get_paper` adds `missing` + `has_pdf`

**Files:** Modify `src/ndif_citations/server/services/papers_svc.py`; Test `tests/test_api_papers_read.py` (append)

- [ ] **Step 1: failing test** — assert `GET /papers/{id}` detail dict includes `missing` (== `_compute_missing(paper)`) and `has_pdf` (bool matching `cached_pdf_path is not None`). Use a fixture paper with a known missing field + no cached PDF (`has_pdf False`).
- [ ] **Step 2: run** → FAIL.
- [ ] **Step 3: implement** — in `get_paper`, after `d = paper.to_full_dict()`, add `d["missing"] = _compute_missing(paper); d["has_pdf"] = cached_pdf_path(paper, out) is not None; return d` (import `cached_pdf_path`).
- [ ] **Step 4: run** `pytest tests/test_api_papers_read.py tests/ -q` → PASS.
- [ ] **Step 5: commit** `git commit -am "feat(api): paper detail includes missing + has_pdf"`

---

### Task 6: FE types + client + hooks

**Files:** Modify `web/src/api/types.ts`, `web/src/api/client.ts`, `web/src/api/hooks.ts`

- [ ] **Step 1:** in `types.ts` `PaperDetail` add: `missing?: string[]`, `has_pdf?: boolean`, `ndif_context_windows?: string[]`, `context_source?: string`.
- [ ] **Step 2:** in `client.ts` add `export const paperPdfUrl = (id: string) => \`/api/papers/${encodeURIComponent(id)}/pdf\``.
- [ ] **Step 3:** (no new hook needed — evidence rides the existing `usePaper`.) Confirm `tsc`: `cd web && bun run build` → succeeds.
- [ ] **Step 4: commit** `git commit -am "feat(web): PaperDetail types for missing/has_pdf/evidence + pdf url helper"`

---

### Task 7: PaperSheet — needs-attention strip, lightbox, PDF button, evidence section, full abstract

**Files:** Modify `web/src/components/papers/PaperSheet.tsx` (READ it fully first)

- [ ] **Step 1:** Add a **needs-attention strip** under the SheetHeader (~after line 651): when `(paper.missing ?? []).length`, render chips per missing field with friendly labels (`image→Thumbnail, affiliations→Affiliations, abstract→Abstract, summary→Summary, venue→Venue`); clicking a chip for an editable field (`venue/affiliations/summary/abstract`) opens the existing edit form (reuse the edit toggle), `image` scrolls to the image controls. Guard with `?? []`.
- [ ] **Step 2:** Make the thumbnail `<img>` (@617) a button opening a **lightbox** (reuse `Dialog` from `components/ui`; if none, use the existing `Sheet`/`AlertDialog` primitive or a fixed overlay) showing the image at natural size; Esc/backdrop closes. Keep "No thumbnail" non-clickable.
- [ ] **Step 3:** In the links row (@734), when `paper.has_pdf`, add `<ExternalLinkButton href={paperPdfUrl(paper.id ?? paperId)} label="Cached PDF" />` (PaperDetail has no `id` — use the `paperId` prop). Opens in new tab (ExternalLinkButton already targets `_blank`).
- [ ] **Step 4:** Un-clamp/expandable abstract (the `line-clamp-[12]` @725): show full text in a scrollable block (or a "show more" toggle). "No abstract" when empty.
- [ ] **Step 5:** Add an **Evidence** section (collapsed by default): if `(paper.ndif_context_windows ?? []).length`, render each window in a `font-mono text-xs` block with a `context_source` badge ("from PDF"/"from abstract"); if empty, "No NDIF evidence found in source."
- [ ] **Step 6:** `cd web && bun run build` → succeeds (tsc). Commit `git commit -am "feat(web): PaperSheet needs-attention, lightbox, cached-PDF, evidence, full abstract"`

---

### Task 8: Papers — needs-attention filter + prev/next in Sheet

**Files:** Modify `web/src/pages/Papers.tsx` (READ it first) + `PaperSheet.tsx` (props)

- [ ] **Step 1:** Add a "Needs attention" filter chip/toggle to the existing filter bar; when on, filter rows to `(row.missing ?? []).length > 0` (client-side; rows already carry `missing`).
- [ ] **Step 2:** Track the open paper's index in the current filtered+sorted row list; pass `onPrev`/`onNext`/`hasPrev`/`hasNext` into `PaperSheet` (extend `Props`). Render ‹ / › buttons in the Sheet header (disabled at ends); add `←`/`→` key handlers while the Sheet is open (guarded so they don't fire while a text input/edit form is focused).
- [ ] **Step 3:** `cd web && bun run build` → succeeds. Commit `git commit -am "feat(web): needs-attention filter + prev/next navigation in PaperSheet"`

---

### Task 9: Backfill evidence + full verification

- [ ] **Step 1:** `pytest tests/ -q` → all green.
- [ ] **Step 2:** Restart server (`lsof -ti:8723|xargs kill; python -m ndif_citations serve --no-open`) — new routes/fields need it.
- [ ] **Step 3:** Back up catalog, then `python -m ndif_citations backfill-evidence` (no LLM) to populate evidence for the ~105 PDF-cached papers; spot-check a paper now has `ndif_context_windows`.
- [ ] **Step 4:** `cd web && bun run build`; cmux-browser smoke: open a paper → needs-attention chips, click thumbnail→lightbox, Cached PDF opens, Evidence renders, prev/next + ←/→ work, needs-attention table filter works. Check `cmux browser errors list` is empty.
- [ ] **Step 5: STOP and report** to the user (live verification results); do not push/merge without go-ahead.

---

## Self-Review
**Spec coverage:** needs-attention panel+filter (T5,T7,T8) · lightbox (T7) · cached-PDF endpoint+button (T4,T7) · abstract display+edit (T7, edit via existing form) · persisted evidence (T1,T2,T3,T7) · prev/next+arrows (T8) — all covered. **Placeholders:** backend steps have complete code; FE steps give exact files/insertion points + new-component code + build/cmux verification (FE integration into a 600-line component is read-then-edit by design). **Type consistency:** `cached_pdf_path(paper, output_dir)`, `compute_context(paper, out)->(windows, source, signal)`, detail keys `missing/has_pdf/ndif_context_windows/context_source`, `paperPdfUrl(id)`, PaperSheet new props `onPrev/onNext/hasPrev/hasNext` — consistent across tasks. **Verify during impl:** `papers_svc.resolve` exists; `slugify` import in pdf_cache; golden-fixture parity for new model fields; a `Dialog` primitive exists in `components/ui` (else use an overlay).
