# Paper Detail as a Curation Workspace — Design

- **Date:** 2026-06-06
- **Branch (base):** `feat/citations-web-app`
- **Status:** Approved (brainstorm) → ready for implementation plan
- **Scope cut:** "Workspace" (verify + fix in-app) + persisted evidence

> Note: `docs/` is gitignored in this repo (the maintainer's choice — design specs live on
> disk, untracked). This spec is not committed; it is the working reference for the plan.

## 1. Problem

The NDIF citations web app already lets a curator browse, edit, bucket, and reprocess papers.
But three things force the curator out of the app or leave them guessing:

1. **The figure thumbnail isn't viewable** — it renders small and isn't clickable, so you
   can't actually read the figure that was extracted.
2. **The PDF lives on disk but isn't reachable** — the pipeline caches ~105/147 PDFs in
   `output/pdfs/`, yet the only "PDF" link points to the external (sometimes paywalled) URL.
3. **The ⚠ flag is cryptic** — it says "missing metadata" but not *which* fields, and only as a
   tiny table tooltip. There is no in-detail view of what needs fixing.

Underlying all three: the detail view is a *display*, not a *workspace*. The curator can't
verify relevance (read the figure / PDF / evidence), see what's missing, or move through the
queue without context-switching.

## 2. Goals / Non-goals

**Goals**
- Make the Paper detail Sheet a self-contained place to **verify** a paper (figure, cached PDF,
  abstract, the NDIF evidence the classifier used) and **see/fix** what's missing.
- Surface **exactly which fields are missing**, in the detail view and as a table filter.
- Show the classifier's **evidence** (context windows) **consistently and cheaply** by
  persisting what the LLM actually saw — not recomputing it per view.
- Let the curator **step through** the filtered list without closing the Sheet.

**Non-goals (deferred)**
- Inline/embedded PDF viewer (we open the cached PDF in a new tab).
- Per-field re-fetch (re-pull affiliations/abstract for one field).
- Full keyboard-driven triage (`j`/`k` row nav, `p`/`d` shortcuts) — only `←`/`→` while the
  Sheet is open is included.
- A separate "Why" panel beyond the Evidence section.

## 3. Feature design

### 3.1 "Needs attention" panel (detail) + table filter

**Backend.** `papers_svc.get_paper()` already returns `to_full_dict()`. Enrich it with a
computed `missing: list[str]` (reuse the existing `_compute_missing`, which checks
`image, affiliations, abstract, summary, venue[empty/weak]`) and `has_pdf: bool` (see 3.3).
`_compute_missing` stays the single source of truth for both the row and the detail.

**Frontend (PaperSheet).** Under the title, render a "Needs attention" strip when
`missing.length > 0`: one chip per missing field with a friendly label
(`image→Thumbnail, affiliations→Affiliations, abstract→Abstract, summary→Summary, venue→Venue`).
Chip click maps to the relevant existing affordance:
- editable text fields (`venue`, `affiliations`, `summary`, `abstract`) → open the edit form
  (all four already live in the 16-field `edit_schema`);
- `image` → scroll to / focus the existing Replace-image / Re-extract controls.

**Frontend (Papers table).** Add a "Needs attention" filter chip in the existing filter bar.
Rows already carry `missing`, so this is **client-side**: keep rows where
`(row.missing ?? []).length > 0`. No API change.

### 3.2 Thumbnail lightbox

Frontend only. The thumbnail `<img>` (served from `/api/images/{basename}`) becomes a button
that opens a modal (reuse the existing Dialog primitive) showing the image at natural size
(max viewport-bounded, scroll if larger). Esc / backdrop click / a close button dismiss it.
The "No thumbnail" placeholder stays when `has_image` is false (not clickable).

### 3.3 Cached-PDF access (new tab)

**Backend.**
- `pdf_cache.py`: add `cached_pdf_path(paper: DiscoveredPaper, output_dir: Path) -> Path | None`.
  It computes the cache filename using the **existing** naming
  (`arxiv-{arxiv_id}.pdf` / `doi-{slugify(doi)}.pdf` / `{slugify(title[:50])}.pdf`) under
  `output_dir / "pdfs"` and returns it **only if it exists** — no `mkdir`, **no network download**
  (unlike `get_cached_pdf`). Refactor `get_cached_pdf` to derive its path from this helper so the
  naming can't drift.
- `server/routers/papers.py`: `GET /api/papers/{id}/pdf` → resolve the paper, call
  `cached_pdf_path`; if `None` → 404; else `FileResponse(path, media_type="application/pdf")`.
  Apply the **same path-traversal guard** as the images route (resolved file's parent must equal
  `(out / "pdfs").resolve()`).
- `papers_svc.get_paper()` sets `has_pdf = cached_pdf_path(...) is not None`.

**Frontend.** In the PaperSheet links row, when `has_pdf`, render a "View cached PDF" button
(secondary) that opens `/api/papers/{id}/pdf` in a new tab. The existing external
Paper / PDF links are unchanged.

### 3.4 Abstract display + edit

Frontend only. `abstract` is already a persisted model field (flows through `to_full_dict()`)
**and already editable** — it's one of the 16 `edit_schema` fields (a textarea in the edit form).
Render a **read** Abstract section in the Sheet showing the full stored text (scroll past a max
height); show "No abstract" when empty (and it appears in the Needs-attention strip). An "Edit"
affordance (and the `abstract` needs-attention chip) opens the **existing** edit form — no new
edit path. (Chosen over inline-edit to keep one edit surface.)

### 3.5 Evidence — persisted, not recomputed

**Why persist.** During classification `process.py` already extracts NDIF context windows
(`extract_ndif_context` → split on `"\n---\n"`), runs them through `_apply_prefilters`
(negative-evidence / comparison-table / acks-only), and feeds the **surviving** windows to the
LLM; `surviving_window_count` sets the confidence band. These windows are the authoritative
"why" but are discarded. Recomputing them at view time would (a) cost a PDF re-parse per open
and (b) risk showing a *different* set than the verdict was based on. So we store what the
classifier used.

**Model** (`models.py`, `DiscoveredPaper`):
- `ndif_context_windows: list[str] = Field(default_factory=list)` — the surviving windows.
- `context_source: str = "none"` — `"pdf" | "abstract" | "none"`.

Both auto-serialize via `model_dump()` → they ride along in `GET /papers/{id}`. **No new
endpoint, no lazy fetch.**

**Shared helper** (`process.py`): factor the existing lines that resolve the PDF, extract
context, split, and pre-filter into
`compute_context(paper, out) -> tuple[list[str], str, str | None]`
returning `(surviving_windows, context_source, prefilter_signal)`. The classify path uses all
three (unchanged behavior — `surviving_window_count = len(surviving_windows)`); it now also
writes `paper.ndif_context_windows = surviving_windows` and `paper.context_source = context_source`.
The backfill reuses the same helper. This guarantees the two paths can't diverge.

**Backfill** (`cli.py`): `ndif-citations backfill-evidence [--ids X,Y] [--output-dir P]` —
**no LLM**. For each paper, call `compute_context` (reads the cached PDF if present, else falls
back to abstract scan exactly as classify does) and store the result; write the catalog back.
Populates the existing ~105 PDF-cached papers immediately. Newly processed/added papers store
it automatically going forward. (Backfilled windows are re-derived with the *same* deterministic
logic; future-processed papers capture it at classify time. In practice identical for a given
PDF + keyword config.)

**Frontend (PaperSheet).** An Evidence section showing:
- a small badge for `context_source` ("from PDF" / "from abstract" / "no evidence");
- each window in a readable monospace block (optionally highlighting the matched NDIF keyword);
- empty + `context_source === "none"` → "No NDIF evidence found in the source."

### 3.6 Prev / Next within the Sheet

Frontend only. `Papers.tsx` holds the filtered + sorted row list and the currently-open paper id.
Track the open row's index; pass `onPrev` / `onNext` callbacks (open the adjacent row's id) and
`hasPrev` / `hasNext` into the PaperSheet. The Sheet header renders ‹ / › buttons (disabled at
ends). While the Sheet is open, `←` / `→` trigger the same (guarded so they don't fire while a
text input / the edit form is focused).

### 3.7 Data quality (handled by the enrichment spec — Plan A)

Truncated abstracts/authors and missing affiliations/identifiers are a **metadata-quality** issue
(Google Scholar returns snippets), not a UI issue. They are addressed by the companion spec
`2026-06-06-robust-enrichment-design.md` and its `re-enrich` command, which runs **before** this UX
plan so the workspace displays clean data. This spec only *displays + edits* whatever is stored
(§3.4); it does not re-fetch metadata.

## 4. Changed surface (file-by-file)

**Backend** (`ndif-citations/src/ndif_citations/`)
- `models.py` — `DiscoveredPaper`: + `ndif_context_windows`, `context_source`.
- `pdf_cache.py` — + `cached_pdf_path()`; `get_cached_pdf` reuses it.
- `process.py` — + `compute_context()` helper; classify writes the two new fields (behavior-preserving).
- `cli.py` — + `backfill-evidence` command. (Abstract/metadata repair = `re-enrich`, in Plan A.)
- `server/services/papers_svc.py` — `get_paper()` adds `missing` + `has_pdf`.
- `server/routers/papers.py` — + `GET /papers/{id}/pdf`.

**Frontend** (`ndif-citations/web/src/`)
- `api/types.ts` — `PaperDetail`: + `missing?: string[]`, `has_pdf?: boolean`,
  `ndif_context_windows?: string[]`, `context_source?: string`.
- `api/client.ts` — + `paperPdfUrl(id)` helper.
- `components/papers/PaperSheet.tsx` — needs-attention strip, thumbnail lightbox, PDF button,
  abstract section, evidence section, prev/next header controls.
- `pages/Papers.tsx` — needs-attention filter chip; track open index, feed prev/next into the Sheet.

## 5. API additions

| Method | Path | Returns |
|---|---|---|
| GET | `/api/papers/{id}/pdf` | `application/pdf` (cached) or 404 |
| GET | `/api/papers/{id}` (existing) | now also `missing`, `has_pdf`, `ndif_context_windows`, `context_source` |

Both new model fields appear on the detail response only (the table row is unchanged besides the
already-present `missing`). `/pdf` is a read; not affected by the active-run mutation guard.

## 6. Testing

**Backend (pytest):**
- `cached_pdf_path`: arxiv/doi/title naming → hit returns path, miss returns `None`, no download
  on miss, traversal-style ids rejected.
- `GET /papers/{id}/pdf`: 200 + `application/pdf` for a cached fixture; 404 when absent;
  traversal guard (parent must be `out/pdfs`).
- `get_paper`: response includes `missing` (matches `_compute_missing`) and correct `has_pdf`.
- `compute_context`: deterministic windows from a fixture PDF / mocked text; pdf vs abstract vs
  none source; parity with the pre-refactor classify control flow (windows + count unchanged).
- `process` classify: persists `ndif_context_windows` / `context_source` without changing the
  resulting band/category (extend existing classify tests).
- `backfill-evidence`: populates the field for a fixture catalog, no LLM call, idempotent.

**Frontend:** `bun run build` (tsc typecheck) + live cmux smoke of each affordance
(lightbox opens, PDF tab opens, needs-attention chips + filter, abstract/evidence render,
prev/next + arrow keys).

## 7. Migration / rollout

0. **Prerequisite:** Plan A (`re-enrich`) has cleaned the catalog (full abstracts/authors/ids).
1. Land model + `compute_context` + classify write + `cached_pdf_path` + `/pdf` route +
   `get_paper` enrichment + `backfill-evidence`.
2. Restart `serve` (new routes).
3. Back up the catalog, then run once (no LLM): `ndif-citations backfill-evidence`
   (populates `ndif_context_windows` for PDF-cached papers).
4. `cd web && bun run build`; verify in the browser.

JSON size: storing surviving windows (~few × ~500 chars per paper) is small relative to the
existing abstracts already in the catalog.

## 8. Risks / open questions

- **Evidence for non-PDF / paywalled papers**: `context_source` will be `"abstract"` or `"none"`;
  the UI states this rather than showing nothing ambiguous.
- **Catalog rewrite by backfill**: mitigated by the existing backup convention
  (`output/backups/*`) before the one-time run.
- **`compute_context` refactor must be behavior-preserving** for classify — covered by a parity
  test against current window/count outputs.
- **Chip→action mapping** for fields that aren't directly editable (abstract) is informational
  (scroll), not an edit — acceptable for v1.

## 9. Out of scope (backlog)

Inline PDF viewer; per-field re-fetch endpoints; full keyboard triage (`j/k/p/d`); a dedicated
provenance/"why" panel beyond Evidence; showing pre-filtered-out (non-surviving) windows.
