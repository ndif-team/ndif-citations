# NDIF Citation Tracker

Automated pipeline to discover, extract metadata for, and catalog all papers citing or using [NDIF](https://ndif.us) (National Deep Inference Fabric) and [NNsight](https://nnsight.net/).

Replaces a manual workflow of checking Google Scholar, copying metadata into spreadsheets, screenshotting figures, and updating website HTML — with a single command.

<p align="center">
  <img src="docs/architecture.svg" alt="Pipeline architecture" width="680"/>
</p>

## Quick start

```bash
# Install
pip install -e .

# Configure API keys
cp .env.example .env
# Edit .env with your keys

# Discovery only (no LLM keys needed)
python -m ndif_citations discover

# Full pipeline
python -m ndif_citations run
```

## Commands

| Command | Description |
|---------|-------------|
| `python -m ndif_citations run` | Full pipeline: discover papers + repos, enrich, process, output |
| `python -m ndif_citations run --fresh` | Full pipeline, ignoring all existing output |
| `python -m ndif_citations run --skip-github` | Papers only — skip GitHub repo discovery |
| `python -m ndif_citations run --skip-papers` | GitHub repos only — skip S2/OpenAlex/LLM |
| `python -m ndif_citations discover` | Discovery only — list papers and repos, no LLM calls |
| `python -m ndif_citations add <url>` | Process a single paper by URL and append to output |
| `python -m ndif_citations edit <id>` | Interactively override any of 16 curated fields on one paper (sets `manual_override=True`) |
| `python -m ndif_citations edit <id> --set field=value` | One-shot field edit, scriptable. Repeat `--set` for multiple fields. Add `--yes` to skip confirm. |
| `python -m ndif_citations reclassify [--ids X,Y]` | Re-run LLM classify on existing papers (apply new pre-filter / band rules) |
| `python -m ndif_citations promote <id>` | Move paper to verified, freeze with `manual_override=True` |
| `python -m ndif_citations demote <id> --reason ...` | Move paper to pending, freeze with `manual_override=True` |
| `python -m ndif_citations discard <id>` | Move paper to discarded, freeze with `manual_override=True` |
| `python -m ndif_citations debug <id>` | Read-only trace for one paper (PDF cache, keyword hits, classification state) |

All commands accept `--output-dir <path>` and `--verbose` flags.

### Classification confidence bands

Every classified paper carries a `category_confidence_band` (categorical) alongside the legacy `category_confidence` float:

| Band | Float equiv | When | Bucket |
|---|---|---|---|
| `CERTAIN` | 1.00 | `manual_override=True` OR pre-filter caught explicit non-use ("alternative to NDIF") | VERIFIED |
| `HIGH` | 0.85 | LLM verdict with ≥2 surviving context windows OR `linked_paper_tier ≤ 2` cross-link | VERIFIED |
| `MEDIUM` | 0.55 | LLM verdict on a single context window OR abstract-only OR pre-filter caught comparison-table / acks-only | PENDING (`MEDIUM_CONFIDENCE`) |
| `LOW` | 0.30 | Keyword fallback (LLM unavailable / errored) | PENDING (`LOW_CONFIDENCE`) |
| `NONE` | 0.00 | UNCLASSIFIED (no evidence / LLM unparseable) | PENDING (`UNCLASSIFIED_*`) |

`manual_override=True` papers route to `FILL_GAPS` if any `has_*` flag is False, so the pipeline backfills empty description / thumbnail / affiliations on the next run without overwriting curated values.

## Output

Each run merges into existing output and reports what changed:

```
★ 5 NEW papers added
✓ 47 already in database
✓ 2 papers updated (venue upgraded)
```

Use `--fresh` to rebuild from scratch.

```
output/
├── research-papers.json       # Website-ready papers (matches ResearchPaper TS interface)
├── research-papers-full.json  # Full paper metadata — persistent state between runs
├── research-papers.csv        # Extended paper columns for spreadsheet / grant reporting
├── github-repos.json          # Website-ready GitHub repos (all nnsight dependents)
├── github-repos-full.json     # Full repo metadata — persistent state between runs
├── github-repos.csv           # Extended repo columns
├── research-data.xlsx         # Two-sheet spreadsheet: "Papers" + "GitHub"
├── images/                    # Extracted paper thumbnails
└── raw/                       # Raw API responses for debugging
```

## Project structure

```
ndif-citations/
├── pyproject.toml
├── .env.example
├── data/known_venues.json
├── docs/
└── src/ndif_citations/
    ├── cli.py        # Click CLI (run, discover, add)
    ├── config.py     # Constants, seed IDs, keywords
    ├── models.py     # Pydantic models (3-way category)
    ├── discover.py   # Phase 1: S2 + OpenAlex + GitHub
    ├── extract.py    # Phase 2: venue, peer review, affiliations
    ├── process.py    # Phase 3: LLM + thumbnails
    ├── output.py     # Phase 4: merge, JSON/CSV, CLI report
    └── utils.py      # PDF download, slugify, dedup, BibTeX
```

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `S2_API_KEY` | No | Semantic Scholar API key (higher rate limits) |
| `LLM_BASE_URL` | For `run` | OpenAI-compatible LLM endpoint |
| `LLM_API_KEY` | For `run` | API key for the LLM provider |
| `LLM_MODEL` | For `run` | Model identifier |
| `OPENALEX_EMAIL` | No | Email for OpenAlex polite pool |
| `GITHUB_TOKEN` | No | GitHub personal access token — upgrades GitHub API from 60 req/hr (anonymous) to 5000 req/hr. Without it, activity fields (stars, forks, last_commit) may be null for repos beyond the first ~30 per run. No scopes needed for public repos. |

LLM keys are only needed for `run` (summaries + classification). The `discover` command works without them.

<details>
<summary><strong>LLM provider examples</strong></summary>

The LLM integration is provider-agnostic via the `openai` Python library:

```bash
# NVIDIA Build (default, free tier)
LLM_BASE_URL=https://integrate.api.nvidia.com/v1
LLM_MODEL=meta/llama-3.1-70b-instruct

# OpenAI
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=gpt-4o-mini

# Local (Ollama)
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=llama3.1
```

</details>

---

## How it works

The pipeline runs in four phases. The [architecture diagram](#ndif-citation-tracker) above shows the full flow.

<details>
<summary><strong>Phase 1: Discovery</strong> — find papers across three sources</summary>

<br/>

- **Semantic Scholar** — traverses the citation graph of the [NDIF seed paper](https://arxiv.org/abs/2407.14561) (ICLR 2025). Catches every paper that formally cites NDIF/NNsight.
- **OpenAlex** — fulltext search across millions of papers. Catches papers that mention NDIF in their text but may not have a formal citation yet.
- **GitHub dependents** — scrapes the [nnsight dependents page](https://github.com/ndif-team/nnsight/network/dependents) and captures **every non-fork, non-archived repo** as a first-class `DiscoveredRepo` entity. READMEs are scanned for arXiv links (for cross-linking to papers) and NDIF keywords (for classification). Activity metadata (stars, forks, last commit, language, license, topics) is fetched via the GitHub API. Stale repos (404 / renamed / archived since last run) are automatically removed from the output on the next run.

**GitHub Repo Classification**

After scraping the GitHub dependents list, each repo is enriched via the GitHub API (stars, forks, description, parent fork info). The README is fetched **once** per repo during enrichment. The following classification happens in a single pass:

- **`repo_type` tagging** — every repo is tagged as one of:
  - `research` — uses NDIF infrastructure (`uses_ndif` category), has a linked paper, or has ≥6 stars and a description
  - `course` — forked from a known course source (e.g. `callummcdougall/ARENA_3.0`), matches course name patterns (ARENA, MATS, CBAI), or lost a template-inherited paper link while having 0 stars and no description
  - `experiment` — everything else (default)

- **Linked paper detection** — `linked_paper_url` is set using a 5-tier priority:
  1. BibTeX block in README (most reliable — author-placed citation)
  2. arXiv ID under a `## Citation` / `## Paper` / `## Reference` section header
  3. Exactly one post-2020 arXiv ID in the entire README
  4. Most recent post-2020 arXiv ID when multiple exist
  5. `null` — no paper detected

- **Shared-paper template detection** — After enrichment, a cross-repo pass detects when ≥ `SHARED_PAPER_THRESHOLD` (default: 5) repos share the same `linked_paper_url`. This signals template-inherited links (e.g. 22+ ARENA course forks all pointing to the same paper). Only the highest-star repo keeps the link; all others are cleared. These unlinked repos with 0 stars and no description are tagged `course`.

- **Checkpoint resilience** — The GitHub dependents HTML scrape persists each page to `output/raw/github_dependents_checkpoint/page_{n}.json` immediately after scraping. If the scrape is interrupted (timeout, network error), the next run resumes from the last successful page. The checkpoint directory is deleted only on full successful completion.

Results are deduplicated by arXiv ID, DOI, and title similarity (>90% via `rapidfuzz`). When duplicates exist across sources, metadata is merged — S2 preferred for structured fields, OpenAlex for affiliations.

</details>

<details>
<summary><strong>Phase 2: Metadata enrichment</strong> — format venues, detect peer review, generate BibTeX</summary>

<br/>

- **Venue formatting** — normalized to website convention (`"ICLR 2025"`, `"ArXiv 2025"`, `"NeurIPS 2024 Workshop on..."`), driven by `data/known_venues.json`.
- **Peer-review detection** — papers at known conferences/journals/workshops flagged as peer-reviewed (for NSF reporting).
- **BibTeX generation** — auto-generated from structured metadata.
- **Affiliation enrichment** — papers missing institutional data looked up via OpenAlex.

</details>

<details>
<summary><strong>Phase 3: Content processing</strong> — LLM classification, summaries, thumbnails</summary>

<br/>

For each paper:

1. **PDF download** — fetched once from open-access URL or arXiv, shared across subsequent steps.
2. **Context extraction** — PDF text searched for keyword mentions. Up to 5 context windows (500 chars each) extracted.
3. **LLM classification** — context sent to the configured LLM, which classifies the paper:
   - `uses_ndif` — runs experiments on NDIF infrastructure
   - `uses_nnsight` — uses the nnsight Python library
   - `referencing` — mentions without active use
4. **LLM summary** — abstract summarized into 1–3 sentences for the website.
5. **Thumbnail extraction** — smart figure detection across all pages, scoring candidates by caption quality (keywords like "architecture", "pipeline", "overview"), size/aspect ratio, and section context (method sections preferred). Extracts the best representative figure as PNG.

If the LLM is unavailable, rule-based fallbacks handle classification (keyword matching) and summarization (first 2 sentences of abstract).

</details>

<details>
<summary><strong>Phase 4: Merge and output</strong> — append new, preserve existing, detect upgrades</summary>

<br/>

New results merge into existing `research-papers-full.json`:

- Matches by arXiv ID, DOI, or title similarity
- Appends genuinely new papers
- Fills metadata gaps on existing papers
- Detects **venue upgrades** (arXiv preprint → conference acceptance)
- Respects `manual_override` — hand-edited papers are never overwritten
- **GitHub repo staleness** — repos that return 404, have been renamed, or become archived since the last run are automatically purged from `github-repos-full.json`. Rate-limit responses never trigger removal (transient API budget issues don't delete real data).

</details>

<details>
<summary><strong>Screenshots</strong></summary>

<br/>

<table>
  <tr>
    <td><img src="docs/pipeline-running.png" alt="Pipeline running" width="100%"/></td>
    <td><img src="docs/pipeline-report.png" alt="Run complete" width="100%"/></td>
  </tr>
  <tr>
    <td align="center"><em>Processing 79 papers with LLM classification</em></td>
    <td align="center"><em>Run complete — 79 papers across 3 categories</em></td>
  </tr>
</table>

</details>

---

## Configuration

<details>
<summary><strong>Discovery and classification keywords</strong> — <code>config.py</code></summary>

<br/>

**Discovery keywords** — what OpenAlex searches for in paper full text:

```python
OPENALEX_SEARCH_QUERIES = [
    "nnsight",                            # library name
    '"national deep inference fabric"',   # full project name (exact phrase)
    "ndif.us",                            # project URL
]
```

Double quotes inside the string trigger exact phrase matching.

**PDF classification keywords** — searched in downloaded PDFs to extract context for the LLM:

```python
NDIF_KEYWORDS = ["nnsight", "NNsight", "NDIF", "ndif.us", "nnsight.net", "import nnsight"]
```

If a PDF contains none of these, the paper defaults to `referencing` without an LLM call.

**Seed paper** — root of the Semantic Scholar citation graph:

```python
SEED_S2_ID = "ARXIV:2407.14561"
```

**Rate limits** — adjust for authenticated API keys with higher quotas:

```python
S2_RATE_LIMIT_SLEEP = 3.0       # unauthenticated
LLM_RATE_LIMIT_SLEEP = 12.0     # NVIDIA Build free tier
OPENALEX_RATE_LIMIT_SLEEP = 0.15
GITHUB_RATE_LIMIT_SLEEP = 2.0
```

</details>

<details>
<summary><strong>Venue recognition</strong> — <code>data/known_venues.json</code></summary>

<br/>

Controls venue formatting, peer-review detection, and venue-type classification:

```json
{
  "conferences": ["ICLR", "NeurIPS", "ICML", ...],
  "journals": ["JMLR", "TMLR", "Nature", ...],
  "preprint_servers": ["ArXiv", "BiorXiv", ...]
}
```

Matching is case-insensitive substring — adding `"WCCI"` matches `"2026 IEEE WCCI"`, etc.

</details>

<details>
<summary><strong>GitHub repo classification</strong> — <code>config.py</code></summary>

<br/>

Controls repo tagging, linked-paper detection, and shared-paper template cleanup:

| Config constant | Default | Description |
|---|---|---|
| `EXCLUDED_GITHUB_REPOS` | `{"ndif-team/nnsight"}` | Repos to exclude entirely from output (e.g. the library itself) |
| `KNOWN_COURSE_SOURCES` | `{"callummcdougall/ARENA_3.0"}` | Repos whose forks are tagged `course`. Extend as new course templates are discovered. |
| `COURSE_NAME_PATTERNS` | `["ARENA", "MATS", "CBAI"]` | Case-insensitive substrings in repo name or description that signal course origin |
| `SHARED_PAPER_THRESHOLD` | `5` | Minimum number of repos sharing a `linked_paper_url` to trigger template detection |

</details>

## Development

Run tests: `pytest tests/` (after `pip install -e ".[dev]"`).

## License

Internal tool for [NDIF](https://ndif.us) — an NSF-funded project at Northeastern University.
