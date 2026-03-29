# NDIF Citation Tracker

Automated pipeline to discover, extract metadata for, and catalog all papers citing or using [NDIF](https://ndif.us) (National Deep Inference Fabric) and [NNsight](https://nnsight.net/).

Replaces a manual workflow of checking Google Scholar/arXiv, copying metadata into spreadsheets, classifying papers, generating summaries, downloading figures, and updating website HTML — with a single CLI command.

## Architecture

<p align="center">
  <img src="docs/architecture.svg" alt="Pipeline architecture" width="680"/>
</p>

### Phase 1: Discovery

Three sources are queried to maximize coverage:

- **Semantic Scholar** — traverses the citation graph of the [NDIF seed paper](https://arxiv.org/abs/2407.14561) (ICLR 2025). Catches every paper that formally cites NDIF/NNsight.
- **OpenAlex** — fulltext search across millions of papers. Catches papers that mention NDIF in their text but may not have a formal citation yet.
- **GitHub dependents** — scrapes the [nnsight dependents page](https://github.com/ndif-team/nnsight/network/dependents), fetches each repo's README via the GitHub API, and extracts arXiv links. Catches code users who haven't published or cited yet.

Results are **deduplicated** by arXiv ID (exact), DOI (exact), and title similarity (>90% via `rapidfuzz`). When duplicates exist across sources, metadata is merged — S2 data is preferred for structured fields, OpenAlex for affiliations.

### Phase 2: Metadata enrichment

- **Venue formatting** — raw API strings normalized to website convention (`"ICLR 2025"`, `"ArXiv 2025"`, `"NeurIPS 2024 Workshop on..."`), driven by the venue lists in `data/known_venues.json`.
- **Peer-review detection** — papers at known conferences/journals/workshops are flagged as peer-reviewed (for NSF grant reporting).
- **BibTeX generation** — auto-generated from structured metadata.
- **Affiliation enrichment** — papers missing institutional data get looked up in OpenAlex by DOI or title.

### Phase 3: Content processing

For each paper:

1. **PDF download** — fetched once from the open-access URL or arXiv, shared across subsequent steps.
2. **Context extraction** — PDF full text searched for keyword mentions (`config.py → NDIF_KEYWORDS`). Up to 5 context windows (500 chars each) are extracted around matches.
3. **LLM classification** — context excerpts sent to the configured LLM, which classifies the paper:
   - `uses_ndif` — runs experiments on NDIF infrastructure
   - `uses_nnsight` — uses the nnsight Python library
   - `referencing` — mentions NDIF/nnsight without active use
4. **LLM summary** — abstract summarized into 1–3 qualitative sentences for the website.
5. **Thumbnail extraction** — largest figure from the first 4 PDF pages extracted and saved as PNG.

If the LLM is unavailable, rule-based fallbacks handle classification (keyword matching) and summarization (first 2 sentences of abstract).

### Phase 4: Merge and output

New results **merge** into existing `research-papers-full.json`:

- Matches by arXiv ID, DOI, or title similarity
- Appends genuinely new papers
- Fills metadata gaps on existing papers (abstracts, affiliations, etc.)
- Detects **venue upgrades** (arXiv preprint → conference acceptance)
- Respects the `manual_override` flag — hand-edited papers are never overwritten

Three output files:

| File | Purpose |
|------|---------|
| `research-papers.json` | Website-ready JSON matching the `ResearchPaper` TypeScript interface |
| `research-papers-full.json` | Full metadata — persistent state between runs |
| `research-papers.csv` | Extended columns for spreadsheet import and grant reporting |

## Screenshots

<table>
  <tr>
    <td><img src="docs/pipeline-running.png" alt="Pipeline running — Phase 3 processing" width="100%"/></td>
    <td><img src="docs/pipeline-report.png" alt="Pipeline complete — final report" width="100%"/></td>
  </tr>
  <tr>
    <td align="center"><em>Processing 79 papers with LLM classification</em></td>
    <td align="center"><em>Run complete — 79 papers across 3 categories</em></td>
  </tr>
</table>

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
| `python -m ndif_citations run` | Full pipeline: discover, enrich, process, output |
| `python -m ndif_citations run --fresh` | Full pipeline, ignoring existing output |
| `python -m ndif_citations discover` | Discovery only — list papers, no LLM calls |
| `python -m ndif_citations add <url>` | Process a single paper by URL and append to output |

All commands accept `--output-dir <path>` and `--verbose` flags.

## Environment variables

| Variable | Required | Description |
|----------|----------|-------------|
| `S2_API_KEY` | No | Semantic Scholar API key (higher rate limits) |
| `LLM_BASE_URL` | For `run` | OpenAI-compatible LLM endpoint |
| `LLM_API_KEY` | For `run` | API key for the LLM provider |
| `LLM_MODEL` | For `run` | Model identifier |
| `OPENALEX_EMAIL` | No | Email for OpenAlex polite pool |

LLM keys are only needed for the full `run` command (summaries + classification). The `discover` command works without them.

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

## Output

```
output/
├── research-papers.json       # Website-ready
├── research-papers-full.json  # Full metadata (merge state)
├── research-papers.csv        # Spreadsheet-importable
├── images/                    # Extracted thumbnails
└── raw/                       # Raw API responses for debugging
```

Each run merges into existing output:

```
★ 5 NEW papers added
✓ 47 already in database
✓ 2 papers updated (venue upgraded)
```

Use `--fresh` to start from scratch.

## Configuration

All configurable values live in two places. Edit these to customize what the pipeline searches for and how it classifies results.

### `src/ndif_citations/config.py`

**Discovery keywords** — what OpenAlex searches for in paper full text:

```python
OPENALEX_SEARCH_QUERIES = [
    "nnsight",                            # the Python library name
    '"national deep inference fabric"',   # full project name (exact phrase)
    "ndif.us",                            # the project URL
]
```

To add a new search term, append to this list. Double quotes inside the string trigger exact phrase matching in OpenAlex. Without them, OpenAlex matches any paper containing all the words in any order.

**PDF classification keywords** — what's searched in downloaded PDFs to extract context windows for the LLM classifier:

```python
NDIF_KEYWORDS = ["nnsight", "NNsight", "NDIF", "ndif.us", "nnsight.net", "import nnsight"]
```

These are case-insensitive during search. If a PDF contains none of these keywords, the paper defaults to `referencing` without an LLM call. Add terms here if NDIF gets a new URL or brand name.

**Seed paper** — the root of the Semantic Scholar citation graph:

```python
SEED_S2_ID = "ARXIV:2407.14561"
```

If NDIF publishes a second major paper that should also be tracked, add a second seed ID and extend `discover_s2_citations()` to query both.

**Rate limits** — adjust if you have authenticated API keys with higher limits:

```python
S2_RATE_LIMIT_SLEEP = 3.0       # seconds between S2 calls (unauthenticated)
LLM_RATE_LIMIT_SLEEP = 12.0     # NVIDIA Build free tier (~5 req/min)
OPENALEX_RATE_LIMIT_SLEEP = 0.15
GITHUB_RATE_LIMIT_SLEEP = 2.0
```

### `data/known_venues.json`

Controls venue formatting, peer-review detection, and venue-type classification. The pipeline checks paper venue strings against these lists:

```json
{
  "conferences": ["ICLR", "NeurIPS", "ICML", ...],
  "journals": ["JMLR", "TMLR", "Nature", ...],
  "preprint_servers": ["ArXiv", "BiorXiv", ...]
}
```

- **conferences** — papers matching these are tagged `peer_reviewed: true`, `venue_type: "conference"`.
- **journals** — same treatment, tagged `venue_type: "journal"`.
- **preprint_servers** — tagged `peer_reviewed: false`, `venue_type: "preprint"`.
- Papers at **workshops** (detected by the word "workshop" in the venue string) are tagged `peer_reviewed: true`, `venue_type: "workshop"`.

To add a new venue, append to the relevant list. Matching is case-insensitive substring — adding `"WCCI"` will match `"2026 IEEE WCCI"`, `"WCCI 2026"`, etc.

## Project structure

```
ndif-citations/
├── pyproject.toml
├── README.md
├── .env.example
├── data/known_venues.json
├── docs/
│   ├── architecture.svg
│   ├── pipeline-running.png
│   └── pipeline-report.png
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

## License

Internal tool for [NDIF](https://ndif.us) — an NSF-funded project at Northeastern University.
