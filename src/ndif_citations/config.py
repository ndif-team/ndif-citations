"""Configuration: environment variables, constants, and seed paper IDs."""

from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

# ---------------------------------------------------------------------------
# Seed papers -- the NDIF/NNsight papers we track citations of
# ---------------------------------------------------------------------------
# S2 has two distinct records of the seed paper (ICLR 2025 + 2024 preprint), each
# with a different set of citers (~6-paper overlap of ~30+ each). Querying both
# and unioning is required for full citation recall.
SEED_S2_IDS = [
    "ARXIV:2407.14561",                              # ICLR 2025 record
    "23b6a2c856a8f610b37e9534970fc49327df902e",      # 2024 preprint record (DOI-only externalIds)
]
SEED_ARXIV_ID = "2407.14561"
SEED_OPENREVIEW_ID = "MxbEiFRf39"  # ICLR 2025

# Papers published before this year cannot legitimately cite/use NDIF (the seed
# was released July 2024). Applied to ALL discovery sources to suppress noise
# (Scholar substring matches like "CNNsite", old physics papers, etc.).
# Papers with year=0 (unknown) are NOT filtered here — they fall through to
# the stub_metadata bucket where a human can review them.
MIN_PAPER_YEAR = 2024

# Exactly match these full lowercase titles to drop them from discovery (the origin papers themselves)
EXCLUDED_PAPER_TITLES = {
    "nnsight and ndif: democratizing access to open-weight foundation model internals",
}

NNSIGHT_GITHUB_REPO = "ndif-team/nnsight"
GITHUB_DEPENDENTS_URL = f"https://github.com/{NNSIGHT_GITHUB_REPO}/network/dependents"

# ---------------------------------------------------------------------------
# Semantic Scholar
# ---------------------------------------------------------------------------
S2_API_KEY: str | None = os.environ.get("S2_API_KEY") or None
S2_FIELDS = [
    "title", "authors", "abstract", "venue", "publicationDate",
    "publicationVenue", "externalIds", "openAccessPdf",
    "citationCount", "url",
]
S2_RATE_LIMIT_SLEEP = 3.0  # seconds between paginated calls (unauthenticated)
S2_MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# OpenAlex
# ---------------------------------------------------------------------------
OPENALEX_EMAIL: str | None = os.environ.get("OPENALEX_EMAIL") or None
OPENALEX_BASE_URL = "https://api.openalex.org"

# Fulltext search queries for OpenAlex paper discovery.
# Use double quotes inside the string for exact phrase matching.
OPENALEX_SEARCH_QUERIES = [
    "nnsight",                            # the Python library name (unique, low false-positive)
    '"national deep inference fabric"',   # full project name as exact phrase
    "ndif.us",                            # the project URL
]
OPENALEX_RATE_LIMIT_SLEEP = 0.15  # ~6-7 req/sec with polite pool

# ---------------------------------------------------------------------------
# SerpAPI (Google Scholar discovery)
# ---------------------------------------------------------------------------
# Recovers papers whose nnsight/NDIF citation S2 and OpenAlex bibliography
# parsers fail to link to the seed record. Free tier = 250 calls/month.
# Per-run usage: 3 calls for cited-by traversal + 6 calls for q="nnsight" = 9 calls.
SERPAPI_API_KEY: str | None = os.environ.get("SERPAPI_API_KEY") or None
SERPAPI_BASE_URL = "https://serpapi.com/search.json"
SCHOLAR_SEED_CLUSTER_ID = "8983123837757913252"  # Google Scholar cluster_id of the seed paper
SCHOLAR_KEYWORD_QUERIES = ["nnsight"]            # NDIF query confirmed redundant in testing
SCHOLAR_PAGE_SIZE = 20                           # Scholar SERP per-page max
SCHOLAR_MAX_PAGES_PER_QUERY = 8                  # safety cap (~160 results / ~8 calls per query)
SCHOLAR_CACHE_TTL_SECONDS = 86400                # 24h cache; --fresh forces refresh
SCHOLAR_RATE_LIMIT_SLEEP = 1.0                   # SerpAPI is generous, but be polite

# ---------------------------------------------------------------------------
# Unpaywall API (free, no key required, just need email)
# ---------------------------------------------------------------------------
UNPAYWALL_EMAIL: str | None = os.environ.get("UNPAYWALL_EMAIL") or OPENALEX_EMAIL
UNPAYWALL_RATE_LIMIT_SLEEP = 0.1  # ~10 req/sec (100K/day limit)

# ---------------------------------------------------------------------------
# LLM (OpenAI-compatible)
# ---------------------------------------------------------------------------
LLM_BASE_URL: str = os.environ.get("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
LLM_API_KEY: str | None = os.environ.get("LLM_API_KEY") or None
LLM_MODEL: str = os.environ.get("LLM_MODEL", "meta/llama-3.1-70b-instruct")
LLM_RATE_LIMIT_SLEEP = 12.0  # NVIDIA Build free tier is ~5 req/min

# ---------------------------------------------------------------------------
# GitHub scraping + API
# ---------------------------------------------------------------------------
GITHUB_TOKEN: str | None = os.environ.get("GITHUB_TOKEN") or None
GITHUB_RATE_LIMIT_SLEEP = 2.0  # seconds between dependents-page requests
GITHUB_API_BASE = "https://api.github.com"
GITHUB_API_TIMEOUT = 10
GITHUB_API_RATE_LIMIT_AUTH = 0.5   # seconds between calls when authenticated
GITHUB_API_RATE_LIMIT_ANON = 1.0   # seconds between calls when anonymous

EXCLUDED_GITHUB_REPOS: set[str] = {"ndif-team/nnsight"}  # repos to drop entirely
KNOWN_COURSE_SOURCES: set[str] = {"callummcdougall/ARENA_3.0"}  # repos whose forks → "course"
COURSE_NAME_PATTERNS: list[str] = ["ARENA", "MATS", "CBAI"]  # case-insensitive name/desc substrings
SHARED_PAPER_THRESHOLD: int = 5  # min repos sharing same linked_paper_url for template detection

# ---------------------------------------------------------------------------
# PDF / content processing -- keyword detection in paper text
# ---------------------------------------------------------------------------
# Keywords searched in PDF full text to extract context for LLM classification.
NDIF_KEYWORDS = ["nnsight", "NNsight", "NDIF", "ndif.us", "nnsight.net", "import nnsight"]

# Keywords searched in GitHub README text to detect NDIF infrastructure usage.
# Intentionally narrower than NDIF_KEYWORDS (no "nnsight" — all repos use it by definition).
# Split into two lists: regex (word-boundary, case-sensitive) and substring (case-insensitive).
NDIF_README_KEYWORDS_REGEX = [r"\bNDIF\b"]                    # word-boundary, case-sensitive
NDIF_README_KEYWORDS_SUBSTR = ["ndif.us", "NDIF cluster", "hosted on NDIF"]  # case-insensitive substring

# Boilerplate phrases that mention "NDIF" but don't indicate infrastructure use.
# A repo whose only NDIF mentions match these (case-insensitive substring) is NOT
# upgraded to uses_ndif. Common in nnsight install instructions and project labels.
NDIF_README_NEGATIVE_PATTERNS = [
    "NDIF Discord",        # "join the NDIF Discord community" — boilerplate install line
    "NDIF Pilot Program",  # heading/label used by some clones
    "join the NDIF",       # generic onboarding boilerplate
]

CONTEXT_WINDOW = 500  # chars around each keyword mention
MAX_CONTEXT_EXCERPTS = 5

# ---------------------------------------------------------------------------
# CrossRef API (free, no key — just polite User-Agent with email)
# ---------------------------------------------------------------------------
CROSSREF_BASE_URL = "https://api.crossref.org"
CROSSREF_RATE_LIMIT_SLEEP = 0.2  # ~5 req/sec polite pool

# ---------------------------------------------------------------------------
# arXiv API (Atom feed, free, no key)
# ---------------------------------------------------------------------------
ARXIV_API_BASE_URL = "https://export.arxiv.org/api/query"
ARXIV_API_RATE_LIMIT_SLEEP = 3.0  # arXiv requests ≥3s between requests
ARXIV_API_BATCH_SIZE = 100  # max id_list length per request

# ---------------------------------------------------------------------------
# Output paths
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = _PROJECT_ROOT / "output"

# ---------------------------------------------------------------------------
# Known venues (for peer-review detection)
# ---------------------------------------------------------------------------
_VENUES_FILE = _PROJECT_ROOT / "data" / "known_venues.json"
KNOWN_VENUES: dict = {}
if _VENUES_FILE.exists():
    with open(_VENUES_FILE) as f:
        KNOWN_VENUES = json.load(f)


def get_output_dir(custom: str | None = None) -> Path:
    """Return the output directory, creating it if needed."""
    out = Path(custom) if custom else DEFAULT_OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(exist_ok=True)
    (out / "raw").mkdir(exist_ok=True)
    return out
