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
SEED_S2_ID = "ARXIV:2407.14561"  # NNsight & NDIF paper
SEED_ARXIV_ID = "2407.14561"
SEED_OPENREVIEW_ID = "MxbEiFRf39"  # ICLR 2025

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
# LLM (OpenAI-compatible)
# ---------------------------------------------------------------------------
LLM_BASE_URL: str = os.environ.get("LLM_BASE_URL", "https://integrate.api.nvidia.com/v1")
LLM_API_KEY: str | None = os.environ.get("LLM_API_KEY") or None
LLM_MODEL: str = os.environ.get("LLM_MODEL", "meta/llama-3.1-70b-instruct")
LLM_RATE_LIMIT_SLEEP = 12.0  # NVIDIA Build free tier is ~5 req/min

# ---------------------------------------------------------------------------
# GitHub scraping
# ---------------------------------------------------------------------------
GITHUB_RATE_LIMIT_SLEEP = 2.0  # seconds between requests

# ---------------------------------------------------------------------------
# PDF / content processing -- keyword detection in paper text
# ---------------------------------------------------------------------------
# Keywords searched in PDF full text to extract context for LLM classification.
NDIF_KEYWORDS = ["nnsight", "NNsight", "NDIF", "ndif.us", "nnsight.net", "import nnsight"]
CONTEXT_WINDOW = 500  # chars around each keyword mention
MAX_CONTEXT_EXCERPTS = 5

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
