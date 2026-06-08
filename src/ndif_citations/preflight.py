"""Pre-run credential check, keyed on which entities the run includes.

Papers are LLM-classified -> need LLM_API_KEY; repos are discovered + README
keyword-classified via the GitHub API -> need GITHUB_TOKEN. Optional keys only
warn.
"""
from __future__ import annotations

import os


def preflight(*, skip_papers: bool, skip_github: bool) -> dict:
    blocking: list[str] = []
    warnings: list[str] = []
    if not skip_papers:
        if not os.environ.get("LLM_API_KEY"):
            blocking.append("LLM_API_KEY is required to process papers (LLM classification/summary).")
        if not os.environ.get("S2_API_KEY"):
            warnings.append("S2_API_KEY not set — Semantic Scholar discovery will be rate-limited.")
        if not os.environ.get("SERPAPI_API_KEY"):
            warnings.append("SERPAPI_API_KEY not set — Google Scholar discovery disabled.")
    if not skip_github:
        if not os.environ.get("GITHUB_TOKEN"):
            blocking.append("GITHUB_TOKEN is required to discover GitHub repos (anonymous GitHub is ~60 req/hr).")
    return {"ok": not blocking, "blocking": blocking, "warnings": warnings}
