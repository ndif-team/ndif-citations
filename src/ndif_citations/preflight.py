"""Pre-run credential check, keyed on which entities the run includes.

Papers are LLM-classified -> need LLM_API_KEY; repos are discovered + README
keyword-classified via the GitHub API -> need GITHUB_TOKEN. Optional keys only
warn.
"""
from __future__ import annotations

import os


def preflight(*, skip_papers: bool, skip_github: bool, validate: bool = False) -> dict:
    # Reads the live os.environ — kept in sync with config by secrets_store.set_keys()
    # (which writes os.environ + calls config.reload_settings before any run starts).
    # validate=False keeps this a pure, fast presence check. validate=True adds a
    # live GitHub token check (free api.github.com/user call) so a present-but-DEAD
    # token is caught before the run instead of failing mid-discovery. The LLM key is
    # intentionally NOT live-validated here (a completion costs tokens on every poll);
    # use the Settings -> API Keys "Test" button for that.
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
        token = os.environ.get("GITHUB_TOKEN")
        if not token:
            blocking.append("GITHUB_TOKEN is required to discover GitHub repos (anonymous GitHub is ~60 req/hr).")
        elif validate:
            from ndif_citations import key_validation
            res = key_validation.test_github(token)
            if not res["ok"]:
                blocking.append(
                    f"GITHUB_TOKEN is set but rejected by GitHub ({res['detail']}). "
                    "Update it in Settings → API Keys."
                )
    return {"ok": not blocking, "blocking": blocking, "warnings": warnings}
