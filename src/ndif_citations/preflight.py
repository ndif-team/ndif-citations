"""Pre-run credential check, keyed on which entities the run includes.

Papers are LLM-classified -> need LLM_API_KEY; repos are discovered + README
keyword-classified via the GitHub API -> need GITHUB_TOKEN. Optional keys only
warn.

validate_llm=True is never set by the polled GET /runs/preflight; its callers are CLI run,
POST /runs and the Phase 3 LLM-using commands (spec 4.2).
"""
from __future__ import annotations

import os


def preflight(*, skip_papers: bool, skip_github: bool, validate: bool = False, validate_llm: bool = False) -> dict:
    # Reads the live os.environ — kept in sync with config by secrets_store.set_keys()
    # (which writes os.environ + calls config.reload_settings before any run starts).
    # validate=False keeps this a pure, fast presence check. validate=True checks
    # GitHub via /rate_limit so a present-but-DEAD token is caught before the run
    # instead of failing mid-discovery. validate_llm=True sends one tiny completion
    # with the real model + extra body (costs tokens, so never on the polled endpoint).
    blocking: list[str] = []
    warnings: list[str] = []
    if not skip_papers:
        llm_key = os.environ.get("LLM_API_KEY")
        if not llm_key:
            blocking.append("LLM_API_KEY is required to process papers (LLM classification/summary).")
        elif validate_llm:
            # One tiny completion with the REAL model + extra body: catches retired
            # models (HTTP 410) and reasoning models returning empty content.
            from ndif_citations import config, key_validation
            res = key_validation.test_llm(config.LLM_BASE_URL, llm_key, config.LLM_MODEL, config.LLM_EXTRA_BODY)
            if not res["ok"]:
                blocking.append(
                    f"LLM check failed ({res['detail']}) — model {config.LLM_MODEL} at {config.LLM_BASE_URL}"
                )
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
