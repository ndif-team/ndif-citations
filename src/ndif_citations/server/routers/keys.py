"""API Keys settings — write-only secret management + Test-connection."""
from __future__ import annotations

import os
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from ndif_citations import config, key_validation, secrets_store
from ndif_citations.server import deps

router = APIRouter(prefix="/api", tags=["keys"])


def _env_path() -> Path:
    return config._PROJECT_ROOT / ".env"


class KeysPut(BaseModel):
    LLM_API_KEY: str | None = None
    S2_API_KEY: str | None = None
    GITHUB_TOKEN: str | None = None
    SERPAPI_API_KEY: str | None = None


class TestReq(BaseModel):
    provider: str  # "llm" | "github" | "s2" | "serpapi"


@router.get("/settings/keys")
def get_keys() -> dict:
    # Re-sync from the .env file first so out-of-band edits/removals reflect (F-002).
    status = secrets_store.refresh_secrets_from_file(_env_path())
    return {k: {"configured": v} for k, v in status.items()}


@router.put("/settings/keys")
def put_keys(body: KeysPut, _guard: None = Depends(deps.require_no_active_run)) -> dict:
    changes = {k: v for k, v in body.model_dump().items() if v is not None}
    try:
        status = secrets_store.set_keys(_env_path(), changes)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {k: {"configured": v} for k, v in status.items()}


@router.delete("/settings/keys/{key}")
def delete_key(key: str, _guard: None = Depends(deps.require_no_active_run)) -> dict:
    try:
        status = secrets_store.clear_key(_env_path(), key)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    return {k: {"configured": v} for k, v in status.items()}


@router.post("/settings/keys/test")
def test_key(body: TestReq) -> dict:
    p = body.provider
    if p == "llm":
        return key_validation.test_llm(config.LLM_BASE_URL, os.environ.get("LLM_API_KEY", ""), config.LLM_MODEL)
    if p == "github":
        return key_validation.test_github(os.environ.get("GITHUB_TOKEN", ""))
    if p == "s2":
        return key_validation.test_s2(os.environ.get("S2_API_KEY", ""))
    if p == "serpapi":
        return key_validation.test_serpapi(os.environ.get("SERPAPI_API_KEY", ""))
    raise HTTPException(status_code=422, detail=f"unknown provider {p!r}")
