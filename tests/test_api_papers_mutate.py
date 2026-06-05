"""Tests for the papers mutation API — Task 4.2.

Covers:
  PATCH /api/papers/{paper_id}           — edit fields
  POST  /api/papers/{paper_id}/bucket    — bucket moves (promote/demote/discard)

Fixture state (from mini-research-papers-full.json):
  verified  (3): "Activation Steering..." (arxiv:2602.16080),
                 "ADAG: Automatically..." (arxiv:2604.07615),
                 "Behind the Scenes..."   (title-based key, manual_override=True)
  pending   (1): "DFWe: Efficient..."     (title-based key)
  discarded (1): "Machine Learning..."    (arxiv:2604.08058)
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from ndif_citations.server import deps
from ndif_citations.server.app import create_app

# ---------------------------------------------------------------------------
# Known IDs from the fixture (mirrors test_api_papers_read.py)
# ---------------------------------------------------------------------------

ARXIV_VERIFIED_ID = "arxiv:2602.16080"        # verified, uses_nnsight, has description+image
ARXIV_PENDING_DISCARDED = "arxiv:2604.08058"  # discarded
ARXIV_ADAG_ID = "arxiv:2604.07615"            # verified, referencing, has affiliations

# Title-based key for "DFWe..." (pending, unclassified, no arxiv_id)
# Derived from DiscoveredPaper.merge_key(): "title:" + title.lower().strip()
PENDING_TITLE = "DFWe: Efficient Knowledge Distillation of Fine-tuned Whisper Encoder for Speech Emotion Recognition"
PENDING_ID = f"title:{PENDING_TITLE.lower().strip()}"

UNKNOWN_ID = "arxiv:9999.00000"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(fixture_state: Path) -> TestClient:
    """TestClient with get_output_dir overridden to fixture_state."""
    app = create_app()
    app.dependency_overrides[deps.get_output_dir] = lambda: fixture_state
    return TestClient(app, raise_server_exceptions=True)


@pytest.fixture()
def client_active_run(fixture_state: Path) -> TestClient:
    """TestClient where the runner reports an active run (for 409 tests)."""
    app = create_app()
    app.dependency_overrides[deps.get_output_dir] = lambda: fixture_state

    # Stub runner with active=True
    stub_runner = MagicMock()
    stub_runner.active = True
    app.dependency_overrides[deps.get_runner] = lambda: stub_runner

    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# 1. PATCH — edit a field
# ---------------------------------------------------------------------------

def test_edit_field(client: TestClient, fixture_state: Path):
    """PATCH a single field → 200; value updated; persists across a GET."""
    resp = client.patch(
        f"/api/papers/{ARXIV_VERIFIED_ID}",
        json={"fields": {"venue": "ICML 2025"}},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["venue"] == "ICML 2025"
    assert data["manual_override"] is True

    # Confirm persistence: re-GET must reflect the change
    get_resp = client.get(f"/api/papers/{ARXIV_VERIFIED_ID}")
    assert get_resp.status_code == 200
    assert get_resp.json()["venue"] == "ICML 2025"


# ---------------------------------------------------------------------------
# 2. PATCH — editing fields updates has_* and can re-derive bucket
# ---------------------------------------------------------------------------

def test_edit_rederives_has_classification(client: TestClient):
    """Setting category to a non-UNCLASSIFIED value sets has_classification=True."""
    # The pending paper is currently unclassified (has_classification=False).
    # Editing category to uses_nnsight should flip has_classification.
    resp = client.patch(
        f"/api/papers/{PENDING_ID}",
        json={"fields": {"category": "uses_nnsight"}},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["has_classification"] is True
    assert data["manual_override"] is True


def test_edit_sets_has_summary_when_description_set(client: TestClient):
    """Setting a description sets has_summary=True."""
    resp = client.patch(
        f"/api/papers/{ARXIV_VERIFIED_ID}",
        json={"fields": {"description": "A new curator-written summary."}},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["has_summary"] is True
    assert data["description"] == "A new curator-written summary."


# ---------------------------------------------------------------------------
# 3. PATCH — unknown field → 422
# ---------------------------------------------------------------------------

def test_edit_unknown_field_422(client: TestClient):
    resp = client.patch(
        f"/api/papers/{ARXIV_VERIFIED_ID}",
        json={"fields": {"bogus": "x"}},
    )
    assert resp.status_code == 422, resp.text


# ---------------------------------------------------------------------------
# 4. PATCH — parse error → 422
# ---------------------------------------------------------------------------

def test_edit_parse_error_422(client: TestClient):
    """year expects an integer; passing 'NaN' should return 422."""
    resp = client.patch(
        f"/api/papers/{ARXIV_VERIFIED_ID}",
        json={"fields": {"year": "NaN"}},
    )
    assert resp.status_code == 422, resp.text


# ---------------------------------------------------------------------------
# 5. PATCH — unknown paper → 404
# ---------------------------------------------------------------------------

def test_edit_unknown_paper_404(client: TestClient):
    resp = client.patch(
        f"/api/papers/{UNKNOWN_ID}",
        json={"fields": {"venue": "ICML 2025"}},
    )
    assert resp.status_code == 404, resp.text


# ---------------------------------------------------------------------------
# 6. POST /bucket — promote to verified
# ---------------------------------------------------------------------------

def test_set_bucket_promote(client: TestClient):
    """POST bucket=verified on a pending paper → 200, bucket=='verified', manual_override==true."""
    resp = client.post(
        f"/api/papers/{PENDING_ID}/bucket",
        json={"bucket": "verified"},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["bucket"] == "verified"
    assert data["manual_override"] is True


# ---------------------------------------------------------------------------
# 7. POST /bucket — discard with reason + detail
# ---------------------------------------------------------------------------

def test_set_bucket_discard_with_reason(client: TestClient):
    """POST bucket=discarded with reason and detail → 200; all fields set."""
    resp = client.post(
        f"/api/papers/{ARXIV_VERIFIED_ID}/bucket",
        json={
            "bucket": "discarded",
            "reason": "manual_discard",
            "detail": "junk",
        },
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["bucket"] == "discarded"
    assert data["reason"] == "manual_discard"
    assert data["reason_detail"] == "junk"
    assert data["manual_override"] is True


# ---------------------------------------------------------------------------
# 8. POST /bucket — bad reason → 422
# ---------------------------------------------------------------------------

def test_set_bucket_bad_reason_422(client: TestClient):
    resp = client.post(
        f"/api/papers/{ARXIV_VERIFIED_ID}/bucket",
        json={"bucket": "pending", "reason": "not_a_reason"},
    )
    assert resp.status_code == 422, resp.text


# ---------------------------------------------------------------------------
# 9. Mutations blocked during active run → 409
# ---------------------------------------------------------------------------

def test_mutation_blocked_during_active_run_patch_409(client_active_run: TestClient):
    """PATCH is blocked with 409 when a pipeline run is active."""
    resp = client_active_run.patch(
        f"/api/papers/{ARXIV_VERIFIED_ID}",
        json={"fields": {"venue": "ICML 2025"}},
    )
    assert resp.status_code == 409, resp.text


def test_mutation_blocked_during_active_run_bucket_409(client_active_run: TestClient):
    """POST /bucket is blocked with 409 when a pipeline run is active."""
    resp = client_active_run.post(
        f"/api/papers/{ARXIV_VERIFIED_ID}/bucket",
        json={"bucket": "verified"},
    )
    assert resp.status_code == 409, resp.text
