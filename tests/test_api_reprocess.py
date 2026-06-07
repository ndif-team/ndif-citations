"""API tests for targeted reprocess + reextract-thumbnail + image upload (Task 4.3).

Endpoints under test:
  POST /api/papers/{paper_id}/reprocess           — queue a reprocess job
  POST /api/papers/{paper_id}/reextract-thumbnail — queue a thumbnail-only job
  POST /api/papers/{paper_id}/image               — synchronous PNG upload

The reprocess endpoints run heavy work on the JobRunner worker; the client
watches via GET /api/runs/{run_id}. Tests patch the process-module callables via
``install_pipeline_fakes`` and additionally override ``generate_summary`` /
``extract_thumbnail`` to return deterministic NEW values.

Fixture IDs (mini-research-papers-full.json):
  arxiv:2602.16080 — verified, uses_nnsight, has description+image (mo=False)
"""
from __future__ import annotations

import io
import struct
import threading
import time
import zlib
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from ndif_citations import orchestrator
from ndif_citations.jobs import JobRunner
from ndif_citations.server import deps
from ndif_citations.server.app import create_app
from tests.helpers.fakes import install_pipeline_fakes

ARXIV_VERIFIED_ID = "arxiv:2602.16080"
ARXIV_VERIFIED_ID_2 = "arxiv:2604.07615"  # second verified paper in fixtures
UNKNOWN_ID = "arxiv:9999.00000"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _wait_until(predicate, timeout: float = 4.0, interval: float = 0.02) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


def _make_client(monkeypatch, fixture_state, runner: JobRunner | None = None):
    """TestClient with pipeline fakes + output-dir + runner overrides."""
    install_pipeline_fakes(monkeypatch, orchestrator)
    _runner = runner if runner is not None else JobRunner()
    app = create_app()
    app.dependency_overrides[deps.get_output_dir] = lambda: fixture_state
    app.dependency_overrides[deps.get_runner] = lambda: _runner
    return TestClient(app, raise_server_exceptions=True), _runner


def _tiny_png() -> bytes:
    """Build a minimal valid 1x1 PNG (correct magic bytes + IHDR/IDAT/IEND)."""
    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = _chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    raw = b"\x00\xff\xff\xff"  # one filtered scanline: filter 0 + RGB white
    idat = _chunk(b"IDAT", zlib.compress(raw))
    iend = _chunk(b"IEND", b"")
    return sig + ihdr + idat + iend


# ---------------------------------------------------------------------------
# 4. reprocess endpoint runs a job → description changes.
# ---------------------------------------------------------------------------

def test_reprocess_endpoint_runs_job(monkeypatch, fixture_state):
    import ndif_citations.process as process_mod

    # _make_client installs the pipeline fakes, so override AFTERWARDS so our
    # deterministic NEW value wins.
    client, runner = _make_client(monkeypatch, fixture_state)
    monkeypatch.setattr(
        process_mod, "generate_summary", lambda paper: "REPROCESSED SUMMARY"
    )

    # Sanity: the curated description differs from the new value beforehand.
    before = client.get(f"/api/papers/{ARXIV_VERIFIED_ID}").json()
    assert before["description"] != "REPROCESSED SUMMARY"

    resp = client.post(
        f"/api/papers/{ARXIV_VERIFIED_ID}/reprocess",
        json={"fields": ["summary"]},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "run_id" in data and data["state"] == "running"
    run_id = data["run_id"]

    assert _wait_until(
        lambda: client.get(f"/api/runs/{run_id}").json().get("state") == "done"
    ), f"job did not finish; state={client.get(f'/api/runs/{run_id}').json().get('state')!r}"

    after = client.get(f"/api/papers/{ARXIV_VERIFIED_ID}").json()
    assert after["description"] == "REPROCESSED SUMMARY"
    assert after["manual_override"] is True


# ---------------------------------------------------------------------------
# 5. reextract-thumbnail endpoint runs a thumbnail-only job.
# ---------------------------------------------------------------------------

def test_reextract_thumbnail_endpoint(monkeypatch, fixture_state):
    import ndif_citations.process as process_mod
    import ndif_citations.pdf_cache as pdf_cache_mod

    # _make_client installs the pipeline fakes; override AFTERWARDS so the
    # thumbnail branch runs with a (fake) PDF + a deterministic new image path.
    client, runner = _make_client(monkeypatch, fixture_state)

    fake_pdf = fixture_state / "pdfs" / "fake.pdf"
    fake_pdf.parent.mkdir(parents=True, exist_ok=True)
    fake_pdf.write_bytes(b"%PDF-1.4 fake")

    monkeypatch.setattr(
        pdf_cache_mod, "get_cached_pdf", lambda paper, out: fake_pdf
    )

    def _fake_extract(paper, output_dir, pdf_path=None):
        return "/images/REEXTRACTED.png"

    monkeypatch.setattr(process_mod, "extract_thumbnail", _fake_extract)

    resp = client.post(f"/api/papers/{ARXIV_VERIFIED_ID}/reextract-thumbnail")
    assert resp.status_code == 200, resp.text
    run_id = resp.json()["run_id"]

    assert _wait_until(
        lambda: client.get(f"/api/runs/{run_id}").json().get("state") == "done"
    ), f"job state={client.get(f'/api/runs/{run_id}').json().get('state')!r}"

    after = client.get(f"/api/papers/{ARXIV_VERIFIED_ID}").json()
    assert after["image"] == "/images/REEXTRACTED.png"


# ---------------------------------------------------------------------------
# 6. reprocess 404 / 422 mapping.
# ---------------------------------------------------------------------------

def test_reprocess_unknown_paper_404(monkeypatch, fixture_state):
    client, runner = _make_client(monkeypatch, fixture_state)
    resp = client.post(
        f"/api/papers/{UNKNOWN_ID}/reprocess", json={"fields": ["summary"]}
    )
    assert resp.status_code == 404, resp.text


def test_reprocess_invalid_field_422(monkeypatch, fixture_state):
    client, runner = _make_client(monkeypatch, fixture_state)
    resp = client.post(
        f"/api/papers/{ARXIV_VERIFIED_ID}/reprocess", json={"fields": ["bogus"]}
    )
    assert resp.status_code == 422, resp.text


def test_reprocess_empty_fields_422(monkeypatch, fixture_state):
    client, runner = _make_client(monkeypatch, fixture_state)
    resp = client.post(
        f"/api/papers/{ARXIV_VERIFIED_ID}/reprocess", json={"fields": []}
    )
    assert resp.status_code == 422, resp.text


# ---------------------------------------------------------------------------
# 7. reprocess 409 when a run/job is already active.
# ---------------------------------------------------------------------------

def test_reprocess_conflict_409(monkeypatch, fixture_state):
    """With a blocking pipeline run active, reprocess → 409."""
    import ndif_citations.process as process_mod

    install_pipeline_fakes(monkeypatch, orchestrator)

    release = threading.Event()
    entered = threading.Event()

    def _blocking_summary(paper):
        entered.set()
        release.wait(timeout=5.0)
        return f"Fake summary for: {paper.title}"

    monkeypatch.setattr(process_mod, "generate_summary", _blocking_summary)

    runner = JobRunner()
    app = create_app()
    app.dependency_overrides[deps.get_output_dir] = lambda: fixture_state
    app.dependency_overrides[deps.get_runner] = lambda: runner
    client = TestClient(app, raise_server_exceptions=True)

    # Start a blocking pipeline run.
    resp1 = client.post("/api/runs", json={"mode": "fresh"})
    assert resp1.status_code == 200, resp1.text
    assert entered.wait(timeout=2.0), "worker never reached blocking summary"
    assert runner.active

    # Reprocess while active → 409.
    resp2 = client.post(
        f"/api/papers/{ARXIV_VERIFIED_ID}/reprocess", json={"fields": ["summary"]}
    )
    assert resp2.status_code == 409, resp2.text

    release.set()
    assert _wait_until(lambda: runner.status().state == "done")


# ---------------------------------------------------------------------------
# 8. image upload — happy path + 422 (non-PNG) + 404 + 409.
# ---------------------------------------------------------------------------

def test_image_upload(monkeypatch, fixture_state):
    client, runner = _make_client(monkeypatch, fixture_state)
    png = _tiny_png()

    resp = client.post(
        f"/api/papers/{ARXIV_VERIFIED_ID}/image",
        files={"file": ("thumb.png", io.BytesIO(png), "image/png")},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["has_thumbnail"] is True
    assert data["manual_override"] is True
    assert data["image"].startswith("/images/")

    # The file was actually written to fixture_state/images/.
    filename = Path(data["image"]).name
    saved = fixture_state / "images" / filename
    assert saved.exists()
    assert saved.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_image_upload_non_png_422(monkeypatch, fixture_state):
    client, runner = _make_client(monkeypatch, fixture_state)
    resp = client.post(
        f"/api/papers/{ARXIV_VERIFIED_ID}/image",
        files={"file": ("notes.txt", io.BytesIO(b"hello world"), "text/plain")},
    )
    assert resp.status_code == 422, resp.text


def test_image_upload_unknown_paper_404(monkeypatch, fixture_state):
    client, runner = _make_client(monkeypatch, fixture_state)
    resp = client.post(
        f"/api/papers/{UNKNOWN_ID}/image",
        files={"file": ("thumb.png", io.BytesIO(_tiny_png()), "image/png")},
    )
    assert resp.status_code == 404, resp.text


def test_image_upload_active_run_409(fixture_state):
    """During an active run, image upload → 409 (require_no_active_run guard)."""
    app = create_app()
    app.dependency_overrides[deps.get_output_dir] = lambda: fixture_state
    stub = MagicMock()
    stub.active = True
    app.dependency_overrides[deps.get_runner] = lambda: stub
    client = TestClient(app, raise_server_exceptions=True)

    resp = client.post(
        f"/api/papers/{ARXIV_VERIFIED_ID}/image",
        files={"file": ("thumb.png", io.BytesIO(_tiny_png()), "image/png")},
    )
    assert resp.status_code == 409, resp.text


# ---------------------------------------------------------------------------
# Batch reprocess: POST /api/papers/reprocess
# ---------------------------------------------------------------------------

def test_batch_reprocess_starts_job(monkeypatch, fixture_state):
    """Batch reprocess of 2 known ids starts a job and runs to completion."""
    import ndif_citations.process as process_mod

    client, runner = _make_client(monkeypatch, fixture_state)
    monkeypatch.setattr(
        process_mod, "generate_summary", lambda paper: f"BATCH SUMMARY: {paper.title}"
    )

    resp = client.post(
        "/api/papers/reprocess",
        json={"ids": [ARXIV_VERIFIED_ID, ARXIV_VERIFIED_ID_2], "fields": ["summary"]},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert "run_id" in data
    assert data["state"] == "running"
    run_id = data["run_id"]

    assert _wait_until(
        lambda: client.get(f"/api/runs/{run_id}").json().get("state") == "done"
    ), f"job did not finish; state={client.get(f'/api/runs/{run_id}').json().get('state')!r}"

    for paper_id in (ARXIV_VERIFIED_ID, ARXIV_VERIFIED_ID_2):
        after = client.get(f"/api/papers/{paper_id}").json()
        assert after["description"].startswith("BATCH SUMMARY:"), (
            f"{paper_id}: unexpected description={after['description']!r}"
        )


def test_batch_reprocess_unknown_id_404(monkeypatch, fixture_state):
    """A batch with one unknown id → 404 and no job started."""
    client, runner = _make_client(monkeypatch, fixture_state)
    resp = client.post(
        "/api/papers/reprocess",
        json={"ids": [ARXIV_VERIFIED_ID, UNKNOWN_ID], "fields": ["summary"]},
    )
    assert resp.status_code == 404, resp.text


def test_batch_reprocess_bad_field_422(monkeypatch, fixture_state):
    """A batch with an invalid field name → 422."""
    client, runner = _make_client(monkeypatch, fixture_state)
    resp = client.post(
        "/api/papers/reprocess",
        json={"ids": [ARXIV_VERIFIED_ID], "fields": ["bogus"]},
    )
    assert resp.status_code == 422, resp.text


def test_batch_reprocess_empty_ids_422(monkeypatch, fixture_state):
    """An empty ids list → 422."""
    client, runner = _make_client(monkeypatch, fixture_state)
    resp = client.post(
        "/api/papers/reprocess",
        json={"ids": [], "fields": ["summary"]},
    )
    assert resp.status_code == 422, resp.text


def test_batch_reprocess_active_run_409(monkeypatch, fixture_state):
    """When a run is already active, batch reprocess → 409."""
    import ndif_citations.process as process_mod

    install_pipeline_fakes(monkeypatch, orchestrator)

    release = threading.Event()
    entered = threading.Event()

    def _blocking_summary(paper):
        entered.set()
        release.wait(timeout=5.0)
        return f"Fake summary for: {paper.title}"

    monkeypatch.setattr(process_mod, "generate_summary", _blocking_summary)

    runner = JobRunner()
    app = create_app()
    app.dependency_overrides[deps.get_output_dir] = lambda: fixture_state
    app.dependency_overrides[deps.get_runner] = lambda: runner
    client = TestClient(app, raise_server_exceptions=True)

    # Start a blocking pipeline run to hold the runner slot.
    resp1 = client.post("/api/runs", json={"mode": "fresh"})
    assert resp1.status_code == 200, resp1.text
    assert entered.wait(timeout=2.0), "worker never reached blocking summary"
    assert runner.active

    # Batch reprocess while active → 409.
    resp2 = client.post(
        "/api/papers/reprocess",
        json={"ids": [ARXIV_VERIFIED_ID], "fields": ["summary"]},
    )
    assert resp2.status_code == 409, resp2.text

    release.set()
    assert _wait_until(lambda: runner.status().state == "done")
