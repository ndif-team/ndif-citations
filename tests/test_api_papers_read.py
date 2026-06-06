"""Tests for the papers read API — Task 4.1.

Covers:
  GET /api/papers               — list + filter + sort
  GET /api/papers/{paper_id}    — single paper by merge_key
  GET /api/stats                — dashboard aggregate counts
  GET /api/images/{slug}        — image serving + path-traversal guard

Fixture state (from mini-research-papers-full.json / mini-github-repos-full.json):
  Papers:
    verified  (3): "Activation Steering..." (arxiv:2602.16080),
                   "ADAG: Automatically..." (arxiv:2604.07615),
                   "Behind the Scenes..."   (title-based key, manual_override=True)
    pending   (1): "DFWe: Efficient..."     (title-based key)
    discarded (1): "Machine Learning..."    (arxiv:2604.08058)
  Repos (3): callummcdougall/ARENA_3.0 (course),
             hijohnnylin/neuronpedia  (research),
             saprmarks/dictionary_learning (research)
"""
from __future__ import annotations

import struct
import zlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from ndif_citations.server import deps
from ndif_citations.server.app import create_app


# ---------------------------------------------------------------------------
# Known IDs / titles from the fixture (update if fixture changes)
# ---------------------------------------------------------------------------

ARXIV_VERIFIED_ID = "arxiv:2602.16080"        # "Activation Steering via Generative Causal Mediation"
ARXIV_VERIFIED_TITLE = "Activation Steering via Generative Causal Mediation"

ARXIV_DISCARDED_ID = "arxiv:2604.08058"       # "Machine Learning the order-disorder Jahn-Teller..."

# Title-based key for "Behind the Scenes..." (no arxiv_id, no doi)
TITLE_VERIFIED_KEY = "title:behind the scenes: mechanistic interpretability of lora-adapted whisper for speech emotion recognition"

PENDING_TITLE_SUBSTR = "DFWe"                 # unique substring in the pending paper title


# ---------------------------------------------------------------------------
# Fixture: TestClient bound to the fixture_state output dir
# ---------------------------------------------------------------------------

@pytest.fixture()
def client(fixture_state: Path) -> TestClient:
    """Return a TestClient with get_output_dir overridden to fixture_state."""
    app = create_app()
    app.dependency_overrides[deps.get_output_dir] = lambda: fixture_state
    return TestClient(app, raise_server_exceptions=True)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _make_minimal_png(path: Path) -> None:
    """Write a tiny valid PNG file to *path* (1x1 red pixel)."""
    # A minimal 1x1 red pixel PNG, hand-crafted so the test has no Pillow dep.
    def _chunk(tag: bytes, data: bytes) -> bytes:
        c = zlib.crc32(tag + data)
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", c)

    ihdr_data = struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0)
    raw_row = b"\x00\xff\x00\x00"  # filter byte + RGB
    idat_data = zlib.compress(raw_row)
    png_bytes = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr_data)
        + _chunk(b"IDAT", idat_data)
        + _chunk(b"IEND", b"")
    )
    path.write_bytes(png_bytes)


# ---------------------------------------------------------------------------
# 1. GET /api/papers — all papers
# ---------------------------------------------------------------------------

def test_list_papers_returns_all(client: TestClient):
    resp = client.get("/api/papers")
    assert resp.status_code == 200
    rows = resp.json()

    # 5 papers total across all buckets
    assert len(rows) == 5

    # Each row must have the expected keys
    expected_keys = {
        "id", "title", "authors", "venue", "year", "category", "bucket",
        "confidence_band", "reason", "source", "has_image", "manual_override", "url",
        "missing",
    }
    for row in rows:
        assert expected_keys == set(row.keys()), f"Row keys mismatch: {set(row.keys())}"

    # A known title must be present
    titles = {r["title"] for r in rows}
    assert ARXIV_VERIFIED_TITLE in titles


# ---------------------------------------------------------------------------
# 2. GET /api/papers?bucket=verified — filter by bucket
# ---------------------------------------------------------------------------

def test_list_papers_filter_verified(client: TestClient):
    resp = client.get("/api/papers", params={"bucket": "verified"})
    assert resp.status_code == 200
    rows = resp.json()

    assert len(rows) == 3
    buckets = {r["bucket"] for r in rows}
    assert buckets == {"verified"}


def test_list_papers_filter_pending(client: TestClient):
    resp = client.get("/api/papers", params={"bucket": "pending"})
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["bucket"] == "pending"


def test_list_papers_filter_discarded(client: TestClient):
    resp = client.get("/api/papers", params={"bucket": "discarded"})
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert rows[0]["bucket"] == "discarded"


# ---------------------------------------------------------------------------
# 3. GET /api/papers?q=<substring> — text search
# ---------------------------------------------------------------------------

def test_list_papers_query_filter(client: TestClient):
    # "DFWe" appears only in the pending paper title
    resp = client.get("/api/papers", params={"q": PENDING_TITLE_SUBSTR})
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    assert PENDING_TITLE_SUBSTR in rows[0]["title"]


def test_list_papers_query_case_insensitive(client: TestClient):
    resp_lower = client.get("/api/papers", params={"q": "activation steering"})
    resp_upper = client.get("/api/papers", params={"q": "ACTIVATION STEERING"})
    assert resp_lower.status_code == 200
    assert resp_upper.status_code == 200
    assert len(resp_lower.json()) == 1
    assert resp_lower.json() == resp_upper.json()


def test_list_papers_query_no_match(client: TestClient):
    resp = client.get("/api/papers", params={"q": "zzz_no_such_paper_zzz"})
    assert resp.status_code == 200
    assert resp.json() == []


# ---------------------------------------------------------------------------
# 4. GET /api/papers?sort=... — ordering
# ---------------------------------------------------------------------------

def test_list_papers_sort_year_asc(client: TestClient):
    resp = client.get("/api/papers", params={"sort": "year_asc"})
    assert resp.status_code == 200
    rows = resp.json()
    years = [r["year"] for r in rows]
    assert years == sorted(years), "year_asc: years not ascending"


def test_list_papers_sort_year_desc_default(client: TestClient):
    resp_default = client.get("/api/papers")
    resp_desc = client.get("/api/papers", params={"sort": "year_desc"})
    assert resp_default.status_code == 200
    assert resp_desc.status_code == 200
    # default == explicit year_desc
    assert resp_default.json() == resp_desc.json()
    years = [r["year"] for r in resp_default.json()]
    assert years == sorted(years, reverse=True), "year_desc: years not descending"


def test_list_papers_sort_title(client: TestClient):
    resp = client.get("/api/papers", params={"sort": "title"})
    assert resp.status_code == 200
    rows = resp.json()
    titles = [r["title"].lower() for r in rows]
    assert titles == sorted(titles), "title sort: not alphabetical"


# ---------------------------------------------------------------------------
# 5. GET /api/papers/{paper_id} — single paper
# ---------------------------------------------------------------------------

def test_get_paper_by_arxiv_id(client: TestClient):
    resp = client.get(f"/api/papers/{ARXIV_VERIFIED_ID}")
    assert resp.status_code == 200
    data = resp.json()
    # to_full_dict() returns the full Pydantic model dump — check a few key fields
    assert data["title"] == ARXIV_VERIFIED_TITLE
    assert data["arxiv_id"] == "2602.16080"
    assert data["bucket"] == "verified"
    # Full dict should have many more fields than the list row
    assert "abstract" in data
    assert "category_confidence" in data


def test_get_paper_by_title_key(client: TestClient):
    resp = client.get(f"/api/papers/{TITLE_VERIFIED_KEY}")
    assert resp.status_code == 200
    data = resp.json()
    assert "behind the scenes" in data["title"].lower()
    assert data["manual_override"] is True


def test_get_paper_unknown_id_returns_404(client: TestClient):
    resp = client.get("/api/papers/arxiv:9999.00000")
    assert resp.status_code == 404


def test_get_paper_unknown_title_key_returns_404(client: TestClient):
    resp = client.get("/api/papers/title:no such paper exists in fixture")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 6. GET /api/stats — dashboard counts
# ---------------------------------------------------------------------------

def test_stats_paper_bucket_counts(client: TestClient):
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    data = resp.json()

    papers = data["papers"]
    assert papers["verified"] == 3
    assert papers["pending"] == 1
    assert papers["discarded"] == 1
    assert papers["total"] == 5


def test_stats_repo_type_counts(client: TestClient):
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    data = resp.json()

    repos = data["repos"]
    # 1 course (ARENA_3.0), 2 research (neuronpedia, dictionary_learning), 0 experiment
    assert repos["course"] == 1
    assert repos["research"] == 2
    assert repos["experiment"] == 0
    assert repos["total"] == 3


def test_stats_category_counts(client: TestClient):
    resp = client.get("/api/stats")
    assert resp.status_code == 200
    data = resp.json()

    cats = data["categories"]
    # From fixture:
    #   uses_nnsight: "Activation Steering..." (verified) + "Behind the Scenes..." (verified)
    #   referencing:  "ADAG..." (verified) + "Machine Learning..." (discarded)
    #   unclassified: "DFWe..." (pending)
    assert cats["uses_nnsight"] == 2
    assert cats["referencing"] == 2
    assert cats["unclassified"] == 1
    assert cats["uses_ndif"] == 0


# ---------------------------------------------------------------------------
# 7. GET /api/images/{slug} — image serving + path-traversal guard
# ---------------------------------------------------------------------------

def test_serve_image_returns_png(client: TestClient, fixture_state: Path):
    # Create a minimal valid PNG in fixture_state/images/
    img_path = fixture_state / "images" / "test.png"
    _make_minimal_png(img_path)

    resp = client.get("/api/images/test.png")
    assert resp.status_code == 200
    assert resp.headers["content-type"] == "image/png"
    assert resp.content == img_path.read_bytes()


def test_serve_image_missing_returns_404(client: TestClient):
    resp = client.get("/api/images/does-not-exist.png")
    assert resp.status_code == 404


def test_serve_image_path_traversal_slash_rejected(client: TestClient):
    # URL-encoded slash — FastAPI decodes %2F before routing; our guard catches it.
    resp = client.get("/api/images/..%2Fetc%2Fpasswd")
    assert resp.status_code == 404


def test_serve_image_path_traversal_dotdot_rejected(client: TestClient):
    resp = client.get("/api/images/..%2F..%2Fetc")
    assert resp.status_code == 404


def test_serve_image_bare_dotdot_rejected(client: TestClient):
    # FastAPI will reject a literal ``..`` path segment (returns 422/404 before
    # our handler runs), but we also handle any decoded variant.
    resp = client.get("/api/images/..%2Fsomething")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 8. missing field — metadata-gap flags on each row
# ---------------------------------------------------------------------------

def test_list_papers_missing_field_present(client: TestClient):
    """Every row must include a `missing` list (may be empty)."""
    resp = client.get("/api/papers")
    assert resp.status_code == 200
    rows = resp.json()
    for row in rows:
        assert "missing" in row, f"Row for {row['title']!r} is missing the 'missing' field"
        assert isinstance(row["missing"], list), "'missing' must be a list"


def test_list_papers_missing_image_flagged(client: TestClient, fixture_state):
    """A paper without an image must include 'image' in its missing list.

    The fixture 'DFWe' paper has no image set, so it must be flagged.
    """
    resp = client.get("/api/papers", params={"q": PENDING_TITLE_SUBSTR})
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    row = rows[0]
    assert not row["has_image"], "Fixture paper should have no image"
    assert "image" in row["missing"], (
        f"'image' should be in missing for a paper without a thumbnail; got {row['missing']}"
    )


def test_list_papers_missing_weak_venue_flagged(client: TestClient):
    """A paper whose venue matches the weak 'ArXiv YYYY' fallback must flag 'venue'."""
    # 'ADAG...' paper has venue='ArXiv 2026' which matches the weak pattern
    resp = client.get("/api/papers", params={"q": "ADAG"})
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    row = rows[0]
    assert "venue" in row["missing"], (
        f"'venue' should be flagged for ArXiv fallback venue; got {row['missing']}"
    )


def test_list_papers_missing_with_image_not_flagged(client: TestClient):
    """A paper that has an image must NOT include 'image' in its missing list."""
    resp = client.get("/api/papers", params={"q": "Activation Steering"})
    assert resp.status_code == 200
    rows = resp.json()
    assert len(rows) == 1
    row = rows[0]
    assert row["has_image"], "Fixture paper should have an image"
    assert "image" not in row["missing"], (
        f"'image' should NOT be in missing for a paper with a thumbnail; got {row['missing']}"
    )
