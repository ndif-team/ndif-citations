"""Service helpers for the papers API.

Read functions are pure (no side-effects). Mutation functions (edit_paper,
set_bucket) load, mutate, and persist via write_outputs.

All functions operate on a given output_dir Path so they are trivially
testable via dependency_overrides.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from ndif_citations.models import Bucket, Category, DiscoveredPaper, PaperReason, PipelineRun
from ndif_citations.output import load_existing_papers, write_outputs
from ndif_citations.utils import slugify
from ndif_citations.venue import _WEAK_VENUE_RE

if TYPE_CHECKING:
    from fastapi import UploadFile

# PNG file signature (magic bytes) — first 8 bytes of every PNG.
_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def resolve(out: Path, paper_id: str) -> DiscoveredPaper | None:
    """Find the paper whose ``merge_key() == paper_id``.

    Shared lookup used by read endpoints and later mutation tasks.
    Returns ``None`` if the paper is not found.
    """
    papers = load_existing_papers(out)
    for paper in papers:
        if paper.merge_key() == paper_id:
            return paper
    return None


def _compute_missing(paper: DiscoveredPaper) -> list[str]:
    """Return list of important empty fields for the paper row."""
    missing: list[str] = []
    if not paper.image:
        missing.append("image")
    if not paper.affiliations:
        missing.append("affiliations")
    if not paper.abstract:
        missing.append("abstract")
    if not paper.description:
        missing.append("summary")
    if not paper.venue or _WEAK_VENUE_RE.match(paper.venue):
        missing.append("venue")
    return missing


def _paper_to_row(paper: DiscoveredPaper) -> dict:
    """Convert a DiscoveredPaper to the list-row dict shape."""
    return {
        "id": paper.merge_key(),
        "title": paper.title,
        "authors": paper.authors,
        "venue": paper.venue,
        "year": paper.year,
        "category": paper.category.value,
        "bucket": paper.bucket.value,
        "confidence_band": paper.category_confidence_band.value,
        "reason": paper.reason.value if paper.reason else None,
        "source": paper.source.value,
        "has_image": bool(paper.image),
        "manual_override": paper.manual_override,
        "url": paper.url,
        "missing": _compute_missing(paper),
    }


def list_rows(
    out: Path,
    *,
    bucket: str | None = None,
    q: str | None = None,
    sort: str = "year_desc",
) -> list[dict]:
    """Return a filtered, sorted list of paper row dicts.

    Parameters
    ----------
    out:
        Output directory (passed to ``load_existing_papers``).
    bucket:
        Optional filter: ``"pending"``, ``"verified"``, or ``"discarded"``.
        If ``None`` all buckets are included.
    q:
        Optional case-insensitive substring search over title + authors + venue.
    sort:
        ``"year_desc"`` (default), ``"year_asc"``, or ``"title"``.
    """
    papers = load_existing_papers(out)

    # Filter by bucket
    if bucket is not None:
        papers = [p for p in papers if p.bucket.value == bucket]

    # Substring search
    if q is not None:
        q_lower = q.lower()
        papers = [
            p for p in papers
            if q_lower in p.title.lower()
            or q_lower in p.authors.lower()
            or q_lower in p.venue.lower()
        ]

    # Sort
    if sort == "year_asc":
        papers.sort(key=lambda p: (p.year, p.title.lower()))
    elif sort == "title":
        papers.sort(key=lambda p: p.title.lower())
    else:  # year_desc (default)
        papers.sort(key=lambda p: (-p.year, p.title.lower()))

    return [_paper_to_row(p) for p in papers]


def get_paper(out: Path, paper_id: str) -> dict | None:
    """Return ``to_full_dict()`` for the paper with the given merge_key, or None.

    Adds two computed keys:
    - ``missing``: list of important empty fields (from ``_compute_missing``).
    - ``has_pdf``: True if a cached PDF exists in ``out/pdfs/``.
    """
    paper = resolve(out, paper_id)
    if paper is None:
        return None
    from ndif_citations.pdf_cache import cached_pdf_path
    d = paper.to_full_dict()
    d["missing"] = _compute_missing(paper)
    d["has_pdf"] = cached_pdf_path(paper, out) is not None
    return d


# ---------------------------------------------------------------------------
# Mutation helpers
# ---------------------------------------------------------------------------

def edit_paper(out: Path, paper_id: str, fields: dict[str, str]) -> dict:
    """Edit one or more fields on a paper and persist.

    Parameters
    ----------
    out:
        Output directory (passed to ``load_existing_papers`` / ``write_outputs``).
    paper_id:
        Paper merge_key (e.g. ``arxiv:2407.14561``).
    fields:
        Mapping of editable field name → raw string value to parse.

    Returns the updated paper's ``to_full_dict()``.

    Raises
    ------
    KeyError:
        If no paper with *paper_id* exists.
    ValueError:
        If a field name is unknown/non-editable, or if parsing fails.
    """
    from ndif_citations import edit_schema
    from ndif_citations.process import _decide_bucket

    papers = load_existing_papers(out)
    paper: DiscoveredPaper | None = None
    for p in papers:
        if p.merge_key() == paper_id:
            paper = p
            break
    if paper is None:
        raise KeyError(paper_id)

    # Apply each field edit
    for name, raw in fields.items():
        f = edit_schema.get_field(name)
        if f is None:
            raise ValueError(f"unknown/non-editable field: {name!r}")
        value = f.parse(raw)  # may raise ValueError for bad input
        setattr(paper, f.name, value)

    # Freeze with manual_override and re-derive has_* flags
    paper.manual_override = True
    paper.has_summary = bool(paper.description)
    paper.has_classification = paper.category != Category.UNCLASSIFIED
    paper.has_thumbnail = bool(paper.image)
    paper.has_affiliations = bool(paper.affiliations)

    # Re-run _decide_bucket only when bucket was not explicitly edited
    if "bucket" not in fields:
        paper.bucket, paper.reason = _decide_bucket(paper)

    write_outputs(papers, out, PipelineRun())
    return paper.to_full_dict()


def set_bucket(
    out: Path,
    paper_id: str,
    bucket: str,
    reason: str | None,
    detail: str | None,
) -> dict:
    """Move a paper to a different bucket and persist.

    Mirrors the promote / demote / discard CLI commands, unified into one
    call where the caller supplies bucket + reason.

    Parameters
    ----------
    out:
        Output directory.
    paper_id:
        Paper merge_key.
    bucket:
        One of ``"pending"``, ``"verified"``, ``"discarded"``.
    reason:
        ``PaperReason`` string value, or ``None`` (allowed for verified).
    detail:
        Free-text supplement stored in ``reason_detail``; may be ``None``.

    Returns the updated paper's ``to_full_dict()``.

    Raises
    ------
    KeyError:
        If no paper with *paper_id* exists.
    ValueError:
        If *bucket* is not a valid ``Bucket`` value, or *reason* is not a
        valid ``PaperReason`` value.
    """
    _valid_buckets = {"pending", "verified", "discarded"}
    if bucket not in _valid_buckets:
        raise ValueError(f"invalid bucket {bucket!r}; must be one of {sorted(_valid_buckets)}")

    papers = load_existing_papers(out)
    paper: DiscoveredPaper | None = None
    for p in papers:
        if p.merge_key() == paper_id:
            paper = p
            break
    if paper is None:
        raise KeyError(paper_id)

    paper.bucket = Bucket(bucket)
    paper.reason = PaperReason(reason) if reason is not None else None  # raises ValueError for bad value
    paper.reason_detail = detail
    paper.manual_override = True

    write_outputs(papers, out, PipelineRun())
    return paper.to_full_dict()


def upload_image(out: Path, paper_id: str, file: "UploadFile") -> dict:
    """Save an uploaded PNG as a paper's thumbnail and persist.

    The bytes are written to ``out/images/{slugify(title)}.png`` (the same path
    convention the pipeline uses), and the paper's ``image`` /
    ``has_thumbnail`` / ``manual_override`` fields are updated.

    Parameters
    ----------
    out:
        Output directory.
    paper_id:
        Paper merge_key.
    file:
        The uploaded file (FastAPI ``UploadFile``).

    Returns the updated paper's ``to_full_dict()``.

    Raises
    ------
    KeyError:
        If no paper with *paper_id* exists.
    ValueError:
        If the upload is not a PNG (by content-type or magic bytes).
    """
    papers = load_existing_papers(out)
    paper: DiscoveredPaper | None = None
    for p in papers:
        if p.merge_key() == paper_id:
            paper = p
            break
    if paper is None:
        raise KeyError(paper_id)

    data = file.file.read()
    # Require PNG magic bytes regardless of declared content-type — content-type
    # alone is attacker-controlled and not a reliable signal.
    if data[:8] != _PNG_MAGIC:
        raise ValueError("uploaded file is not a PNG")

    filename = f"{slugify(paper.title)}.png"
    images_dir = out / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    (images_dir / filename).write_bytes(data)

    paper.image = f"/images/{filename}"
    paper.has_thumbnail = True
    paper.manual_override = True

    write_outputs(papers, out, PipelineRun())
    return paper.to_full_dict()
