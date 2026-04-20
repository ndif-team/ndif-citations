"""Phase 4: Output — JSON/CSV writing with append/merge logic and CLI report."""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path

from ndif_citations.models import DiscoveredPaper, PipelineRun
from ndif_citations.utils import is_duplicate

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Load existing data
# ---------------------------------------------------------------------------

def load_existing_papers(output_dir: Path) -> list[DiscoveredPaper]:
    """Load existing papers from the full JSON file."""
    full_json = output_dir / "research-papers-full.json"
    if not full_json.exists():
        return []

    try:
        with open(full_json) as f:
            data = json.load(f)

        papers = []
        for item in data:
            try:
                paper = DiscoveredPaper.model_validate(item)
                papers.append(paper)
            except Exception as e:
                logger.warning(f"Failed to parse existing paper: {e}")

        logger.info(f"Loaded {len(papers)} existing papers from {full_json}")
        return papers

    except Exception as e:
        logger.warning(f"Failed to load existing papers: {e}")
        return []


# ---------------------------------------------------------------------------
# Merge logic
# ---------------------------------------------------------------------------

def merge_papers(existing: list[DiscoveredPaper],
                 new: list[DiscoveredPaper],
                 run: PipelineRun | None = None) -> tuple[list[DiscoveredPaper], PipelineRun]:
    """Merge new papers into existing data.

    Returns (merged_list, run_stats).
    """
    if run is None:
        run = PipelineRun()

    if not existing:
        # First run -- everything is new
        run.new_papers = len(new)
        run.total_unique = len(new)
        return new, run

    # Build lookup from existing papers
    existing_by_arxiv: dict[str, int] = {}
    existing_by_doi: dict[str, int] = {}
    existing_titles: list[tuple[int, str]] = []

    for i, p in enumerate(existing):
        if p.arxiv_id:
            existing_by_arxiv[p.arxiv_id] = i
        if p.doi:
            existing_by_doi[p.doi] = i
        existing_titles.append((i, p.title))

    merged = list(existing)  # Start with existing
    new_count = 0
    updated_count = 0

    for paper in new:
        match_idx: int | None = None

        # Check arXiv ID
        if paper.arxiv_id and paper.arxiv_id in existing_by_arxiv:
            match_idx = existing_by_arxiv[paper.arxiv_id]

        # Check DOI
        if match_idx is None and paper.doi and paper.doi in existing_by_doi:
            match_idx = existing_by_doi[paper.doi]

        # Check title similarity
        if match_idx is None:
            for idx, title in existing_titles:
                if is_duplicate(paper.title, title):
                    match_idx = idx
                    break

        if match_idx is not None:
            # Paper already exists -- update non-manual fields
            existing_paper = merged[match_idx]
            was_updated = _update_existing(existing_paper, paper)
            if was_updated:
                updated_count += 1
        else:
            # New paper!
            merged.append(paper)
            new_count += 1
            # Update lookups
            idx = len(merged) - 1
            if paper.arxiv_id:
                existing_by_arxiv[paper.arxiv_id] = idx
            if paper.doi:
                existing_by_doi[paper.doi] = idx
            existing_titles.append((idx, paper.title))

    run.existing_papers = len(existing)
    run.new_papers = new_count
    run.updated_papers = updated_count
    run.total_unique = len(merged)

    logger.info(f"Merge: {new_count} new, {updated_count} updated, {len(existing)} existing -> {len(merged)} total")
    return merged, run


def _update_existing(existing: DiscoveredPaper, new: DiscoveredPaper) -> bool:
    """Update an existing paper with new data, preserving manual overrides.

    Returns True if anything was changed.
    """
    changed = False

    # Never overwrite manually edited fields
    if existing.manual_override:
        return False

    # Update API-sourced fields if they've improved
    if new.authors and len(new.authors) > len(existing.authors):
        existing.authors = new.authors
        changed = True

    if new.affiliations and not existing.affiliations:
        existing.affiliations = new.affiliations
        changed = True

    # Venue upgrade (arXiv -> conference)
    from ndif_citations.extract import detect_venue_type
    old_type = detect_venue_type(existing.venue)
    new_type = detect_venue_type(new.venue)
    if old_type == "preprint" and new_type in ("conference", "workshop", "journal"):
        existing.venue = new.venue
        existing.peer_reviewed = True
        existing.venue_type = new_type
        changed = True

    # Fill missing fields
    if not existing.abstract and new.abstract:
        existing.abstract = new.abstract
        changed = True
    if not existing.pdf_url and new.pdf_url:
        existing.pdf_url = new.pdf_url
        changed = True
    if not existing.doi and new.doi:
        existing.doi = new.doi
        changed = True
    if not existing.arxiv_id and new.arxiv_id:
        existing.arxiv_id = new.arxiv_id
        changed = True
    if not existing.s2_paper_id and new.s2_paper_id:
        existing.s2_paper_id = new.s2_paper_id
        changed = True
    if not existing.openalex_id and new.openalex_id:
        existing.openalex_id = new.openalex_id
        changed = True

    return changed


# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------

def write_outputs(papers: list[DiscoveredPaper], output_dir: Path, run: PipelineRun) -> None:
    """Write all output files: website JSON, full JSON, CSV."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort: year descending, then title ascending
    papers.sort(key=lambda p: (-p.year, p.title.lower()))

    # Reconcile thumbnail paths and compute stats before writing JSON
    from ndif_citations.utils import slugify

    extracted_count = 0
    missing = []

    images_dir = output_dir / "images"
    for p in papers:
        expected_path = images_dir / f"{slugify(p.title)}.png"
        if expected_path.exists():
            extracted_count += 1
            p.image = f"/images/{expected_path.name}"
        else:
            missing.append(p.title)

    run.thumbnails_extracted = extracted_count
    run.thumbnails_missing = len(missing)
    run.missing_thumbnails = missing

    run.low_confidence = [
        f'"{p.title}" -- classified as "{p.detail_category.value}" (confidence: {p.category_confidence:.2f})'
        for p in papers
        if p.category_confidence < 0.7
    ]

    # 1. Website JSON (matches ResearchPaper TS interface)
    website_data = [p.to_website_dict() for p in papers]
    website_json = output_dir / "research-papers.json"
    with open(website_json, "w") as f:
        json.dump(website_data, f, indent=2, ensure_ascii=False)
    logger.info(f"Wrote {len(papers)} papers to {website_json}")

    # 2. Full JSON (all metadata)
    full_data = [p.to_full_dict() for p in papers]
    full_json = output_dir / "research-papers-full.json"
    with open(full_json, "w") as f:
        json.dump(full_data, f, indent=2, ensure_ascii=False, default=str)
    logger.info(f"Wrote {len(papers)} papers to {full_json}")

    # 3. CSV
    csv_path = output_dir / "research-papers.csv"
    _write_csv(papers, csv_path)
    logger.info(f"Wrote {len(papers)} papers to {csv_path}")


def _write_csv(papers: list[DiscoveredPaper], csv_path: Path) -> None:
    """Write papers to CSV with extended columns."""
    fieldnames = [
        "title", "authors", "affiliations", "venue", "year", "url", "pdf_url",
        "image", "description", "category", "detail_category",
        "peer_reviewed", "venue_type", "abstract", "bibtex",
        "arxiv_id", "doi", "s2_paper_id", "openalex_id",
        "date_discovered", "category_confidence", "source",
        "manual_override", "github_repo_url",
    ]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for paper in papers:
            row = {
                "title": paper.title,
                "authors": paper.authors,
                "affiliations": paper.affiliations,
                "venue": paper.venue,
                "year": paper.year,
                "url": paper.url,
                "pdf_url": paper.pdf_url or "",
                "image": paper.image or "",
                "description": paper.description,
                "category": paper.website_category.value,
                "detail_category": paper.detail_category.value,
                "peer_reviewed": paper.peer_reviewed,
                "venue_type": paper.venue_type or "",
                "abstract": paper.abstract or "",
                "bibtex": paper.bibtex or "",
                "arxiv_id": paper.arxiv_id or "",
                "doi": paper.doi or "",
                "s2_paper_id": paper.s2_paper_id or "",
                "openalex_id": paper.openalex_id or "",
                "date_discovered": paper.date_discovered.isoformat() if paper.date_discovered else "",
                "category_confidence": f"{paper.category_confidence:.2f}",
                "source": paper.source.value,
                "manual_override": paper.manual_override,
                "github_repo_url": paper.github_repo_url or "",
            }
            writer.writerow(row)


# ---------------------------------------------------------------------------
# CLI Report
# ---------------------------------------------------------------------------

def print_report(run: PipelineRun, papers: list[DiscoveredPaper], output_dir: Path) -> None:
    """Print a rich CLI summary report."""
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table

    console = Console()

    # Header
    console.print()
    console.print(Panel.fit(
        "[bold cyan]NDIF Citation Tracker -- Run Complete[/bold cyan]",
        border_style="cyan",
    ))
    console.print()

    # Sources
    console.print("[bold]Sources checked:[/bold]")
    console.print(f"  > Semantic Scholar citations: [green]{run.s2_citations_found}[/green] found")
    console.print(f"  > OpenAlex fulltext search: [green]{run.openalex_found}[/green] found")
    console.print(f"  > GitHub dependents: [green]{run.github_dependents_found}[/green] repos with papers")
    console.print()

    # Merge results
    console.print(f"[bold]After deduplication:[/bold] [cyan]{run.total_unique}[/cyan] unique papers")
    console.print()

    if run.existing_papers > 0:
        console.print("[bold]Merge results:[/bold]")
        if run.new_papers > 0:
            console.print(f"  * [bold green]{run.new_papers} NEW[/bold green] papers added")
        else:
            console.print("  - [dim]No new papers found[/dim]")
        console.print(f"  > {run.existing_papers} already in database")
        if run.updated_papers > 0:
            console.print(f"  > {run.updated_papers} papers updated")
    else:
        console.print(f"[bold]First run:[/bold] [green]{run.total_unique}[/green] papers cataloged")
    console.print()

    # Category breakdown
    using_ndif = sum(1 for p in papers if p.detail_category.value == "uses_ndif")
    unclassified = sum(1 for p in papers if p.detail_category.value == "unclassified")
    using_nnsight = sum(1 for p in papers if p.detail_category.value == "uses_nnsight")
    referencing = sum(1 for p in papers if p.detail_category.value == "referencing")

    console.print("[bold]Results:[/bold]")
    console.print(f"  > {using_ndif} papers categorized as [green]\"uses_ndif\"[/green]")
    console.print(f"  > {using_nnsight} papers categorized as [green]\"uses_nnsight\"[/green]")
    console.print(f"  > {referencing} papers categorized as [yellow]\"referencing\"[/yellow]")
    console.print(f"  > {run.thumbnails_extracted} thumbnails extracted")
    if run.thumbnails_missing > 0:
        console.print(f"  ! {run.thumbnails_missing} papers need manual thumbnails")
    console.print()

        # Unclassified papers with reasons
    unclassified_papers = [
        (p.title, "no_pdf" if not p.pdf_url and not p.arxiv_id else "no_keywords")
        for p in papers
        if p.detail_category.value == "unclassified"
    ]
    if unclassified_papers:
        console.print(f"[bold yellow]UNCLASSIFIED papers ({len(unclassified_papers)}):[/bold yellow]")
        for title, reason in unclassified_papers[:15]:
            console.print(f' - "{title}" ({reason})')
        if len(unclassified_papers) > 15:
            console.print(f" ... and {len(unclassified_papers) - 15} more")
        console.print()

# Low confidence
    if run.low_confidence:
        console.print("[bold yellow]Category classifications with low confidence:[/bold yellow]")
        for i, msg in enumerate(run.low_confidence[:10], 1):
            console.print(f"  {i}. {msg}")
        console.print()

    # Missing thumbnails
    if run.missing_thumbnails:
        console.print("[bold yellow]Papers needing manual thumbnails:[/bold yellow]")
        for i, title in enumerate(run.missing_thumbnails[:15], 1):
            console.print(f"  {i}. \"{title}\"")
        if len(run.missing_thumbnails) > 15:
            console.print(f"  ... and {len(run.missing_thumbnails) - 15} more")
        console.print()

    # Output files
    console.print("[bold]Output files:[/bold]")
    console.print(f"  -> {output_dir / 'research-papers.json'}  ({run.total_unique} papers)")
    console.print(f"  -> {output_dir / 'research-papers-full.json'}  ({run.total_unique} papers, all metadata)")
    console.print(f"  -> {output_dir / 'research-papers.csv'}  ({run.total_unique} papers, extended columns)")
    console.print(f"  -> {output_dir / 'images/'}  ({run.thumbnails_extracted} thumbnails)")
    console.print()

    # Usage hint
    console.print("[dim]To use with the NDIF website:[/dim]")
    console.print("[dim]  1. Copy output/research-papers.json -> data/research-papers.json in the website repo[/dim]")
    console.print("[dim]  2. Copy output/images/*.png -> public/images/ in the website repo[/dim]")
    console.print()

    # Errors
    if run.errors:
        console.print(f"[bold red]Errors ({len(run.errors)}):[/bold red]")
        for err in run.errors:
            console.print(f"  x {err}")
    else:
        console.print("[green]Errors (0): (none)[/green]")

    console.print()
