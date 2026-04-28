"""CLI entry point for the NDIF Citation Tracking Pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

import click
from rich.console import Console
from rich.logging import RichHandler

from ndif_citations import config
from ndif_citations.models import PipelineRun
from ndif_citations.router import route_papers, get_bucket_summary

console = Console()


def _setup_logging(verbose: bool = False) -> None:
    """Configure logging with rich handler."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(console=console, rich_tracebacks=True)],
    )


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging")
def cli(verbose: bool) -> None:
    """NDIF Citation Tracker — discover and catalog papers citing NDIF/NNsight."""
    _setup_logging(verbose)


@cli.command()
@click.option("--output-dir", "-o", default=None, help="Custom output directory")
@click.option("--fresh", is_flag=True, help="Ignore existing data, start from scratch")
def run(output_dir: str | None, fresh: bool) -> None:
    """Full pipeline: discover → extract → process → output."""
    from ndif_citations.discover import (
        deduplicate_papers,
        discover_github_dependents,
        discover_openalex,
        discover_s2_citations,
    )
    from ndif_citations.extract import check_venue_upgrades, enrich_papers
    from ndif_citations.output import (
        load_existing_papers,
        merge_papers,
        print_report,
        write_outputs,
    )
    from ndif_citations.process import process_papers

    out = config.get_output_dir(output_dir)
    raw_dir = out / "raw"
    run_stats = PipelineRun()

    console.print("\n[bold cyan]NDIF Citation Tracker[/bold cyan] — Starting full pipeline\n")

    # Phase 1: Discovery
    console.print("[bold]Phase 1:[/bold] Discovery")

    s2_papers = discover_s2_citations(raw_dir)
    run_stats.s2_citations_found = len(s2_papers)

    openalex_papers = discover_openalex(raw_dir)
    run_stats.openalex_found = len(openalex_papers)

    github_papers = discover_github_dependents(raw_dir)
    run_stats.github_dependents_found = len(github_papers)

    all_papers = s2_papers + openalex_papers + github_papers
    console.print(f"  Found {len(all_papers)} total papers across all sources")

    # Deduplicate
    unique_papers = deduplicate_papers(all_papers)
    console.print(f"  After deduplication: [cyan]{len(unique_papers)}[/cyan] unique papers\n")

    # Phase 2: Metadata enrichment
    console.print("[bold]Phase 2:[/bold] Metadata Enrichment")
    unique_papers = enrich_papers(unique_papers, raw_dir)
    console.print(f"  Enriched {len(unique_papers)} papers\n")

    # Phase 2.5: Router - decide which papers need processing
    console.print("[bold]Phase 2.5:[/bold] Routing")
    from ndif_citations.router import route_papers
    from ndif_citations.output import load_existing_papers

    existing_papers = load_existing_papers(out) if not fresh else []
    decisions = route_papers(unique_papers, existing_papers)

    skipped = sum(1 for d in decisions if d.bucket.value in ("skip", "protected"))
    console.print(f" [dim]{skipped} papers skipped (already complete)[/dim]")
    console.print(f" [green]{len(decisions) - skipped}[/green] papers need processing\n")


    # Phase 3: Content processing
    console.print("[bold]Phase 3:[/bold] Content Processing (LLM summaries, classification, thumbnails)")
    processed_papers = process_papers(decisions, out)
    console.print(f" Processed {len(processed_papers)} papers\n")

    # Phase 4: Output with merge
    console.print("[bold]Phase 4:[/bold] Output")

    if fresh:
        console.print("  [yellow]--fresh flag: ignoring existing data[/yellow]")
        existing_papers = []
    else:
        existing_papers = load_existing_papers(out)

    merged_papers, run_stats = merge_papers(existing_papers, [d.paper for d in decisions], run_stats)

    # Check for venue upgrades (arXiv -> conference)
    if existing_papers:
        upgrades = check_venue_upgrades(unique_papers, existing_papers)
        if upgrades:
            console.print(f"  [green]{len(upgrades)} venue upgrade(s) detected:[/green]")
            for msg in upgrades:
                console.print(f"    {msg}")
            console.print()

    write_outputs(merged_papers, out, run_stats)

    # Print final report
    print_report(run_stats, merged_papers, out)


@cli.command()
@click.option("--output-dir", "-o", default=None, help="Custom output directory")
def discover(output_dir: str | None) -> None:
    """Discovery only — show what papers exist without LLM processing."""
    from ndif_citations.discover import (
        deduplicate_papers,
        discover_github_dependents,
        discover_openalex,
        discover_s2_citations,
    )
    from ndif_citations.extract import enrich_papers

    out = config.get_output_dir(output_dir)
    raw_dir = out / "raw"

    console.print("\n[bold cyan]NDIF Citation Tracker[/bold cyan] — Discovery Mode\n")

    # Phase 1: Discovery
    s2_papers = discover_s2_citations(raw_dir)
    console.print(f"  Semantic Scholar: [green]{len(s2_papers)}[/green] citations")

    openalex_papers = discover_openalex(raw_dir)
    console.print(f"  OpenAlex: [green]{len(openalex_papers)}[/green] papers")

    github_papers = discover_github_dependents(raw_dir)
    console.print(f"  GitHub: [green]{len(github_papers)}[/green] papers from dependents")

    all_papers = s2_papers + openalex_papers + github_papers
    unique_papers = deduplicate_papers(all_papers)

    # Basic enrichment (no LLM)
    unique_papers = enrich_papers(unique_papers, raw_dir)

    console.print(f"\n[bold]After deduplication:[/bold] [cyan]{len(unique_papers)}[/cyan] unique papers\n")

    # Print list
    from rich.table import Table
    table = Table(title="Discovered Papers", show_lines=True)
    table.add_column("#", style="dim", width=4)
    table.add_column("Title", max_width=50)
    table.add_column("Year", width=6)
    table.add_column("Venue", max_width=30)
    table.add_column("Source", width=12)

    for i, paper in enumerate(sorted(unique_papers, key=lambda p: (-p.year, p.title)), 1):
        table.add_row(
            str(i),
            paper.title[:50] + ("..." if len(paper.title) > 50 else ""),
            str(paper.year) if paper.year else "?",
            paper.venue[:30] if paper.venue else "?",
            paper.source.value,
        )

    console.print(table)
    console.print()


@cli.command()
@click.argument("url")
@click.option("--output-dir", "-o", default=None, help="Custom output directory")
def add(url: str, output_dir: str | None) -> None:
    """Process a single paper by URL and append it to the output."""
    from ndif_citations.extract import enrich_papers
    from ndif_citations.models import DiscoveredPaper, DiscoverySource
    from ndif_citations.output import load_existing_papers, merge_papers, write_outputs
    from ndif_citations.process import process_papers
    from ndif_citations.utils import extract_arxiv_id_from_url

    out = config.get_output_dir(output_dir)

    console.print(f"\n[bold cyan]NDIF Citation Tracker[/bold cyan] — Adding paper: {url}\n")

    # Create a paper from the URL
    arxiv_id = extract_arxiv_id_from_url(url)

    paper = DiscoveredPaper(
        title="[Pending metadata lookup]",
        url=url,
        arxiv_id=arxiv_id,
        source=DiscoverySource.MANUAL_ADD,
    )

    # Try to look up metadata via S2
    if arxiv_id:
        try:
            from semanticscholar import SemanticScholar
            sch = SemanticScholar(api_key=config.S2_API_KEY) if config.S2_API_KEY else SemanticScholar()
            s2_paper = sch.get_paper(f"ARXIV:{arxiv_id}", fields=config.S2_FIELDS)
            if s2_paper:
                paper.title = getattr(s2_paper, "title", paper.title)
                authors_list = getattr(s2_paper, "authors", []) or []
                paper.authors = ", ".join(
                    a.get("name", "") if isinstance(a, dict) else getattr(a, "name", str(a))
                    for a in authors_list
                )
                paper.abstract = getattr(s2_paper, "abstract", None)
                paper.venue = getattr(s2_paper, "venue", "") or ""
                pub_date_str = getattr(s2_paper, "publicationDate", None)
                if pub_date_str:
                    from datetime import date
                    try:
                        if isinstance(pub_date_str, str):
                            paper.publication_date = date.fromisoformat(pub_date_str)
                        else:
                            paper.publication_date = pub_date_str
                        paper.year = paper.publication_date.year
                    except (ValueError, AttributeError):
                        pass
                external_ids = getattr(s2_paper, "externalIds", {}) or {}
                paper.doi = external_ids.get("DOI")
                paper.s2_paper_id = getattr(s2_paper, "paperId", None)
                open_access = getattr(s2_paper, "openAccessPdf", None)
                if open_access:
                    paper.pdf_url = (open_access.get("url")
                                     if isinstance(open_access, dict)
                                     else getattr(open_access, "url", None))
        except Exception as e:
            console.print(f"  [yellow]S2 lookup failed: {e}[/yellow]")

    papers = [paper]

    # Enrich
    papers = enrich_papers(papers)
    console.print(f"  Title: [cyan]{papers[0].title}[/cyan]")
    console.print(f"  Authors: {papers[0].authors}")
    console.print(f"  Venue: {papers[0].venue}")

    # Process
    papers = process_papers(papers, out)
    console.print(f"  Category: [green]{papers[0].detail_category.value}[/green]")
    console.print(f"  Description: {papers[0].description[:100]}...")

    # Merge and save
    existing = load_existing_papers(out)
    merged, run_stats = merge_papers(existing, papers)
    write_outputs(merged, out, run_stats)

    if run_stats.new_papers > 0:
        console.print(f"\n  ★ [bold green]Paper added successfully![/bold green]")
    else:
        console.print(f"\n  ─ [dim]Paper already exists in database[/dim]")

    console.print()


if __name__ == "__main__":
    cli()
