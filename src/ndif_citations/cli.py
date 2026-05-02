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
@click.option("--skip-github", is_flag=True, default=False, help="Skip GitHub repo discovery and output")
@click.option("--skip-papers", is_flag=True, default=False, help="Skip paper discovery, LLM processing, and paper output")
def run(output_dir: str | None, fresh: bool, skip_github: bool, skip_papers: bool) -> None:
    """Full pipeline: discover → extract → process → output."""
    if skip_github and skip_papers:
        console.print("[bold red]Error:[/bold red] --skip-github and --skip-papers cannot both be set (nothing to do)")
        raise SystemExit(1)

    from ndif_citations.discover import (
        deduplicate_papers,
        discover_github_dependents,
        discover_openalex,
        discover_s2_citations,
        enrich_repos_from_github_api,
        link_repos_to_papers,
        _tag_repo_type,
        _unlink_shared_template_papers,
    )
    from ndif_citations.extract import check_venue_upgrades, enrich_papers
    from ndif_citations.output import (
        _write_repos_outputs,
        _write_xlsx,
        load_existing_papers,
        load_existing_repos,
        merge_papers,
        merge_repos,
        print_report,
        write_outputs,
    )
    from ndif_citations.process import process_papers, process_repos
    from ndif_citations.router import route_papers, route_repos

    out = config.get_output_dir(output_dir)
    raw_dir = out / "raw"
    run_stats = PipelineRun()

    console.print("\n[bold cyan]NDIF Citation Tracker[/bold cyan] — Starting full pipeline\n")

    # ------------------------------------------------------------------ #
    # Phase 1: Discovery
    # ------------------------------------------------------------------ #
    console.print("[bold]Phase 1:[/bold] Discovery")

    unique_papers = []
    if not skip_papers:
        s2_papers = discover_s2_citations(raw_dir)
        run_stats.s2_citations_found = len(s2_papers)

        openalex_papers = discover_openalex(raw_dir)
        run_stats.openalex_found = len(openalex_papers)

        all_papers = s2_papers + openalex_papers
        unique_papers = deduplicate_papers(all_papers)
        console.print(f"  Papers — S2: {run_stats.s2_citations_found}, OpenAlex: {run_stats.openalex_found}")
        console.print(f"  After deduplication: [cyan]{len(unique_papers)}[/cyan] unique papers")
    else:
        console.print("  [dim]--skip-papers: skipping S2/OpenAlex discovery[/dim]")

    discovered_repos = []
    if not skip_github:
        discovered_repos = discover_github_dependents(raw_dir)
        run_stats.github_dependents_found = len(discovered_repos)
        console.print(f"  GitHub dependents: [green]{len(discovered_repos)}[/green] repos found")
    else:
        console.print("  [dim]--skip-github: skipping GitHub discovery[/dim]")
    console.print()

    # ------------------------------------------------------------------ #
    # Phase 2: Enrichment
    # ------------------------------------------------------------------ #
    console.print("[bold]Phase 2:[/bold] Metadata Enrichment")

    if not skip_papers and unique_papers:
        unique_papers = enrich_papers(unique_papers, raw_dir)
        console.print(f"  Enriched {len(unique_papers)} papers")

    repo_removal_counts: dict[str, int] = {"404": 0, "rename_redirect": 0, "archived": 0}
    if not skip_github and discovered_repos:
        console.print("  Enriching repos via GitHub API (stars, forks, last commit)...")
        discovered_repos, repo_removal_counts = enrich_repos_from_github_api(discovered_repos)
        console.print(f"  {len(discovered_repos)} repos retained after staleness check")

        # Drop excluded repos (e.g. ndif-team/nnsight — the library itself)
        pre_filter = len(discovered_repos)
        discovered_repos = [r for r in discovered_repos if r.merge_key() not in config.EXCLUDED_GITHUB_REPOS]
        if len(discovered_repos) < pre_filter:
            console.print(f"  Excluded {pre_filter - len(discovered_repos)} repo(s) from EXCLUDED_GITHUB_REPOS")

        # Cross-repo cleanup: unlink shared template papers (runs on merged set)
        if not fresh:
            existing_for_cross = load_existing_repos(out)
            # Merge: discovered repos override existing by merge_key
            by_key = {r.merge_key(): r for r in existing_for_cross}
            by_key.update({r.merge_key(): r for r in discovered_repos})
            all_for_cross = list(by_key.values())
        else:
            all_for_cross = discovered_repos

        unlinked_set = _unlink_shared_template_papers(all_for_cross)
        if unlinked_set:
            console.print(f"  Shared-paper cleanup: {len(unlinked_set)} template links unlinked")

        # Tag every repo (runs on the merged set for consistent cross-repo state)
        for repo in all_for_cross:
            repo.repo_type = _tag_repo_type(repo, unlinked_set)

        # Cross-link repos <-> papers (minimal URL fields)
        if not skip_papers:
            link_repos_to_papers(discovered_repos, unique_papers)
            console.print("  Cross-linked repos and papers")
    console.print()

    # ------------------------------------------------------------------ #
    # Phase 2.5: Routing
    # ------------------------------------------------------------------ #
    console.print("[bold]Phase 2.5:[/bold] Routing")

    decisions = []
    if not skip_papers:
        existing_papers = load_existing_papers(out) if not fresh else []
        decisions = route_papers(unique_papers, existing_papers)
        skipped = sum(1 for d in decisions if d.bucket.value in ("skip", "protected"))
        console.print(f"  Papers — {len(decisions) - skipped} to process, {skipped} skipped")

    repo_decisions = []
    if not skip_github and discovered_repos:
        existing_repos = load_existing_repos(out) if not fresh else []
        repo_decisions = route_repos(discovered_repos, existing_repos)
        repo_skipped = sum(1 for d in repo_decisions if d.bucket.value in ("skip", "protected"))
        console.print(f"  Repos — {len(repo_decisions) - repo_skipped} to process, {repo_skipped} skipped")
    console.print()

    # ------------------------------------------------------------------ #
    # Phase 3: Processing
    # ------------------------------------------------------------------ #
    console.print("[bold]Phase 3:[/bold] Content Processing")

    processed_papers = []
    if not skip_papers and decisions:
        console.print("  Running LLM summaries, classification, thumbnails...")
        processed_papers = process_papers(decisions, out)
        console.print(f"  Processed {len(processed_papers)} papers")

    processed_repos = []
    if not skip_github and repo_decisions:
        console.print("  Classifying repos (keyword-only)...")
        processed_repos = process_repos(repo_decisions)
        console.print(f"  Classified {len(processed_repos)} repos")
    console.print()

    # ------------------------------------------------------------------ #
    # Phase 4: Output
    # ------------------------------------------------------------------ #
    console.print("[bold]Phase 4:[/bold] Output")

    merged_papers = []
    if not skip_papers:
        if fresh:
            console.print("  [yellow]--fresh flag: rebuilding papers from scratch[/yellow]")
            existing_for_merge = []
        else:
            existing_for_merge = load_existing_papers(out)
        merged_papers, run_stats = merge_papers(existing_for_merge, [d.paper for d in decisions], run_stats)

        if existing_for_merge:
            upgrades = check_venue_upgrades(unique_papers, existing_for_merge)
            if upgrades:
                console.print(f"  [green]{len(upgrades)} venue upgrade(s) detected[/green]")

        write_outputs(merged_papers, out, run_stats)

    merged_repos = []
    if not skip_github:
        if fresh:
            console.print("  [yellow]--fresh flag: rebuilding repos from scratch[/yellow]")
            existing_repos_for_merge = []
        else:
            existing_repos_for_merge = load_existing_repos(out)
        merged_repos = merge_repos(processed_repos, existing_repos_for_merge)
        _write_repos_outputs(merged_repos, out)

    # Write combined XLSX (only if both sides ran, or just one)
    _write_xlsx(
        merged_papers if not skip_papers else [],
        merged_repos if not skip_github else [],
        out,
        skip_papers=skip_papers,
        skip_github=skip_github,
    )

    # Print final report
    print_report(
        run_stats, merged_papers, out,
        repos=merged_repos,
        skip_github=skip_github,
        skip_papers=skip_papers,
        repos_removed_counts=repo_removal_counts,
    )


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

    github_repos = discover_github_dependents(raw_dir)
    run_stats_github = len(github_repos)
    console.print(f"  GitHub: [green]{run_stats_github}[/green] repos discovered")

    all_papers = s2_papers + openalex_papers
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

    # Repos summary
    if github_repos:
        console.print(f"\n[bold]GitHub Repos:[/bold] [cyan]{len(github_repos)}[/cyan] discovered\n")
        from rich.table import Table as RichTable
        repo_table = RichTable(title="Discovered GitHub Repos", show_lines=True)
        repo_table.add_column("#", style="dim", width=4)
        repo_table.add_column("Owner/Repo", max_width=40)
        repo_table.add_column("arXiv links in README", max_width=40)
        for i, repo in enumerate(github_repos, 1):
            repo_table.add_row(
                str(i),
                f"{repo.owner}/{repo.repo}",
                ", ".join(repo.readme_arxiv_ids[:3]) or "(none)",
            )
        console.print(repo_table)
        console.print()


@cli.command()
@click.argument("url")
@click.option("--output-dir", "-o", default=None, help="Custom output directory")
def add(url: str, output_dir: str | None) -> None:
    """Process a single paper by URL and append it to the output."""
    from ndif_citations.extract import enrich_papers
    from ndif_citations.models import Category, DiscoveredPaper, DiscoverySource
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
    console.print(f"  Category: [green]{papers[0].category.value}[/green]")
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


@cli.command()
@click.argument("paper_id")
@click.option("--output-dir", "-o", default=None, help="Custom output directory")
@click.option("--out", "out_file", default=None, help="Write trace to file instead of stdout")
def debug(paper_id: str, output_dir: str | None, out_file: str | None) -> None:
    """Step-by-step trace for a single paper. Read-only against the cache.

    PAPER_ID can be an arXiv ID (e.g. 2407.14561), a DOI, or a full URL.
    """
    import sys
    from rich.panel import Panel

    from ndif_citations.output import load_existing_papers
    from ndif_citations.utils import (
        extract_ndif_context,
        extract_text_from_pdf,
        normalize_arxiv_id,
        extract_arxiv_id_from_url,
    )
    from ndif_citations.pdf_cache import get_cached_pdf
    from ndif_citations import config as cfg

    _setup_logging(verbose=True)  # debug implies --verbose

    out = cfg.get_output_dir(output_dir)

    # Determine target key from paper_id arg
    if paper_id.startswith("http"):
        candidate_arxiv = extract_arxiv_id_from_url(paper_id)
    else:
        candidate_arxiv = normalize_arxiv_id(paper_id)

    # Output sink
    if out_file:
        sink = open(out_file, "w")
    else:
        sink = sys.stdout

    def _print(text: str = "") -> None:
        print(text, file=sink)

    _print(f"\n{'='*60}")
    _print(f"  debug trace: {paper_id}")
    _print(f"{'='*60}\n")

    # --- 1. Identifiers ---
    papers = load_existing_papers(out)
    paper = None
    for p in papers:
        if candidate_arxiv and p.arxiv_id == candidate_arxiv:
            paper = p
            break
        if paper_id.startswith("10.") and p.doi == paper_id:
            paper = p
            break

    if paper is None:
        _print(f"[WARNING] Paper not found in {out}/research-papers-full.json")
        _print(f"          arXiv ID attempted: {candidate_arxiv!r}")
        if out_file:
            sink.close()
        return

    _print("## 1. Identifiers")
    _print(f"   Title:               {paper.title}")
    _print(f"   arXiv ID:            {paper.arxiv_id}")
    _print(f"   DOI:                 {paper.doi}")
    _print(f"   URL:                 {paper.url}")
    _print(f"   GitHub:              {paper.github_repo_url}")
    tier_desc = {1: "BibTeX block", 2: "Citation section", 3: "single post-2020 ID", 4: "most-recent of many"}
    tier_str = (
        f"{paper.linked_paper_tier}  ({tier_desc.get(paper.linked_paper_tier, 'unknown')})"
        if paper.linked_paper_tier is not None else "None"
    )
    _print(f"   Linked paper tier:   {tier_str}  (1=BibTeX, 2=Citation, 3=single, 4=multiple)")
    _print()

    # --- 2. PDF cache check ---
    _print("## 2. PDF cache check")
    pdf_path = get_cached_pdf(paper, out)
    if pdf_path and pdf_path.exists():
        stat = pdf_path.stat()
        magic = pdf_path.read_bytes()[:4]
        _print(f"   Cache path:     {pdf_path}")
        _print(f"   File size:      {stat.st_size:,} bytes")
        _print(f"   PDF magic:      {magic!r}  ({'OK' if magic == b'%PDF' else 'BAD - not a PDF'})")
        import datetime
        _print(f"   Last modified:  {datetime.datetime.fromtimestamp(stat.st_mtime)}")
    else:
        _print(f"   [NO CACHED PDF FOUND]")
        pdf_path = None
    _print()

    # --- 3. Text extraction ---
    _print("## 3. Text extraction")
    if pdf_path:
        full_text = extract_text_from_pdf(pdf_path)
        _print(f"   Characters extracted: {len(full_text):,}")
        _print(f"   First 300 chars:")
        _print(f"   {full_text[:300]!r}")
        _print(f"   Last 300 chars:")
        _print(f"   {full_text[-300:]!r}")
    else:
        full_text = ""
        _print("   [Skipped — no PDF]")
    _print()

    # --- 4. Keyword hits ---
    _print("## 4. Keyword hits")
    if full_text:
        text_lower = full_text.lower()
        for kw in cfg.NDIF_KEYWORDS:
            positions = []
            idx = 0
            while True:
                idx = text_lower.find(kw.lower(), idx)
                if idx == -1:
                    break
                positions.append(idx)
                idx += len(kw)
            _print(f"   {kw!r}: {len(positions)} occurrence(s)")
            for pos in positions[:3]:
                start = max(0, pos - 100)
                end = min(len(full_text), pos + len(kw) + 100)
                _print(f"     pos={pos}: ...{full_text[start:end]!r}...")
    else:
        _print("   [No text available]")
    _print()

    # --- 5. Abstract scan ---
    _print("## 5. Abstract scan")
    if paper.abstract:
        abstract_lower = paper.abstract.lower()
        matched_kws = [kw for kw in cfg.NDIF_KEYWORDS if kw.lower() in abstract_lower]
        _print(f"   Abstract length: {len(paper.abstract)} chars")
        _print(f"   Keywords matched in abstract: {matched_kws}")
    else:
        _print("   [No abstract available]")
    _print()

    # --- 6. Classification (cached state — no re-call) ---
    _print("## 6. Classification (cached state)")
    _print(f"   category:             {paper.category.value}")
    _print(f"   category_confidence:  {paper.category_confidence}")
    _print(f"   [Re-classification skipped in debug mode — use process_papers to re-run]")
    if paper.linked_paper_tier is not None and paper.linked_paper_tier <= 2:
        from ndif_citations.process import _augment_prompt_with_tier, UNIFIED_PROMPT
        aug_block = _augment_prompt_with_tier("", paper.linked_paper_tier)
        _print(f"   Tier-aware prompt augmentation: yes (tier {paper.linked_paper_tier})")
        _print(f"   Augmentation block: {aug_block.strip()!r}")
    else:
        _print(f"   Tier-aware prompt augmentation: no")
    _print()

    # --- 7. Final verdict ---
    _print("## 7. Final verdict")
    _print(f"   category:             {paper.category.value}")
    _print(f"   category_confidence:  {paper.category_confidence}")
    _print(f"   unclassified_reason:  {paper.unclassified_reason!r}")
    _print(f"   classification_signal: {paper.classification_signal!r}")
    _print(f"   has_summary:          {paper.has_summary}")
    _print(f"   has_classification:   {paper.has_classification}")
    _print(f"   has_thumbnail:        {paper.has_thumbnail}")
    _print()

    if out_file:
        sink.close()
        console.print(f"  Trace written to [cyan]{out_file}[/cyan]")


@cli.command()
@click.option("--ids", default=None, help="Comma-separated arXiv IDs, DOIs, or URLs to reclassify")
@click.option("--output-dir", "-o", default=None, help="Custom output directory")
@click.option("--dry-run", is_flag=True, help="Print changes without writing files")
def reclassify(ids: str | None, output_dir: str | None, dry_run: bool) -> None:
    """Re-run classify_category on existing papers without a full pipeline run.

    Useful to apply new pre-filter fixes to already-classified papers.
    Papers with manual_override=True are skipped.
    """
    from ndif_citations.models import Category
    from ndif_citations.output import load_existing_papers, write_outputs
    from ndif_citations.pdf_cache import get_cached_pdf
    from ndif_citations.process import classify_category
    from ndif_citations.utils import normalize_arxiv_id, extract_arxiv_id_from_url
    from ndif_citations import config as cfg
    from ndif_citations.models import PipelineRun

    _setup_logging(verbose=True)

    out = cfg.get_output_dir(output_dir)
    papers = load_existing_papers(out)

    if not papers:
        console.print(f"[bold red]No papers found in {out}/research-papers-full.json[/bold red]")
        return

    # Build lookup: all possible IDs → paper index
    def _resolve_id(paper_id: str) -> list[int]:
        """Return indices of papers matching the given ID."""
        matches: list[int] = []
        if paper_id.startswith("http"):
            candidate_arxiv = extract_arxiv_id_from_url(paper_id)
        else:
            candidate_arxiv = normalize_arxiv_id(paper_id)

        for i, p in enumerate(papers):
            if candidate_arxiv and p.arxiv_id == candidate_arxiv:
                matches.append(i)
            elif paper_id.startswith("10.") and p.doi == paper_id:
                matches.append(i)
            elif p.url == paper_id:
                matches.append(i)
        return matches

    # Determine which papers to reclassify
    if ids:
        id_list = [x.strip() for x in ids.split(",") if x.strip()]
        target_indices: list[int] = []
        for paper_id in id_list:
            found = _resolve_id(paper_id)
            if not found:
                console.print(f"  [yellow]WARNING:[/yellow] Paper not found for ID {paper_id!r}")
            else:
                target_indices.extend(found)
    else:
        target_indices = list(range(len(papers)))

    if not target_indices:
        console.print("[yellow]No papers to reclassify.[/yellow]")
        return

    console.print(
        f"\n[bold cyan]Reclassify[/bold cyan] — "
        f"{len(target_indices)} paper(s) targeted"
        f"{' [DRY RUN]' if dry_run else ''}\n"
    )

    changes: list[tuple[str, str, str, str | None]] = []  # (title, old, new, signal)

    for idx in target_indices:
        paper = papers[idx]

        if paper.manual_override:
            console.print(f"  [dim]SKIP (manual_override=True):[/dim] {paper.title[:60]}")
            continue

        old_cat = paper.category.value
        pdf_path = get_cached_pdf(paper, out)

        new_cat, new_conf = classify_category(paper, out, pdf_path=pdf_path)

        if new_cat != paper.category or new_conf != paper.category_confidence:
            signal = paper.classification_signal
            changes.append((paper.title, old_cat, new_cat.value, signal))
            if not dry_run:
                paper.category = new_cat
                paper.category_confidence = new_conf
                paper.has_classification = new_cat != Category.UNCLASSIFIED

    # Print diff
    if changes:
        console.print(f"\n[bold]Changes ({len(changes)}):[/bold]")
        for title, old, new, signal in changes:
            sig_str = f" (signal: {signal})" if signal else " (signal: llm)"
            console.print(
                f"  {title[:60]}: [yellow]{old}[/yellow] → [green]{new}[/green]{sig_str}"
            )
    else:
        console.print("\n[dim]No classification changes.[/dim]")

    if not dry_run and changes:
        run = PipelineRun()
        write_outputs(papers, out, run)
        console.print(
            f"\n[green]Wrote updated research-papers-full.json and research-papers.json[/green]"
        )
    elif dry_run:
        console.print("\n[dim]Dry run — no files written.[/dim]")


def _resolve_paper(papers, paper_id: str):
    """Return (index, paper) matching paper_id (arXiv ID, DOI, or URL). Returns (None, None) if not found."""
    from ndif_citations.utils import normalize_arxiv_id, extract_arxiv_id_from_url

    if paper_id.startswith("http"):
        candidate_arxiv = extract_arxiv_id_from_url(paper_id)
    else:
        candidate_arxiv = normalize_arxiv_id(paper_id)

    for i, p in enumerate(papers):
        if candidate_arxiv and p.arxiv_id == candidate_arxiv:
            return i, p
        if paper_id.startswith("10.") and p.doi == paper_id:
            return i, p
        if p.url == paper_id:
            return i, p
    return None, None


@cli.command()
@click.argument("paper_id")
@click.option("--output-dir", "-o", default=None, help="Custom output directory")
@click.option("--detail", default=None, help="Optional audit note (stored in reason_detail)")
@click.option("--dry-run", is_flag=True, help="Preview without writing files")
def promote(paper_id: str, output_dir: str | None, detail: str | None, dry_run: bool) -> None:
    """Move a paper to the verified bucket and freeze it with manual_override=True."""
    from ndif_citations.models import Bucket, PipelineRun
    from ndif_citations.output import load_existing_papers, write_outputs

    out = config.get_output_dir(output_dir)
    papers = load_existing_papers(out)

    idx, paper = _resolve_paper(papers, paper_id)
    if paper is None:
        console.print(f"[bold yellow]WARNING:[/bold yellow] Paper not found for ID {paper_id!r}")
        return

    console.print(f"  [{paper.bucket.value}] → [verified]: {paper.title[:70]}")
    if dry_run:
        console.print("  [dim]Dry run — no changes written.[/dim]")
        return

    paper.bucket = Bucket.VERIFIED
    paper.reason = None
    paper.reason_detail = detail
    paper.manual_override = True

    run = PipelineRun()
    write_outputs(papers, out, run)
    console.print("  [green]✓ Promoted and saved.[/green]")


@cli.command()
@click.argument("paper_id")
@click.option("--reason", "reason_str", required=True,
              help="Demotion reason (openalex_source, low_confidence, stub_metadata, "
                   "unclassified_no_keywords, unclassified_llm, manual_demote)")
@click.option("--output-dir", "-o", default=None, help="Custom output directory")
@click.option("--detail", default=None, help="Optional audit note (stored in reason_detail)")
@click.option("--dry-run", is_flag=True, help="Preview without writing files")
def demote(paper_id: str, reason_str: str, output_dir: str | None, detail: str | None, dry_run: bool) -> None:
    """Move a paper to the pending bucket and freeze it with manual_override=True."""
    from ndif_citations.models import Bucket, PaperReason, PipelineRun
    from ndif_citations.output import load_existing_papers, write_outputs

    try:
        reason = PaperReason(reason_str)
    except ValueError:
        valid = [r.value for r in PaperReason]
        console.print(f"[bold red]Error:[/bold red] Unknown reason {reason_str!r}. Valid: {valid}")
        raise SystemExit(1)

    out = config.get_output_dir(output_dir)
    papers = load_existing_papers(out)

    idx, paper = _resolve_paper(papers, paper_id)
    if paper is None:
        console.print(f"[bold yellow]WARNING:[/bold yellow] Paper not found for ID {paper_id!r}")
        return

    console.print(f"  [{paper.bucket.value}] → [pending ({reason.value})]: {paper.title[:70]}")
    if dry_run:
        console.print("  [dim]Dry run — no changes written.[/dim]")
        return

    paper.bucket = Bucket.PENDING
    paper.reason = reason
    paper.reason_detail = detail
    paper.manual_override = True

    run = PipelineRun()
    write_outputs(papers, out, run)
    console.print("  [green]✓ Demoted and saved.[/green]")


@cli.command()
@click.argument("paper_id")
@click.option("--output-dir", "-o", default=None, help="Custom output directory")
@click.option("--detail", default=None, help="Curator reason for discarding (stored in reason_detail)")
@click.option("--dry-run", is_flag=True, help="Preview without writing files")
def discard(paper_id: str, output_dir: str | None, detail: str | None, dry_run: bool) -> None:
    """Move a paper to the discarded bucket and freeze it with manual_override=True."""
    from ndif_citations.models import Bucket, PaperReason, PipelineRun
    from ndif_citations.output import load_existing_papers, write_outputs

    out = config.get_output_dir(output_dir)
    papers = load_existing_papers(out)

    idx, paper = _resolve_paper(papers, paper_id)
    if paper is None:
        console.print(f"[bold yellow]WARNING:[/bold yellow] Paper not found for ID {paper_id!r}")
        return

    console.print(f"  [{paper.bucket.value}] → [discarded]: {paper.title[:70]}")
    if detail:
        console.print(f"  Reason: {detail}")
    if dry_run:
        console.print("  [dim]Dry run — no changes written.[/dim]")
        return

    paper.bucket = Bucket.DISCARDED
    paper.reason = PaperReason.MANUAL_DISCARD
    paper.reason_detail = detail
    paper.manual_override = True

    run = PipelineRun()
    write_outputs(papers, out, run)
    console.print("  [green]✓ Discarded and saved.[/green]")


if __name__ == "__main__":
    cli()
