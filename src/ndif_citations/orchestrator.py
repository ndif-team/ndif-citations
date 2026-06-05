"""Event-emitting pipeline orchestrator (Task 1.6).

This module extracts the orchestration that currently lives inside
``cli.run()`` (cli.py ~lines 42-266) into five discrete, event-emitting stages:

    discover_stage -> enrich_stage -> route_stage -> process_stage -> finalize_stage

``run_pipeline`` chains them in order and returns a ``FinalizeResult``.

This is an EXTRACT-NOT-REWRITE: each stage wraps the SAME pipeline function calls
in the SAME order and with the SAME branching as ``cli.run``. The only behavioral
substitution is that every ``console.print(...)`` becomes an ``events.emit(...)``.
No pipeline logic, thresholds, or ordering is changed. ``cli.run`` is intentionally
left untouched in this task (Task 1.8 repoints it at this orchestrator).

Pipeline functions are imported at MODULE TOP (not lazily inside functions) so that
tests can monkeypatch ``orchestrator.<name>`` — e.g.
``install_pipeline_fakes(monkeypatch, orchestrator)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from ndif_citations import config, events
from ndif_citations.discover import (
    deduplicate_papers,
    discover_github_dependents,
    discover_openalex,
    discover_s2_citations,
    discover_scholar,
    enrich_repos_from_github_api,
    filter_by_min_year,
    link_repos_to_papers,
    _tag_repo_type,
    _unlink_shared_template_papers,
)
from ndif_citations.extract import check_venue_upgrades, enrich_papers
from ndif_citations.models import DiscoveredPaper, DiscoveredRepo, PipelineRun
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
from ndif_citations.router import (
    RepoRoutingDecision,
    RoutingDecision,
    route_papers,
    route_repos,
)


# ---------------------------------------------------------------------------
# Stage result contracts (C2)
# ---------------------------------------------------------------------------

@dataclass
class DiscoverResult:
    papers: list[DiscoveredPaper]
    repos: list[DiscoveredRepo]
    run_stats: PipelineRun


@dataclass
class EnrichResult:
    papers: list[DiscoveredPaper]
    repos: list[DiscoveredRepo]
    removal_counts: dict[str, int]
    existing_repos: list[DiscoveredRepo]


@dataclass
class RouteResult:
    paper_decisions: list[RoutingDecision]
    repo_decisions: list[RepoRoutingDecision]
    existing_papers: list[DiscoveredPaper]
    enrich: EnrichResult
    # Populated by process_stage: the exact list process_repos() returns.
    # For SKIP/PROTECTED repos this substitutes decision.existing_repo for
    # decision.repo, which merge_repos depends on — so finalize MUST merge from
    # this list (matching cli.run's `merge_repos(processed_repos, ...)`), not
    # from `[d.repo for d in repo_decisions]`.
    processed_repos: list[DiscoveredRepo] = field(default_factory=list)


@dataclass
class FinalizeResult:
    merged_papers: list[DiscoveredPaper]
    merged_repos: list[DiscoveredRepo]
    run_stats: PipelineRun


# ---------------------------------------------------------------------------
# Stage 1: Discovery  (cli.run Phase 1, lines ~80-119)
# ---------------------------------------------------------------------------

def discover_stage(
    out: Path,
    *,
    skip_papers: bool,
    skip_github: bool,
    fresh: bool,
) -> DiscoverResult:
    """Discover papers (S2 / OpenAlex / Scholar) and GitHub repos.

    Mirrors cli.run Phase 1 exactly:
      * papers only if ``not skip_papers``
      * ``discover_scholar(..., force_refresh=fresh)``
      * dedup + ``filter_by_min_year(config.MIN_PAPER_YEAR)``
      * repos only if ``not skip_github``
    """
    raw_dir = out / "raw"
    run_stats = PipelineRun()

    events.emit("stage_start", stage="discover")

    unique_papers: list[DiscoveredPaper] = []
    if not skip_papers:
        s2_papers = discover_s2_citations(raw_dir)
        run_stats.s2_citations_found = len(s2_papers)

        openalex_papers = discover_openalex(raw_dir)
        run_stats.openalex_found = len(openalex_papers)

        scholar_papers = discover_scholar(raw_dir, force_refresh=fresh)
        run_stats.scholar_found = len(scholar_papers)

        all_papers = s2_papers + openalex_papers + scholar_papers
        unique_papers = deduplicate_papers(all_papers)
        before_year = len(unique_papers)
        unique_papers = filter_by_min_year(unique_papers, config.MIN_PAPER_YEAR)
        dropped_old = before_year - len(unique_papers)

        events.emit(
            "source_count",
            stage="discover",
            s2=run_stats.s2_citations_found,
            openalex=run_stats.openalex_found,
            scholar=run_stats.scholar_found,
        )
        events.emit(
            "dedup",
            stage="discover",
            before_year=before_year,
            after_year=len(unique_papers),
            dropped_old=dropped_old,
            min_year=config.MIN_PAPER_YEAR,
        )
    else:
        events.emit("log", stage="discover", message="--skip-papers: skipping S2/OpenAlex/Scholar discovery")

    discovered_repos: list[DiscoveredRepo] = []
    if not skip_github:
        discovered_repos = discover_github_dependents(raw_dir)
        run_stats.github_dependents_found = len(discovered_repos)
        events.emit("source_count", stage="discover", github=len(discovered_repos))
    else:
        events.emit("log", stage="discover", message="--skip-github: skipping GitHub discovery")

    events.emit("stage_done", stage="discover", papers=len(unique_papers), repos=len(discovered_repos))

    return DiscoverResult(papers=unique_papers, repos=discovered_repos, run_stats=run_stats)


# ---------------------------------------------------------------------------
# Stage 2: Enrichment  (cli.run Phase 2, lines ~121-179)
# ---------------------------------------------------------------------------

def enrich_stage(
    out: Path,
    d: DiscoverResult,
    *,
    skip_papers: bool,
    skip_github: bool,
    fresh: bool,
) -> EnrichResult:
    """Enrich papers (metadata) and repos (GitHub API + cross-repo cleanup).

    Mirrors cli.run Phase 2 exactly, including the ``fresh`` branch that builds
    ``all_for_cross`` from the merged (existing + discovered) repo set, the
    shared-template unlink, the ``_tag_repo_type`` loop with the guarded course
    ``linked_paper_url`` clear, and the final ``link_repos_to_papers``.
    """
    raw_dir = out / "raw"
    unique_papers = d.papers
    discovered_repos = d.repos

    events.emit("stage_start", stage="enrich")

    if not skip_papers and unique_papers:
        unique_papers = enrich_papers(unique_papers, raw_dir)
        events.emit("log", stage="enrich", message=f"Enriched {len(unique_papers)} papers")

    repo_removal_counts: dict[str, int] = {"404": 0, "rename_redirect": 0, "archived": 0}
    existing_repos_loaded: list[DiscoveredRepo] = []
    if not skip_github and discovered_repos:
        events.emit("log", stage="enrich", message="Enriching repos via GitHub API (stars, forks, last commit)...")
        discovered_repos, repo_removal_counts = enrich_repos_from_github_api(discovered_repos)
        events.emit("log", stage="enrich", message=f"{len(discovered_repos)} repos retained after staleness check")

        # Drop excluded repos (e.g. ndif-team/nnsight — the library itself)
        pre_filter = len(discovered_repos)
        discovered_repos = [r for r in discovered_repos if r.merge_key() not in config.EXCLUDED_GITHUB_REPOS]
        if len(discovered_repos) < pre_filter:
            events.emit(
                "log", stage="enrich",
                message=f"Excluded {pre_filter - len(discovered_repos)} repo(s) from EXCLUDED_GITHUB_REPOS",
            )

        # Cross-repo cleanup: unlink shared template papers (runs on merged set)
        if not fresh:
            existing_repos_loaded = load_existing_repos(out)
            # Merge: discovered repos override existing by merge_key
            by_key = {r.merge_key(): r for r in existing_repos_loaded}
            by_key.update({r.merge_key(): r for r in discovered_repos})
            all_for_cross = list(by_key.values())
        else:
            all_for_cross = discovered_repos

        unlinked_set = _unlink_shared_template_papers(all_for_cross)
        if unlinked_set:
            events.emit(
                "log", stage="enrich",
                message=f"Shared-paper cleanup: {len(unlinked_set)} template links unlinked",
            )

        # Tag every repo (runs on the merged set for consistent cross-repo state)
        course_cleared = 0
        for repo in all_for_cross:
            repo.repo_type = _tag_repo_type(repo, unlinked_set)
            # Course repos cite many papers — none is canonical. Clear the link
            # so they neither display a 📄 badge nor cross-link to any paper.
            # Skip this side effect for curator-overridden repos: if a human
            # set both repo_type AND linked_paper_url, trust them.
            if (
                repo.repo_type == "course"
                and repo.linked_paper_url
                and not repo.manual_override
            ):
                repo.linked_paper_url = None
                repo.linked_paper_tier = None
                course_cleared += 1
        if course_cleared:
            events.emit(
                "log", stage="enrich",
                message=f"Cleared linked_paper_url on {course_cleared} course repo(s)",
            )

        # Cross-link repos <-> papers (minimal URL fields)
        if not skip_papers:
            link_repos_to_papers(discovered_repos, unique_papers)
            events.emit("log", stage="enrich", message="Cross-linked repos and papers")

    events.emit("stage_done", stage="enrich", papers=len(unique_papers), repos=len(discovered_repos))

    return EnrichResult(
        papers=unique_papers,
        repos=discovered_repos,
        removal_counts=repo_removal_counts,
        existing_repos=existing_repos_loaded,
    )


# ---------------------------------------------------------------------------
# Stage 3: Routing  (cli.run Phase 2.5, lines ~181-199)
# ---------------------------------------------------------------------------

def route_stage(
    out: Path,
    e: EnrichResult,
    *,
    skip_papers: bool,
    skip_github: bool,
    fresh: bool,
) -> RouteResult:
    """Route papers and repos against the existing DB.

    Mirrors cli.run Phase 2.5: load existing (papers/repos) only if ``not fresh``
    (else ``[]``), then ``route_papers`` / ``route_repos``.
    """
    unique_papers = e.papers
    discovered_repos = e.repos

    events.emit("stage_start", stage="route")

    decisions: list[RoutingDecision] = []
    existing_papers: list[DiscoveredPaper] = []
    if not skip_papers:
        existing_papers = load_existing_papers(out) if not fresh else []
        decisions = route_papers(unique_papers, existing_papers)
        skipped = sum(1 for dd in decisions if dd.bucket.value in ("skip", "protected"))
        events.emit(
            "route_summary", stage="route",
            kind="papers",
            to_process=len(decisions) - skipped,
            skipped=skipped,
        )

    repo_decisions: list[RepoRoutingDecision] = []
    if not skip_github and discovered_repos:
        existing_repos = load_existing_repos(out) if not fresh else []
        repo_decisions = route_repos(discovered_repos, existing_repos)
        repo_skipped = sum(1 for dd in repo_decisions if dd.bucket.value in ("skip", "protected"))
        events.emit(
            "route_summary", stage="route",
            kind="repos",
            to_process=len(repo_decisions) - repo_skipped,
            skipped=repo_skipped,
        )

    events.emit("stage_done", stage="route")

    return RouteResult(
        paper_decisions=decisions,
        repo_decisions=repo_decisions,
        existing_papers=existing_papers,
        enrich=e,
    )


# ---------------------------------------------------------------------------
# Stage 4: Processing  (cli.run Phase 3, lines ~201-217)
# ---------------------------------------------------------------------------

def process_stage(
    out: Path,
    r: RouteResult,
    *,
    skip_papers: bool,
    skip_github: bool,
    cancel_check: Optional[Callable[[], bool]] = None,
) -> tuple[int, int]:
    """Process papers (LLM summaries/classification/thumbnails) and repos.

    Mirrors cli.run Phase 3. ``process_papers`` mutates each ``decision.paper`` in
    place; we return the COUNT of paper- and repo-decisions that were processed
    so ``finalize_stage`` can merge from the decisions (matching cli.run, which
    merges ``[d.paper for d in decisions]`` — NOT the ``process_papers`` return).

    ``cancel_check`` is accepted but unused in this task; real cooperative-cancel
    wiring lands in Task 1.7.
    """
    events.emit("stage_start", stage="process")

    n_papers = 0
    if not skip_papers and r.paper_decisions:
        events.emit("log", stage="process", message="Running LLM summaries, classification, thumbnails...")
        processed_papers = process_papers(r.paper_decisions, out)
        n_papers = len(r.paper_decisions)
        events.emit("log", stage="process", message=f"Processed {len(processed_papers)} papers")

    n_repos = 0
    if not skip_github and r.repo_decisions:
        events.emit("log", stage="process", message="Classifying repos (keyword-only)...")
        processed_repos = process_repos(r.repo_decisions)
        n_repos = len(r.repo_decisions)
        # Stash the exact process_repos() output so finalize_stage merges from it
        # (preserves the SKIP/PROTECTED existing_repo substitution from cli.run).
        r.processed_repos = processed_repos
        events.emit("log", stage="process", message=f"Classified {len(processed_repos)} repos")

    events.emit("stage_done", stage="process", papers=n_papers, repos=n_repos)

    return n_papers, n_repos


# ---------------------------------------------------------------------------
# Stage 5: Finalize / Output  (cli.run Phase 4, lines ~219-266)
# ---------------------------------------------------------------------------

def finalize_stage(
    out: Path,
    r: RouteResult,
    run_stats: PipelineRun,
    *,
    skip_papers: bool,
    skip_github: bool,
    fresh: bool,
    completed: Optional[tuple[int, int]] = None,
) -> FinalizeResult:
    """Merge, write outputs, and emit the final report.

    Mirrors cli.run Phase 4 exactly. Papers are merged from the DECISIONS
    (``[d.paper for d in paper_decisions]``), matching cli.run line ~231. Repos
    are merged from the PROCESSED repo decisions (``[d.repo for d in
    repo_decisions]`` after process_repos mutated them), matching the
    ``processed_repos`` that cli.run passes to ``merge_repos``. When
    ``process_stage`` ran, it stashed that list on ``r.processed_repos``; we merge
    from there. If it is empty (no repos processed), we fall back to the
    decision repos sliced to ``n_repos``.

    ``completed`` is an optional ``(n_papers, n_repos)`` slice from
    ``process_stage``. When None, all decisions are used (full run) — i.e. the
    slice is the full decision list.
    """
    enrich = r.enrich

    events.emit("stage_start", stage="finalize")

    if completed is None:
        n_papers = len(r.paper_decisions)
        n_repos = len(r.repo_decisions)
    else:
        n_papers, n_repos = completed

    merged_papers: list[DiscoveredPaper] = []
    if not skip_papers:
        if fresh:
            events.emit("log", stage="finalize", message="--fresh flag: rebuilding papers from scratch")
            existing_for_merge: list[DiscoveredPaper] = []
        else:
            existing_for_merge = load_existing_papers(out)
        processed_paper_objs = [d.paper for d in r.paper_decisions[:n_papers]]
        merged_papers, run_stats = merge_papers(existing_for_merge, processed_paper_objs, run_stats)

        events.emit(
            "merge_result", stage="finalize",
            kind="papers",
            new=run_stats.new_papers,
            updated=run_stats.updated_papers,
            existing=run_stats.existing_papers,
            total=run_stats.total_unique,
        )

        if existing_for_merge:
            upgrades = check_venue_upgrades(enrich.papers, existing_for_merge)
            if upgrades:
                events.emit(
                    "log", stage="finalize",
                    message=f"{len(upgrades)} venue upgrade(s) detected",
                )

        write_outputs(merged_papers, out, run_stats)

    merged_repos: list[DiscoveredRepo] = []
    if not skip_github:
        if fresh:
            events.emit("log", stage="finalize", message="--fresh flag: rebuilding repos from scratch")
            existing_repos_for_merge: list[DiscoveredRepo] = []
        else:
            existing_repos_for_merge = load_existing_repos(out)
        # Prefer the process_repos() output stashed by process_stage (it carries
        # the SKIP/PROTECTED existing_repo substitution). Fall back to the
        # decision repos only when process_stage produced nothing.
        if r.processed_repos:
            processed_repo_objs = r.processed_repos[:n_repos]
        else:
            processed_repo_objs = [d.repo for d in r.repo_decisions[:n_repos]]
        merged_repos = merge_repos(processed_repo_objs, existing_repos_for_merge)
        _write_repos_outputs(merged_repos, out)
        events.emit(
            "merge_result", stage="finalize",
            kind="repos",
            total=len(merged_repos),
        )

    # Write combined XLSX (only if both sides ran, or just one)
    _write_xlsx(
        merged_papers if not skip_papers else [],
        merged_repos if not skip_github else [],
        out,
        skip_papers=skip_papers,
        skip_github=skip_github,
    )

    # Print final report — print_report itself stays callable by the CLI (Task 1.8).
    print_report(
        run_stats, merged_papers, out,
        repos=merged_repos,
        skip_github=skip_github,
        skip_papers=skip_papers,
        repos_removed_counts=enrich.removal_counts,
    )
    events.emit(
        "report", stage="finalize",
        total_unique=run_stats.total_unique,
        new_papers=run_stats.new_papers,
        updated_papers=run_stats.updated_papers,
        existing_papers=run_stats.existing_papers,
        verified=sum(1 for p in merged_papers if p.bucket.value == "verified"),
        pending=sum(1 for p in merged_papers if p.bucket.value == "pending"),
        discarded=sum(1 for p in merged_papers if p.bucket.value == "discarded"),
        repos=len(merged_repos),
        repos_removed_counts=enrich.removal_counts,
    )

    events.emit("stage_done", stage="finalize")

    return FinalizeResult(
        merged_papers=merged_papers,
        merged_repos=merged_repos,
        run_stats=run_stats,
    )


# ---------------------------------------------------------------------------
# Top-level driver
# ---------------------------------------------------------------------------

def run_pipeline(
    out: Path,
    *,
    mode: str,
    skip_papers: bool = False,
    skip_github: bool = False,
) -> FinalizeResult:
    """Run the full pipeline by chaining the five stages in order.

    ``mode`` is one of {"fresh", "incremental"}. "fresh" maps to the old
    ``--fresh`` behavior (the boolean threaded into every stage).

    No gate pause is inserted here — that is Phase 3 work. Returns the
    ``FinalizeResult`` from ``finalize_stage``.
    """
    fresh = mode == "fresh"

    d = discover_stage(out, skip_papers=skip_papers, skip_github=skip_github, fresh=fresh)
    e = enrich_stage(out, d, skip_papers=skip_papers, skip_github=skip_github, fresh=fresh)
    r = route_stage(out, e, skip_papers=skip_papers, skip_github=skip_github, fresh=fresh)
    completed = process_stage(out, r, skip_papers=skip_papers, skip_github=skip_github)
    return finalize_stage(
        out, r, d.run_stats,
        skip_papers=skip_papers,
        skip_github=skip_github,
        fresh=fresh,
        completed=completed,
    )
