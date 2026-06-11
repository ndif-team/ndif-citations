"""Service helpers for the repos API.

Read functions are pure (no side-effects). Mutation functions (edit_repo,
exclude_repo) load, mutate, and persist via _write_repos_outputs.

All functions operate on a given output_dir Path so they are trivially
testable via dependency_overrides.
"""
from __future__ import annotations

from pathlib import Path

from ndif_citations import config, settings_store
from ndif_citations.models import DiscoveredRepo
from ndif_citations.output import _write_repos_outputs, load_existing_repos

_VALID_REPO_TYPES = {"research", "course", "experiment"}
_EDITABLE_FIELDS = {"repo_type", "linked_paper_url", "description"}


def _repo_to_row(repo: DiscoveredRepo) -> dict:
    """Convert a DiscoveredRepo to the list-row dict shape."""
    return {
        "id": repo.merge_key(),
        "owner": repo.owner,
        "repo": repo.repo,
        "url": repo.url,
        "description": repo.description,
        "stars": repo.stars,
        "forks": repo.forks,
        "language": repo.language,
        "repo_type": repo.repo_type,
        "category": repo.category.value,
        "linked_paper_url": repo.linked_paper_url,
        "last_commit": repo.last_commit.isoformat() if repo.last_commit else None,
        "manual_override": repo.manual_override,
    }


def _desc_str(value: "str | None") -> tuple:
    """Sort key fragment: ISO date string descending (None handled by caller)."""
    return tuple(-ord(c) for c in (value or ""))


def list_rows(
    out: Path,
    *,
    repo_type: str | None = None,
    q: str | None = None,
    sort: str = "stars_desc",
) -> list[dict]:
    """Return a filtered, sorted list of repo row dicts.

    Parameters
    ----------
    out:
        Output directory (passed to ``load_existing_repos``).
    repo_type:
        Optional filter: ``"research"``, ``"course"``, or ``"experiment"``.
        If ``None`` all types are included.
    q:
        Optional case-insensitive substring search over ``owner/repo`` and description.
    sort:
        ``"stars_desc"`` (default), ``"recent"`` (by last_commit desc, None last),
        ``"added"`` (by first_seen desc, None last),
        or ``"name"`` (owner/repo alphabetical).
    """
    repos = load_existing_repos(out)

    # Filter by repo_type
    if repo_type is not None:
        repos = [r for r in repos if r.repo_type == repo_type]

    # Substring search over owner/repo and description
    if q is not None:
        q_lower = q.lower()
        repos = [
            r for r in repos
            if q_lower in r.merge_key().lower()
            or q_lower in (r.description or "").lower()
        ]

    # Sort
    if sort == "recent":
        # last_commit desc; repos with no last_commit sort last
        repos.sort(
            key=lambda r: (
                r.last_commit is None,            # False (0) before True (1) → dated first
                -(r.last_commit.toordinal() if r.last_commit else 0),  # newer = higher ordinal = more negative
            )
        )
    elif sort == "added":
        # first_seen desc; repos without a stamp sort last; stars desc tiebreak
        # (the whole pre-2026-06-10 catalog shares one backfilled first_seen)
        repos.sort(
            key=lambda r: (
                r.first_seen is None,
                _desc_str(r.first_seen),
                -(r.stars or 0),
            )
        )
    elif sort == "name":
        repos.sort(key=lambda r: r.merge_key().lower())
    else:  # stars_desc (default)
        repos.sort(
            key=lambda r: (
                r.stars is None,       # None last
                -(r.stars or 0),
                r.owner.lower(),
                r.repo.lower(),
            )
        )

    return [_repo_to_row(r) for r in repos]


def get_repo(out: Path, repo_id: str) -> dict | None:
    """Return ``to_full_dict()`` for the repo with the given merge_key, or None."""
    repos = load_existing_repos(out)
    for repo in repos:
        if repo.merge_key() == repo_id:
            return repo.to_full_dict()
    return None


# ---------------------------------------------------------------------------
# Mutation helpers
# ---------------------------------------------------------------------------

def edit_repo(out: Path, repo_id: str, fields: dict) -> dict:
    """Edit one or more fields on a repo and persist.

    Parameters
    ----------
    out:
        Output directory.
    repo_id:
        Repo merge_key (``owner/repo``).
    fields:
        Mapping of editable field name → value. Only ``repo_type``,
        ``linked_paper_url``, and ``description`` are allowed.

    Returns the updated repo's ``to_full_dict()``.

    Raises
    ------
    KeyError:
        If no repo with *repo_id* exists.
    ValueError:
        If a field name is not in the allowed set, or if ``repo_type`` is not
        one of ``research``, ``course``, ``experiment``.
    """
    # Validate field names up front
    unknown = set(fields.keys()) - _EDITABLE_FIELDS
    if unknown:
        raise ValueError(
            f"unknown/non-editable field(s): {sorted(unknown)}; "
            f"allowed: {sorted(_EDITABLE_FIELDS)}"
        )

    repos = load_existing_repos(out)
    repo: DiscoveredRepo | None = None
    for r in repos:
        if r.merge_key() == repo_id:
            repo = r
            break
    if repo is None:
        raise KeyError(repo_id)

    # Validate and apply each field
    if "repo_type" in fields:
        rt = fields["repo_type"]
        if rt not in _VALID_REPO_TYPES:
            raise ValueError(
                f"invalid repo_type {rt!r}; must be one of {sorted(_VALID_REPO_TYPES)}"
            )
        repo.repo_type = rt

    if "linked_paper_url" in fields:
        repo.linked_paper_url = fields["linked_paper_url"]

    if "description" in fields:
        repo.description = fields["description"]

    # Mark as manually overridden so the pipeline won't re-tag on next run
    repo.manual_override = True

    _write_repos_outputs(repos, out)
    return repo.to_full_dict()


def exclude_repo(out: Path, repo_id: str) -> dict:
    """Exclude a repo permanently: add to settings and remove from the DB.

    Steps:
    1. Append *repo_id* (``owner/repo``) to ``excluded_github_repos`` in
       settings.json (idempotent — skip if already present).
    2. Call ``config.reload_settings()`` so the in-process config reflects the
       change immediately.
    3. Remove the repo from the on-disk DB (both ``github-repos.json`` and
       ``github-repos-full.json``).

    Returns ``{"excluded": repo_id, "remaining": <count>, "was_present": <bool>}``.
    Note: the repo is added to the excluded list even if it is not currently in
    the DB (so it stays out on future runs).
    """
    # 1. Update settings — load current, add if not already listed, save.
    current_overrides = settings_store.load_overrides(config._SETTINGS_FILE)
    excluded_list: list[str] = list(current_overrides.get("excluded_github_repos", []))

    # Fall back to config's live value if the key isn't in the overrides file
    if not excluded_list:
        excluded_list = [r for r in config.EXCLUDED_GITHUB_REPOS]

    if repo_id not in excluded_list:
        excluded_list.append(repo_id)
        settings_store.save(config._SETTINGS_FILE, {"excluded_github_repos": excluded_list})
        config.reload_settings()

    # 2. Remove from DB (lenient: if not present, still report 0 remaining change)
    repos = load_existing_repos(out)
    before = len(repos)
    remaining = [r for r in repos if r.merge_key() != repo_id]
    was_present = len(remaining) < before

    _write_repos_outputs(remaining, out)

    return {
        "excluded": repo_id,
        "remaining": len(remaining),
        "was_present": was_present,
    }
