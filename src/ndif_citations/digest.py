"""Weekly digest — what one catalog run changed (spec §5.3). Pure functions."""
from __future__ import annotations

import re
from datetime import datetime, timezone

_SAFE_URL = re.compile(r'^https?://[^\s<>`"\[\]]+$')


def plain(text: str) -> str:
    """Neutralize @-mentions and HTML comments (which would hide the rest of the body) in untrusted Markdown."""
    return " ".join((text or "").split()).replace("@", "@\u200b").replace("<!--", "<\u200b!--")


def fence(text: str, limit: int = 400) -> str:
    """Untrusted excerpt → a code fence (no Markdown/HTML/#refs/mentions render inside)."""
    body = " ".join((text or "").replace("`", "'").split())[:limit]
    return f"```\n{body}\n```"


def _flatten(full: dict | None) -> dict[str, dict]:
    from ndif_citations.models import DiscoveredPaper
    out: dict[str, dict] = {}
    for bucket in ("pending", "verified", "discarded"):
        for item in (full or {}).get(bucket, []):
            out[DiscoveredPaper.model_validate(item).merge_key()] = {**item, "bucket": bucket}
    return out


def _link(p: dict) -> str:
    """First of url/pdf_url that is a plain http(s) URL (verbatim, so /discard matches the stored URL), else ""."""
    for u in (p.get("url"), p.get("pdf_url")):
        if u and _SAFE_URL.match(u):
            return u
    return ""


def _no_llm_answer(p: dict) -> bool:
    return p.get("classification_signal") == "keyword_fallback" or p.get("unclassified_reason") == "llm_unparseable"


def build_digest(before: dict | None, after: dict, before_repos: list | None, after_repos: list, *,
                 age_out_skipped: int = 0, run_url: str | None = None) -> tuple[str, bool]:
    """Markdown digest + whether anything worth a human glance happened."""
    b, a = _flatten(before), _flatten(after)
    became = lambda bucket: [p for k, p in a.items()  # noqa: E731
                             if p["bucket"] == bucket and b.get(k, {}).get("bucket") != bucket]
    published = [p for p in became("verified") if not p.get("manual_override")]
    discarded = [p for p in became("discarded") if not p.get("manual_override")]
    pending = became("pending")
    fallbacks = [p for k, p in a.items() if _no_llm_answer(p) and not _no_llm_answer(b.get(k, {}))]
    br = {f"{r['owner']}/{r['repo']}" for r in (before_repos or [])}
    ar = {f"{r['owner']}/{r['repo']}" for r in (after_repos or [])}
    repos_added, repos_removed = sorted(ar - br), sorted(br - ar)

    lines = [f"## Catalog run — {datetime.now(timezone.utc):%Y-%m-%d}"]
    if run_url:
        lines.append(f"[Run log]({run_url})")
    if published:
        lines.append(f"\n**Auto-published ({len(published)})** — glance at these; to remove one, "
                     "comment the `/discard` line shown under it.")
        for p in published:
            windows = p.get("ndif_context_windows") or []
            lines.append(f"- **{plain(p['title'])}** — {p.get('category')} · {_link(p)}")
            if windows:
                lines.append(fence(windows[0], 300))
            if _link(p):
                lines.append(f"  `/discard {_link(p)}` · `/verify {_link(p)} uses_ndif|uses_nnsight|referencing`")
    if discarded:
        lines.append(f"\n**Auto-discarded ({len(discarded)})** — the full PDF never mentions NDIF/NNsight.")
        lines += [f"- {plain(p['title'])} · {_link(p)}" for p in discarded]
    if pending:
        lines.append(f"\n**Needs review ({len(pending)})**")
        lines += [f"- {plain(p['title'])} — {p.get('reason') or 'needs_review'} · {_link(p)}" for p in pending]
    if fallbacks:
        lines.append(f"\n**LLM fallbacks:** {len(fallbacks)} paper(s) got no usable LLM answer (keyword rules "
                     "after an LLM error, or an unparseable reply) — they are held for review.")
    if repos_added or repos_removed or age_out_skipped:
        lines.append(f"\n**Repos:** +{len(repos_added)} new, −{len(repos_removed)} removed.")
        if age_out_skipped:
            lines.append(f"⚠ Age-out skipped: {age_out_skipped} repos looked stale at once — the "
                         "dependents scrape is probably partial; nothing was removed.")
    counts = {k: sum(1 for p in a.values() if p["bucket"] == k) for k in ("verified", "pending", "discarded")}
    lines.append(f"\nTotals: {counts['verified']} verified · {counts['pending']} pending · "
                 f"{counts['discarded']} discarded · {len(ar)} repos")
    has_changes = bool(published or discarded or pending or fallbacks or repos_added
                       or repos_removed or age_out_skipped)
    return "\n".join(lines) + "\n", has_changes
