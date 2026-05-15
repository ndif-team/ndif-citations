"""Multi-source venue resolution and normalization.

Produces canonical venue strings for the website's research-page listing:
    - Preprints:    "ArXiv 2025"
    - Conferences:  "ICML 2025", "EACL 2026", "Nature Methods 2024"
    - Workshops:    "BlackboxNLP 2025", "ICLR 2026 Workshop on Foo"
    - Posters/Spotlights: stripped to bare venue (no "Poster" suffix)

Public surface:
    - resolve_venue(paper, sources) -> str
    - normalize_venue(venue, year) -> str
    - decode_doi_prefix(doi) -> str        # exposed for testing
"""

from __future__ import annotations

import re
from typing import Optional

from ndif_citations import config


# ---------------------------------------------------------------------------
# Static rule tables
# ---------------------------------------------------------------------------

# DOI prefix → canonical (short_name, year-from-capture-group).
# Order matters — more specific patterns first.
_DOI_PREFIX_RULES: list[tuple[re.Pattern, str]] = [
    (re.compile(r"^10\.18653/v1/(\d{4})\.findings-emnlp", re.I), "Findings of EMNLP {year}"),
    (re.compile(r"^10\.18653/v1/(\d{4})\.findings-acl",   re.I), "Findings of ACL {year}"),
    (re.compile(r"^10\.18653/v1/(\d{4})\.findings-naacl", re.I), "Findings of NAACL {year}"),
    (re.compile(r"^10\.18653/v1/(\d{4})\.blackboxnlp",    re.I), "BlackboxNLP {year}"),
    (re.compile(r"^10\.18653/v1/(\d{4})\.acl-",           re.I), "ACL {year}"),
    (re.compile(r"^10\.18653/v1/(\d{4})\.naacl-",         re.I), "NAACL {year}"),
    (re.compile(r"^10\.18653/v1/(\d{4})\.emnlp-",         re.I), "EMNLP {year}"),
    (re.compile(r"^10\.18653/v1/(\d{4})\.eacl-",          re.I), "EACL {year}"),
    (re.compile(r"^10\.18653/v1/(\d{4})\.aacl-",          re.I), "AACL {year}"),
    # FAccT (ACM Conference on Fairness, Accountability, and Transparency)
    (re.compile(r"^10\.1145/3715275",                     re.I), "FAccT 2025"),
    (re.compile(r"^10\.1145/3630106",                     re.I), "FAccT 2024"),
    # arXiv-as-DOI: explicit preprint
    (re.compile(r"^10\.48550/arxiv\.",                    re.I), "ArXiv"),
]


# Strings that should be treated as preprint/repository placeholders, not venues.
# Match against the WHOLE normalized venue (case-insensitive).
_PREPRINT_SENTINELS: list[re.Pattern] = [
    re.compile(r"^arxiv(\.org)?(\s+\d{4})?$", re.I),
    re.compile(r"^arxiv\s+e-?prints?(\s+\d{4})?$", re.I),
    re.compile(r"^arxiv\s+preprint.*$", re.I),
    re.compile(r"^corr(\s+\d{4})?$", re.I),
    re.compile(r"^biorxiv(\s+\d{4})?$", re.I),
    re.compile(r"^medrxiv(\s+\d{4})?$", re.I),
    re.compile(r"^ssrn.*$", re.I),
    re.compile(r"^available at ssrn.*$", re.I),
    re.compile(r"^uva-dare.*$", re.I),
    re.compile(r"^openreview\.net$", re.I),
    re.compile(r"^https?://.*$", re.I),  # bare URLs leaked in
    re.compile(r"^url\s+https?://.*$", re.I),  # "URL https://arxiv.org/..." Scholar leak
    # Truncation residue (highly specific to avoid eating real venues):
    # year-only
    re.compile(r"^\d{4}$"),
    # ends with a dangling preposition + year (the venue name was eaten by Scholar)
    re.compile(r"^.*\b(of|on|at|in|for|the)\s+\d{4}$", re.I),
    # bare conference-shell phrases (acronym/proper-noun was eaten)
    re.compile(r"^international conference on(\s+\d{4})?$", re.I),
    re.compile(r"^conference on(\s+\d{4})?$", re.I),
]


# Suffixes to strip from the END of a venue string (case-insensitive).
# Workshops are NOT in this list — they're preserved per Pranav's spec.
_STRIP_SUFFIXES = [
    " Poster",
    " Spotlight",
    " Oral",
    " Conference Desk Rejected Submission",
    " Desk Rejected Submission",
]


# Known long-form → short-form mappings. Loaded from data/known_venues.json
# (acronym_map section), with fallback defaults below if config not loaded.
_DEFAULT_ACRONYM_MAP = {
    "International Conference on Machine Learning": "ICML",
    "Neural Information Processing Systems": "NeurIPS",
    "Advances in Neural Information Processing Systems": "NeurIPS",
    "International Conference on Learning Representations": "ICLR",
    "Annual Meeting of the Association for Computational Linguistics": "ACL",
    "North American Chapter of the Association for Computational Linguistics": "NAACL",
    "Conference of the European Chapter of the Association for Computational Linguistics": "EACL",
    "Conference on Empirical Methods in Natural Language Processing": "EMNLP",
    "Conference on Fairness, Accountability and Transparency": "FAccT",
    "Conference on Fairness, Accountability, and Transparency": "FAccT",
    "AAAI Conference on Artificial Intelligence": "AAAI",
    "International Joint Conference on Artificial Intelligence": "IJCAI",
}


def _acronym_map() -> dict[str, str]:
    """Resolve the long→short map from config.KNOWN_VENUES, with built-in fallback."""
    am = (config.KNOWN_VENUES or {}).get("acronym_map") or {}
    if am:
        return am
    return _DEFAULT_ACRONYM_MAP


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decode_doi_prefix(doi: str) -> str:
    """Decode a DOI prefix into a canonical venue string. Returns "" on no match.

    The {year} placeholder in the rule template is replaced by the regex's
    first capture group when present.
    """
    if not doi:
        return ""
    d = doi.strip()
    # Strip URL prefix if someone passed a full doi.org URL
    d = re.sub(r"^https?://(dx\.)?doi\.org/", "", d, flags=re.I)
    for pattern, template in _DOI_PREFIX_RULES:
        m = pattern.search(d)
        if not m:
            continue
        year = m.group(1) if m.groups() else ""
        return template.format(year=year).strip()
    return ""


def is_preprint_sentinel(venue: str) -> bool:
    """True if `venue` is a preprint-server / repository / URL placeholder OR mangled output.

    Mangled output happens when a free-form bibliographic string is fed through the
    acronym map and partial cleanup. We reject such results so the resolve_venue
    cascade falls through to the next source rather than emit something worse than
    the input. Specifically:
      - Unbalanced parentheses (e.g. "Foo (NeurIPS 2025" — close paren got stripped)
      - Leading ordinal ("39th Conference on ...", "19th 2026", "14th 2025")
    """
    if not venue or not venue.strip():
        return True
    v = venue.strip()
    if any(p.match(v) for p in _PREPRINT_SENTINELS):
        return True
    if v.count("(") != v.count(")"):
        return True
    if re.match(r"^\d+(st|nd|rd|th)\b", v, re.I):
        return True
    return False


def has_known_venue_token(venue: str) -> bool:
    """True if `venue` contains a recognized conference acronym or journal name.

    Used as the confidence gate in `resolve_venue` — a normalized venue is only
    accepted when it contains a known token. Anything else (truncated Scholar
    strings like "Advances in Neural 2025", unrecognized one-off venues like
    "Handbook of Human 2025") is treated as junk and the cascade falls through
    to the ArXiv fallback. This is intentional: a paper whose venue we can't
    confidently identify should display as "ArXiv {year}" and end up in pending
    for human review, rather than carry a half-broken venue label.

    Recognition sources:
      - config.KNOWN_VENUES["conferences"]: word-boundary match (case-insensitive)
      - config.KNOWN_VENUES["journals"]: substring match (case-insensitive)
      - acronym_map keys (long-form names like "International Conference on Machine
        Learning"): substring match — these are recognized because we know how to
        map them to a clean acronym, so a pre-cleanup long-form should pass the gate.
    """
    if not venue:
        return False
    v = venue.strip()
    if not v:
        return False
    confs = (config.KNOWN_VENUES or {}).get("conferences", []) or []
    for conf in confs:
        if re.search(rf"\b{re.escape(conf)}\b", v, re.I):
            return True
    v_lower = v.lower()
    journals = (config.KNOWN_VENUES or {}).get("journals", []) or []
    for j in journals:
        if j.lower() in v_lower:
            return True
    for long_form in _acronym_map():
        if long_form.lower() in v_lower:
            return True
    return False


def is_confident_venue(venue: str) -> bool:
    """True if `venue` is a real, confidently-recognized venue string.

    A venue passes when it's not a sentinel/mangled string AND contains a
    recognized acronym or journal name. Used to gate each source in the
    `resolve_venue` cascade — fall through to the next source if False.
    Also used by the merge layer (output._update_existing) to decide whether
    to protect an existing venue from being overwritten by an ArXiv fallback.
    """
    if not venue or is_preprint_sentinel(venue):
        return False
    return has_known_venue_token(venue)


# Backwards-compat alias for internal callers
_is_confident_venue = is_confident_venue


def normalize_venue(venue: str, year: int = 0) -> str:
    """Normalize a raw venue string. Returns "" if it normalizes to a preprint sentinel.

    Pipeline:
        1. Strip ellipsis (U+2026) and "..." truncation residue
        2. Strip leading "Proceedings of (the) (Nth)" prefix
        3. Strip trailing "(Volume N: ...)" tail
        4. Strip trailing Poster/Spotlight/Oral suffixes
        5. Strip "(Cornell University)" parenthetical
        6. Long-form → acronym substitution
        7. Collapse repeated year (e.g. "2025 2025" -> "2025")
        8. Append year if missing and one was provided
        9. Return "" if result matches a preprint sentinel
    """
    if not venue:
        return ""
    v = venue.strip()

    # 1. Ellipsis / truncation residue
    v = v.replace("…", "")  # U+2026 horizontal ellipsis
    v = re.sub(r"\.{3,}", "", v)  # literal "..." in any quantity
    v = re.sub(r"\s+", " ", v).strip(" ,;-")

    # 2. Leading "Proceedings of (the) (Nth)"
    v = re.sub(
        r"^Proceedings?\s+of\s+(?:the\s+)?(?:\d+(?:st|nd|rd|th)\s+)?",
        "",
        v,
        flags=re.I,
    ).strip()

    # 3. "(Volume N: ...)" tail (ACL Anthology long names) — may sit before a trailing year
    v = re.sub(r"\s*\(Volume\s+\d+:[^)]*\)", "", v, flags=re.I).strip()

    # 4. Strip Poster/Spotlight/Oral wherever they sit (often before a trailing year).
    # Match as a whole word so "Posterior" etc. aren't damaged.
    for suffix in _STRIP_SUFFIXES:
        v = re.sub(rf"\s+{re.escape(suffix.strip())}\b", "", v, flags=re.I)
    v = re.sub(r"\s+", " ", v).strip()

    # 5. (Cornell University) parenthetical
    v = re.sub(r"\s*\(Cornell\s+University\)\s*", " ", v, flags=re.I).strip()

    # 5b. Leading article ("the ICLR 2026" → "ICLR 2026") — common in arXiv
    # journal_ref strings where the comment was prosaic rather than structured.
    v = re.sub(r"^(the|a|an)\s+", "", v, flags=re.I).strip()

    # 6. Long-form → acronym (try longest keys first to avoid partial matches)
    am = _acronym_map()
    for long_form in sorted(am, key=len, reverse=True):
        if long_form.lower() in v.lower():
            # Substitute the long form with the acronym, preserving year/suffix
            v = re.sub(re.escape(long_form), am[long_form], v, flags=re.I).strip()
            break

    # 7. Collapse repeated year
    v = re.sub(r"\b(20\d{2})\s+\1\b", r"\1", v).strip()

    # Cleanup: strip ", " or " , " before a year (left over by acronym substitution
    # of "X, YYYY" sources like arXiv journal_ref → "ICLR, 2025"); collapse whitespace.
    v = re.sub(r"\s*,\s+(?=\d{4}\b)", " ", v)
    v = re.sub(r"\s+", " ", v).strip(" ,;-")

    # 9. Preprint sentinel check (must come BEFORE appending year — "ArXiv 2025"
    # would otherwise be stripped on the second normalize call but we want it
    # to round-trip cleanly when called from resolve_venue's fallback path).
    if is_preprint_sentinel(v):
        return ""

    # 8. Append year if missing
    if year > 0:
        has_year = bool(re.search(r"\b20\d{2}\b", v))
        if not has_year:
            v = f"{v} {year}".strip()

    return v


def _parse_arxiv_comment(text: str) -> str:
    """Extract a venue from an arXiv <comment> or <journal-ref> string.

    Recognized leads:
        "Accepted at <X> <YYYY>"
        "Accepted to <X> <YYYY>"
        "Accepted in <X> <YYYY>"
        "Published in <X> <YYYY>"
        "Appeared at <X> <YYYY>"
        "To appear in <X> <YYYY>"
        "Camera-ready for <X> <YYYY>"

    Returns "" if no recognized pattern matches.
    """
    if not text:
        return ""
    pat = re.compile(
        r"\b(?:Accepted (?:at|to|in)|Published (?:in|at)|Appeared (?:at|in)|"
        r"Presented at|To appear (?:in|at)|Camera-ready for)\b\s+"
        r"(.+?)\s+\b(20\d{2})\b",
        re.I,
    )
    m = pat.search(text)
    if not m:
        return ""
    venue = m.group(1).strip().rstrip(",.;:")
    year = m.group(2)
    if not venue:
        return ""
    return f"{venue} {year}"


def resolve_venue(
    paper,
    sources: Optional[dict] = None,
    *,
    title_search_fn=None,
) -> str:
    """Pick the canonical venue for `paper` from all available sources.

    Priority cascade — first source whose normalized output is non-empty wins:
        1. DOI prefix decode (deterministic, no API)
        2. arXiv <journal-ref> parsed via _parse_arxiv_comment
        3. arXiv <comment> parsed via _parse_arxiv_comment
        4. OpenAlex primary_location.source.display_name (sources["openalex"])
        5. S2 publicationVenue.name (sources["s2"])
        6. CrossRef container_title (sources["crossref"])
        7. OpenReview venue (sources["openreview"])
        8. Existing paper.venue if non-placeholder
        9. title_search_fn(paper.title) if provided and 1-8 produced nothing
       10. Final fallback: "ArXiv {year}"

    `sources` is an optional dict with the raw venue strings each upstream
    source returned (already extracted by the enrichment pass). Each key is
    optional; missing keys are skipped.

    `title_search_fn` is an optional callable — if provided, called only when
    all higher-priority sources are empty AND paper has no arxiv_id/doi/openalex_id.
    """
    sources = sources or {}
    year = paper.year if paper.year > 0 else 0

    # 1. DOI prefix decode — deterministic & trusted (hardcoded mappings).
    # DOI-decoded results bypass the confidence gate because they're synthesized
    # from known prefixes (ACL, EACL, FAccT, etc.) which are all in our known list.
    if paper.doi:
        decoded = decode_doi_prefix(paper.doi)
        if decoded:
            normalized = normalize_venue(decoded, year)
            if normalized:
                return normalized

    # 2. arXiv journal-ref — only trust the structured "Accepted at X YYYY" form.
    # The raw journal_ref is free-form bibliographic text (e.g.,
    # "39th Conference on Neural Information Processing Systems (NeurIPS 2025)")
    # and routinely produces mangled venues if we feed it directly to normalize.
    jr = sources.get("arxiv_journal_ref") or ""
    if jr:
        parsed = _parse_arxiv_comment(jr)
        if parsed:
            normalized = normalize_venue(parsed, year)
            if _is_confident_venue(normalized):
                return normalized

    # 3. arXiv comment
    cmt = sources.get("arxiv_comment") or ""
    if cmt:
        parsed = _parse_arxiv_comment(cmt)
        if parsed:
            normalized = normalize_venue(parsed, year)
            if _is_confident_venue(normalized):
                return normalized

    # 4-7. External-source venue strings, in priority order
    for key in ("openalex", "s2", "crossref", "openreview"):
        raw = sources.get(key) or ""
        if not raw:
            continue
        normalized = normalize_venue(raw, year)
        if _is_confident_venue(normalized):
            return normalized

    # 8. Existing paper.venue
    existing = paper.venue or ""
    if existing:
        normalized = normalize_venue(existing, year)
        if _is_confident_venue(normalized):
            return normalized

    # 9. Title-search fallback — only when we have NO identifiers to retry with
    has_id = bool(paper.arxiv_id or paper.doi or paper.openalex_id)
    if not has_id and title_search_fn and paper.title:
        searched = title_search_fn(paper.title) or ""
        if searched:
            normalized = normalize_venue(searched, year)
            if _is_confident_venue(normalized):
                return normalized

    # 10. Final fallback — display as preprint; the paper goes to pending for
    # human review where the venue can be set via manual_override.
    return f"ArXiv {year}" if year > 0 else "ArXiv"
