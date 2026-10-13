"""Digest = what this run changed (spec §5.3)."""
from ndif_citations.digest import build_digest, fence, plain
from ndif_citations.models import Bucket, Category, PaperReason
from tests.conftest import make_paper


def cat(*papers):
    out = {"pending": [], "verified": [], "discarded": []}
    for p in papers:
        out[p.bucket.value].append(p.to_full_dict())
    return out


def test_lists_auto_published_discarded_and_pending():
    pub = make_paper(title="Auto Pub", arxiv_id="1", bucket=Bucket.VERIFIED, category=Category.USES_NNSIGHT,
                     url="https://arxiv.org/abs/1")
    pub.ndif_context_windows = ["we use nnsight to patch"]
    gone = make_paper(title="Zero Hits", arxiv_id="2", bucket=Bucket.DISCARDED)
    gone.reason = PaperReason.ZERO_PDF_HITS
    held = make_paper(title="Held", arxiv_id="3", bucket=Bucket.PENDING, category=Category.USES_NDIF)
    held.reason = PaperReason.NEEDS_REVIEW
    md, changed = build_digest(cat(), cat(pub, gone, held), [], [])
    assert changed
    assert "Auto Pub" in md and "https://arxiv.org/abs/1" in md and "/discard https://arxiv.org/abs/1" in md
    assert "Zero Hits" in md and "Held" in md


def test_curator_verified_paper_is_not_called_auto_published():
    human = make_paper(title="Human", arxiv_id="9", bucket=Bucket.VERIFIED, manual_override=True)
    md, _ = build_digest(cat(), cat(human), [], [])
    assert "Auto-published" not in md


def test_no_changes():
    p = make_paper(arxiv_id="5", bucket=Bucket.VERIFIED)
    md, changed = build_digest(cat(p), cat(p), [], [])
    assert changed is False


def test_age_out_flag_counts_as_change():
    _, changed = build_digest(cat(), cat(), [], [], age_out_skipped=37)
    assert changed


def test_fence_and_plain_neutralize_untrusted_text():
    assert "`" not in fence("```rm -rf``` @team #12").strip("`\n")
    assert plain("cc @maintainers") == "cc @\u200bmaintainers"
    assert plain("x <!-- hides the rest") == "x <\u200b!-- hides the rest"


def test_newly_unparseable_paper_counts_as_llm_fallback():
    p = make_paper(arxiv_id="7", bucket=Bucket.VERIFIED)
    held = make_paper(arxiv_id="7", bucket=Bucket.VERIFIED)
    held.unclassified_reason = "llm_unparseable"
    md, changed = build_digest(cat(p), cat(held), [], [])
    assert changed
    assert "**LLM fallbacks:** 1 paper(s)" in md
    # already unparseable last week -> not new
    _, changed2 = build_digest(cat(held), cat(held), [], [])
    assert changed2 is False


def test_plain_collapses_newlines():
    assert plain("x\n# Heading\n@team") == "x # Heading @\u200bteam"


def test_unsafe_urls_are_dropped_and_safe_ones_keep_hint():
    for bad in ("https://x.org/a`b", "https://x.org/a\n# Injected", "https://x.org/a\n", "https://x.org/<!--", "javascript:alert(1)"):
        p = make_paper(title="T", arxiv_id="1", bucket=Bucket.VERIFIED, url=bad)
        p.pdf_url = None
        md, _ = build_digest(cat(), cat(p), [], [])
        assert "`/discard " not in md and "Injected" not in md and "`b" not in md
    ok = make_paper(title="T", arxiv_id="2", bucket=Bucket.VERIFIED, url="https://arxiv.org/abs/2")
    md, _ = build_digest(cat(), cat(ok), [], [])
    assert "`/discard https://arxiv.org/abs/2`" in md
