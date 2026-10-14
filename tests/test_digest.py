"""Digest = what this run changed (spec §5.3)."""
from ndif_citations.digest import build_digest, fence, inline, plain
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


def test_inline_renders_untrusted_markdown_inert():
    assert "<details>" not in inline("<details><summary>x</summary>")
    link = inline("[x](https://evil)")
    assert "\\[x\\]" in link and "https://" not in link
    assert "#12" not in inline("see #12") and "#3" not in inline("owner/repo#3")
    assert "://" not in inline("https://evil.example") and "www." not in inline("www.evil.example")
    assert "GH-12" not in inline("GH-12") and "gh-12" not in inline("gh-12").lower()
    assert inline("~~x~~ $y$") == "\\~\\~x\\~\\~ \\$y\\$"
    assert "![" not in inline("![x](https://e/i.png)")
    mix = inline("<details><summary>ok</summary> [click](https://evil.example) *b* _i_ `c` @team")
    assert all(f"\\{c}" in mix for c in "<>[]*_`") and "@\u200bteam" in mix and "\u200b" in inline("#1")


def test_inline_keeps_html_entities_literal():
    out = inline("&#64;x &#35;1 &commat;team")
    assert "&#64;" not in out and "&#35;" not in out
    assert out.count("\\&") == 3 and out.count("&") == 3  # every & escaped: no entity decodes to @ or #


def test_digest_body_uses_inline_for_titles_and_reasons():
    pub = make_paper(title="[click](https://evil.example) <details>", arxiv_id="1", bucket=Bucket.VERIFIED,
                     url="https://arxiv.org/abs/1")
    held = make_paper(title="see #12 ~~x~~", arxiv_id="3", bucket=Bucket.PENDING)
    held.reason = PaperReason.NEEDS_REVIEW
    md, _ = build_digest(cat(), cat(pub, held), [], [])
    assert "<details>" not in md and "](https://evil" not in md and "#12" not in md and "~~x" not in md
    assert "needs\\_review" in md and "/discard https://arxiv.org/abs/1" in md  # safe URLs untouched


def _published(n, title_len=20):
    out = []
    for i in range(n):
        p = make_paper(title=f"P{i} " + "x" * title_len, arxiv_id=str(i), bucket=Bucket.VERIFIED,
                       url=f"https://arxiv.org/abs/{i}")
        p.ndif_context_windows = ["we use nnsight " * 30]
        out.append(p)
    return out


def test_digest_caps_each_section_for_githubs_body_limit():
    md, _ = build_digest(cat(), cat(*_published(300)), [], [])
    assert len(md) < 65_536
    assert "**Auto-published (300)**" in md
    assert "+250 more — full list in output/research-papers-full.json on main" in md
    assert "run summary" not in md  # the step summary gets this same capped text
    assert "P49 " in md and "P50 " not in md


def test_digest_body_is_hard_truncated():
    md, _ = build_digest(cat(), cat(*_published(300, title_len=1500)), [], [])
    assert len(md) < 60_200 and "truncated" in md and md.rstrip().endswith("output/research-papers-full.json on main.")
    assert md.count("```") % 2 == 0  # a cut inside a fence is closed before the note
