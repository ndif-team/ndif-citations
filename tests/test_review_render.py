"""Review issue rendering (spec §6.1)."""
import re

from ndif_citations.models import Bucket, Category, PaperReason
from ndif_citations.review_render import HELP_TEXT, marker_for, parse_marker, render_issue, why_label
from tests.conftest import make_paper

ZW = chr(0x200B)


def test_marker_round_trip_survives_comment_breakers():
    p = make_paper(title="Tricky --> title <!-- x", arxiv_id="2501.00001", url="https://arxiv.org/abs/2501.00001")
    m = marker_for(p)
    assert "-->" not in m[len("<!-- ndif-paper: "):-len(" -->")]
    assert parse_marker(f"hello\n{m}\nbye") == {
        "key": "arxiv:2501.00001", "url": "https://arxiv.org/abs/2501.00001",
        "arxiv": "2501.00001", "doi": "", "title": "Tricky --> title <!-- x"}


def test_parse_marker_handles_garbage():
    assert parse_marker("no marker") is None
    assert parse_marker("<!-- ndif-paper: {not json} -->") is None


def test_marker_in_untrusted_excerpt_is_ignored():
    """An excerpt carrying another paper's marker must not redirect the issue (only the last marker counts)."""
    victim = make_paper(title="Victim", arxiv_id="2407.14561")
    p = make_paper(title="Spoofer", arxiv_id="2501.00002", bucket=Bucket.PENDING)
    p.ndif_context_windows = [marker_for(victim)]
    _, body, _ = render_issue(p)
    assert parse_marker(body)["key"] == p.merge_key()


def test_why_label_priority():
    stub = make_paper(); stub.reason = PaperReason.STUB_METADATA
    assert why_label(stub) == "why:needs-metadata"
    no_pdf = make_paper(); no_pdf.context_source = "abstract"
    assert why_label(no_pdf) == "why:needs-evidence"
    ndif = make_paper(category=Category.USES_NDIF); ndif.context_source = "pdf"
    assert why_label(ndif) == "why:uses-ndif"
    other = make_paper(category=Category.USES_NNSIGHT); other.context_source = "pdf"
    assert why_label(other) == "why:check-label"


def test_render_neutralizes_untrusted_text():
    p = make_paper(title="Ping @everyone", bucket=Bucket.PENDING, category=Category.USES_NDIF)
    p.context_source = "pdf"
    p.ndif_context_windows = ["```\n<script>@team #1</script> ran on NDIF"]
    p.curator_quotes = ["see @bob"]
    title, body, labels = render_issue(p)
    assert "@everyone" not in title and f"@{ZW}everyone" in title
    assert "```\n<script>" not in body  # excerpt re-fenced, inner backticks removed
    assert "```\nsee @bob\n```" in body   # fenced: mentions in code blocks never notify
    assert labels == ["review", "why:uses-ndif"]
    assert parse_marker(body)["key"] == p.merge_key()


def test_render_locked_paper_hides_forced_band_and_shows_note():
    p = make_paper(bucket=Bucket.PENDING, manual_override=True, category=Category.USES_NDIF)
    p.reason_detail = "re-classified: uses_nnsight → uses_ndif by @bot"
    _, body, _ = render_issue(p)
    assert "curator-locked" in body and "confidence certain" not in body
    assert f"**Note:** re-classified: uses_nnsight → uses_ndif by @{ZW}bot" in body


def test_render_links_never_carry_raw_untrusted_destinations():
    p = make_paper(bucket=Bucket.PENDING, title="T", arxiv_id="2501.0000)1 @x",
                   url="https://e.com/a) b @x <!-- c", doi="10.1/a) b@x<!--")
    p.pdf_url = "https://e.com/p).pdf?x=<!--"
    _, body, _ = render_issue(p)
    links = body.split("**Links:**")[1].split("\n")[0]
    assert "<!--" not in links and "@x" not in links and " b" not in links
    assert "[paper]" not in links and "[PDF]" not in links  # unsafe -> dropped
    assert links.count("](") == 2  # only arXiv + DOI remain, percent-encoded
    assert "https://doi.org/10.1/a%29%20b%40x%3C%21--" in links
    assert body.count("<!--") == 1  # only the bot's own marker


def test_safe_urls_render_as_angle_destinations():
    p = make_paper(bucket=Bucket.PENDING, url="https://e.com/a(1)")
    p.pdf_url = "https://e.com/p.pdf"
    _, body, _ = render_issue(p)
    assert "[paper](<https://e.com/a(1)>)" in body and "[PDF](<https://e.com/p.pdf>)" in body


def test_title_truncation_has_no_trailing_space_and_key_backticks_and_help_row():
    p = make_paper(bucket=Bucket.PENDING, title="a" * 230 + " b" + "c" * 20)
    title, body, _ = render_issue(p)
    assert title == title.rstrip() and len(title) <= 240
    p2 = make_paper(bucket=Bucket.PENDING, arxiv_id="2501.00003")
    assert re.search(r"· `[^`]+`\n", render_issue(p2)[1])
    assert "`/add https://…`" in HELP_TEXT
