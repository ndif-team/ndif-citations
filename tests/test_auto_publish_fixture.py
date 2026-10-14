"""The auto-publish phrase gate vs human labels (spec §4.7). Fixture: tests/helpers/build_labeled_fixture.py."""
import json
from pathlib import Path

from ndif_citations import config
from ndif_citations.models import Bucket, Category, Confidence
from ndif_citations.process import _passes_auto_publish_rule
from tests.conftest import make_paper

ROWS = json.loads((Path(__file__).parent / "fixtures" / "labeled_evidence.json").read_text())


def _as_llm_says(row, category):
    p = make_paper(title=row["title"], category=category, bucket=Bucket.PENDING)
    p.context_source = "pdf"
    p.category_confidence_band = Confidence.HIGH
    p.classification_signal = "llm"
    p.ndif_context_windows = row["windows"]
    return p


def test_fixture_is_present_and_sized():
    assert len(ROWS) >= 90


def test_worst_case_llm_never_publishes_a_human_referencing_paper_as_usage():
    """Even if the LLM called EVERY paper uses_nnsight, no human-referencing paper passes."""
    false_pos = [r["title"] for r in ROWS
                 if r["category"] == "referencing"
                 and _passes_auto_publish_rule(_as_llm_says(r, Category.USES_NNSIGHT))]
    assert false_pos == []


def test_rule_still_covers_most_usage_papers():
    usage = [r for r in ROWS if r["category"] == "uses_nnsight"]
    covered = [r for r in usage if _passes_auto_publish_rule(_as_llm_says(r, Category.USES_NNSIGHT))]
    assert len(covered) >= 0.6 * len(usage), f"{len(covered)}/{len(usage)}"


def test_no_emails_in_fixture():
    assert "@" not in json.dumps(ROWS).replace("@\u200b", "")
