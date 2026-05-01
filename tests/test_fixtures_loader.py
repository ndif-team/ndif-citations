"""Validates that every *.txt fixture has a matching *.expected.json."""
import json
from pathlib import Path

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "papers"

VALID_CATEGORIES = {"uses_ndif", "uses_nnsight", "referencing", "unclassified"}


def test_every_txt_has_expected_json():
    txt_files = sorted(FIXTURES_DIR.glob("*.txt"))
    assert len(txt_files) >= 10, f"Expected at least 10 fixtures, found {len(txt_files)}"
    for txt in txt_files:
        expected = txt.with_suffix(".expected.json")
        assert expected.exists(), f"Missing {expected.name} for {txt.name}"


def test_expected_json_schema():
    for json_file in FIXTURES_DIR.glob("*.expected.json"):
        data = json.loads(json_file.read_text())
        assert "expected_category" in data, f"{json_file.name} missing expected_category"
        assert data["expected_category"] in VALID_CATEGORIES, (
            f"{json_file.name}: unknown category '{data['expected_category']}'"
        )
        assert "expected_keyword_hits_min" in data
        assert "expected_keyword_hits_max" in data
        assert data["expected_keyword_hits_min"] <= data["expected_keyword_hits_max"]
        assert "notes" in data


def test_fixtures_are_not_empty():
    for txt in FIXTURES_DIR.glob("*.txt"):
        content = txt.read_text().strip()
        assert content, f"{txt.name} is empty"
        assert len(content) < 2048, f"{txt.name} exceeds 2KB limit ({len(content)} bytes)"


def test_required_scenarios_present():
    names = {f.stem for f in FIXTURES_DIR.glob("*.txt")}
    required = {
        "methods_use_ndif",
        "methods_use_nnsight",
        "acks_only_ndif",
        "acks_only_nnsight",
        "related_work_only",
        "references_only",
        "negative_evidence",
        "comparison_table",
        "no_mentions",
        "bibtex_present",
    }
    missing = required - names
    assert not missing, f"Missing required fixtures: {missing}"
