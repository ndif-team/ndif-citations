"""The catalog becomes public in Phase 2: no email addresses may be written (spec §5.1)."""
import json
import re

from ndif_citations.models import PipelineRun
from ndif_citations.output import write_outputs
from tests.conftest import make_paper

EMAIL = re.compile(r"(?:\{[^{}@]{1,200}\}|[\w.+-]+)@[\w-]+(?:\.[\w-]+)*\.[A-Za-z]{2,}\b")


def test_written_catalog_has_no_emails(tmp_path):
    p = make_paper(affiliations="Northeastern University jane.doe@northeastern.edu, MIT; "
                                 "TU X {a.b,c}@uni-x.de; MIT CSAIL {alice, bob}@mit.edu")
    p.ndif_context_windows = ["contact: bob@mit.edu — we use nnsight"]
    write_outputs([p], tmp_path, PipelineRun())
    raw = (tmp_path / "research-papers-full.json").read_text()
    assert not EMAIL.search(raw)
    saved = json.loads(raw)["verified"][0]
    assert "[email]" in saved["affiliations"] and "we use nnsight" in saved["ndif_context_windows"][0]
    assert saved["affiliations"].count("[email]") == 3 and "}" not in saved["affiliations"]
