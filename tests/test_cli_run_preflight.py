"""CLI run must stop before the pipeline when preflight blocks (spec §4.2)."""
from click.testing import CliRunner

from ndif_citations.cli import cli


def test_run_exits_1_when_llm_check_fails(monkeypatch, tmp_path):
    monkeypatch.setenv("LLM_API_KEY", "x")
    monkeypatch.setenv("GITHUB_TOKEN", "t")
    monkeypatch.setattr("ndif_citations.key_validation.test_llm",
                        lambda *a, **k: {"ok": False, "detail": "HTTP 410"})
    ran = []
    monkeypatch.setattr("ndif_citations.orchestrator.run_pipeline", lambda *a, **k: ran.append(1))
    result = CliRunner().invoke(cli, ["run", "--output-dir", str(tmp_path)])
    assert result.exit_code == 1
    assert "LLM check failed (HTTP 410)" in result.output
    assert ran == []
