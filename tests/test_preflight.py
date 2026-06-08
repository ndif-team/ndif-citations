from ndif_citations import preflight


def test_papers_run_blocks_without_llm(monkeypatch):
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_x")
    r = preflight.preflight(skip_papers=False, skip_github=True)
    assert r["ok"] is False
    assert any("LLM_API_KEY" in b for b in r["blocking"])


def test_repos_run_blocks_without_github(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "x")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    r = preflight.preflight(skip_papers=True, skip_github=False)
    assert r["ok"] is False
    assert any("GITHUB_TOKEN" in b for b in r["blocking"])


def test_full_run_ok_when_both_present(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "x")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_x")
    r = preflight.preflight(skip_papers=False, skip_github=False)
    assert r["ok"] is True
    assert r["blocking"] == []


def test_optional_keys_warn(monkeypatch):
    monkeypatch.setenv("LLM_API_KEY", "x")
    monkeypatch.delenv("S2_API_KEY", raising=False)
    r = preflight.preflight(skip_papers=False, skip_github=True)
    assert r["ok"] is True
    assert any("S2_API_KEY" in w for w in r["warnings"])
