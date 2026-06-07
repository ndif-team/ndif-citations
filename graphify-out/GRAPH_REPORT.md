# Graph Report - ndif-citations  (2026-06-05)

## Corpus Check
- 97 files · ~1,921,401 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 1906 nodes · 5665 edges · 30 communities detected
- Extraction: 39% EXTRACTED · 61% INFERRED · 0% AMBIGUOUS · INFERRED: 3439 edges (avg confidence: 0.63)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]
- [[_COMMUNITY_Community 9|Community 9]]
- [[_COMMUNITY_Community 10|Community 10]]
- [[_COMMUNITY_Community 11|Community 11]]
- [[_COMMUNITY_Community 12|Community 12]]
- [[_COMMUNITY_Community 13|Community 13]]
- [[_COMMUNITY_Community 14|Community 14]]
- [[_COMMUNITY_Community 15|Community 15]]
- [[_COMMUNITY_Community 16|Community 16]]
- [[_COMMUNITY_Community 17|Community 17]]
- [[_COMMUNITY_Community 18|Community 18]]
- [[_COMMUNITY_Community 19|Community 19]]
- [[_COMMUNITY_Community 20|Community 20]]
- [[_COMMUNITY_Community 21|Community 21]]
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 32|Community 32]]

## God Nodes (most connected - your core abstractions)
1. `DiscoveredPaper` - 270 edges
2. `Category` - 264 edges
3. `make_paper()` - 222 edges
4. `Bucket` - 194 edges
5. `DiscoveredRepo` - 171 edges
6. `PaperReason` - 168 edges
7. `Confidence` - 139 edges
8. `JobRunner` - 133 edges
9. `PipelineRun` - 127 edges
10. `ProcessingBucket` - 106 edges

## Surprising Connections (you probably didn't know these)
- `Tests for the manual_override → FILL_GAPS routing path.  When a paper has manual` --uses--> `ProcessingBucket`  [INFERRED]
  tests/test_router_protected_fill_gaps.py → src/ndif_citations/router.py
- `Schema and behavioral tests for the GitHub pipeline revamp (2026-05-20).` --uses--> `DiscoveredRepo`  [INFERRED]
  tests/test_output_schema.py → src/ndif_citations/models.py
- `Belt-and-suspenders guard: if a real pipeline output exists, every     entry in` --uses--> `DiscoveredRepo`  [INFERRED]
  tests/test_output_schema.py → src/ndif_citations/models.py
- `Tests for _merge_paper_data and deduplicate_papers (discover.py).` --uses--> `DiscoverySource`  [INFERRED]
  tests/test_discover_dedup.py → src/ndif_citations/models.py
- `Tests for the repos API — Task 4.4.  Covers:   GET  /api/repos` --uses--> `DiscoveredRepo`  [INFERRED]
  tests/test_api_repos.py → src/ndif_citations/models.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.03
Nodes (333): Apply the merger decisions to research-papers-full.json.  Reads:   output/resear, Parse a manual Date like '8/26/2025' -> (year, iso-date or None)., add(), cli(), demote(), discard(), discover(), edit() (+325 more)

### Community 1 - "Community 1"
Cohesion: 0.03
Nodes (63): make_paper(), deduplicate_papers(), _merge_paper_data(), _MockCompletion, MockLLMClient, Mock LLM client that duck-types the OpenAI SDK surface used by classify_category, Deterministic LLM mock for classify_category / generate_summary tests., Queue a canned reply for the next create() call. Returns self for chaining. (+55 more)

### Community 2 - "Community 2"
Cohesion: 0.03
Nodes (174): create_app(), FastAPI application factory for ndif-citations.  Usage (uvicorn) ---------------, Build and return the configured FastAPI application.      * Mounts the ``/runs``, BaseModel, get_runner(), FastAPI dependencies for ndif-citations server.  Module-level singletons (``_run, Return the module-level JobRunner instance.      Override via ``app.dependency_o, Dependency guard: raise 409 if a run is already in progress.      Intended for m (+166 more)

### Community 3 - "Community 3"
Cohesion: 0.02
Nodes (150): run(), _apply_settings(), Configuration: environment variables, constants, and seed paper IDs., # NOTE: "workshop" intentionally omitted per user decision 2026-05-20 —, # NOTE: " course " (space-padded) catches "ML course 2025 project" without, Rebuild KNOWN_VENUES from _VENUES_FILE (data/known_venues.json)., Apply *s* (from settings_store.load) onto this module's globals., Re-read .env secrets and re-apply settings.json overrides.      Safe to call at (+142 more)

### Community 4 - "Community 4"
Cohesion: 0.02
Nodes (91): debug(), discover_openalex(), discover_s2_citations(), discover_scholar(), enrich_repos_from_github_api(), filter_by_min_year(), _openalex_search(), _openalex_work_to_discovered() (+83 more)

### Community 5 - "Community 5"
Cohesion: 0.04
Nodes (26): _apply_prefilters(), _augment_prompt_with_tier(), _check_discard_zero_pdf_hits(), _compute_confidence_band(), _context_is_comparison_table(), extract_thumbnail(), get_layout_predictor_args(), _get_llm_client() (+18 more)

### Community 6 - "Community 6"
Cohesion: 0.04
Nodes (77): REST router for paper thumbnail images — ``/api/images/{slug}``.  Endpoints ----, Serve ``out/images/{slug}`` as a PNG.      Returns 404 if the file does not exis, serve_image(), resolve(), apply(), _backup(), _configured_target(), detect_target() (+69 more)

### Community 7 - "Community 7"
Cohesion: 0.04
Nodes (34): _make_fitz_doc(), _mock_fitz(), Tests for affiliation-extraction helpers (utils.py).  Each Bug A/B/C/D/E comment, Build a fitz.open() mock returning page0_text from doc[0].get_text()., Inject a fitz mock into sys.modules (fitz is imported locally inside functions)., ACL/EMNLP inline block (Strategy 3) succeeds., Bug C: footnote block detected but yields no entries → falls through to inline., TestAffilClean (+26 more)

### Community 8 - "Community 8"
Cohesion: 0.05
Nodes (49): str, isolated_settings(), _make_client(), _make_target(), TDD tests for the publish REST endpoints — Task 5.1.  GET  /api/publish/target, dry_run=true returns the diff and writes nothing., Omitting dry_run defaults to a dry run (read-only)., dry_run=false applies and returns summary + diff + build_hint. (+41 more)

### Community 9 - "Community 9"
Cohesion: 0.06
Nodes (30): _best_url(), detect_peer_review(), detect_venue_type(), _enrich_affiliations_from_openalex(), enrich_papers(), _extract_affiliations_from_authorships(), _openalex_fetch_work(), Phase 2: Metadata extraction and enrichment. (+22 more)

### Community 10 - "Community 10"
Cohesion: 0.05
Nodes (32): Unit tests for the unified canonical schema in data/known_venues.json.  Verifies, has_known_venue_token must still recognize venues that previously matched., TestDerivedAccessors, TestEndToEndRecognition, TestSchemaShape, Unit tests for src/ndif_citations/venue.py — venue resolution + normalization., Each input is from the actual mess survey of research-papers-full.json., test_doi_decodes_correctly() (+24 more)

### Community 11 - "Community 11"
Cohesion: 0.1
Nodes (14): fixture_state(), make_repo(), no_sleep(), link_repos_to_papers(), _tag_repo_type(), _unlink_shared_template_papers(), route_repos(), Tests for _unlink_shared_template_papers and _tag_repo_type (discover.py). (+6 more)

### Community 12 - "Community 12"
Cohesion: 0.08
Nodes (12): _arxiv_id_year(), _detect_linked_paper(), Tests for _detect_linked_paper and _arxiv_id_year (discover.py)., TestArxivIdYear, TestDetectLinkedPaper, Tests for README parsing utilities (utils.py)., TestExtractBibtexArxivIds, TestParseReadmeSections (+4 more)

### Community 13 - "Community 13"
Cohesion: 0.06
Nodes (15): client(), client_with_tmp_settings(), _FakeActiveRunner, Tests for the repos API — Task 4.4.  Covers:   GET  /api/repos, Exclude a repo: verify DB removal and settings update., Excluding a repo not in the DB still adds it to excluded list, returns was_prese, Stub runner that always reports active=True., _tag_repo_type must return the existing repo_type when manual_override=True. (+7 more)

### Community 14 - "Community 14"
Cohesion: 0.07
Nodes (6): client(), _make_minimal_png(), Tests for the papers read API — Task 4.1.  Covers:   GET /api/papers, Return a TestClient with get_output_dir overridden to fixture_state., Write a tiny valid PNG file to *path* (1x1 red pixel)., test_serve_image_returns_png()

### Community 15 - "Community 15"
Cohesion: 0.08
Nodes (21): client(), client_active_run(), Tests for the papers mutation API — Task 4.2.  Covers:   PATCH /api/papers/{pape, Setting a description sets has_summary=True., year expects an integer; passing 'NaN' should return 422., POST bucket=verified on a pending paper → 200, bucket=='verified', manual_overri, POST bucket=discarded with reason and detail → 200; all fields set., PATCH is blocked with 409 when a pipeline run is active. (+13 more)

### Community 16 - "Community 16"
Cohesion: 0.15
Nodes (5): _fallback_classification(), _fallback_summary(), Tests for _fallback_classification and _fallback_summary (process.py)., TestFallbackClassification, TestFallbackSummary

### Community 17 - "Community 17"
Cohesion: 0.15
Nodes (13): get_field(), _parse_bucket(), _parse_category(), _parse_reason(), edit_paper(), Smoke tests for the EDITABLE_FIELDS schema., test_bool_parser_handles_yes_no(), test_bucket_parses_enum() (+5 more)

### Community 18 - "Community 18"
Cohesion: 0.16
Nodes (5): _detect_venue_type(), _is_venue_upgrade(), Tests for route_papers, route_repos and helpers (router.py)., TestDetectVenueType, TestIsVenueUpgrade

### Community 19 - "Community 19"
Cohesion: 0.16
Nodes (7): Tests for slugify, is_duplicate, and generate_bibtex (utils.py)., TestGenerateBibtex, TestIsDuplicate, generate_bibtex(), is_duplicate(), Check if two titles are similar enough to be considered duplicates., Generate a BibTeX entry from structured metadata.

### Community 20 - "Community 20"
Cohesion: 0.14
Nodes (9): _FakeTimer, Tests for the `ndif-citations serve` CLI command (Task 2.6).  Patching strategy, With --no-open, no Timer is created at all., Minimal threading.Timer stand-in: records args, never starts a thread., serve --no-open --port 9999 → uvicorn.run called with the right args., serve --no-open --host 127.0.0.1 --port 8001 → correct host/port forwarded., Without --no-open, a Timer is scheduled with the correct URL.          We patch, reset() (+1 more)

### Community 21 - "Community 21"
Cohesion: 0.2
Nodes (3): _select_classification_prompt(), Tests for _select_classification_prompt (US-D7)., TestSelectClassificationPrompt

### Community 22 - "Community 22"
Cohesion: 0.26
Nodes (11): arxiv_from_url(), doi_from_url(), first3(), manual_venue_is_real(), match_to_json(), norm_title(), Authoritative merger analysis.  Loads:   - output/research-papers-full.json (JSO, A manual venue counts as 'real' info only if it's not blank/ArXiv/preprint. (+3 more)

### Community 23 - "Community 23"
Cohesion: 0.18
Nodes (11): edit_repo(), exclude_repo(), get_repo(), list_repos(), REST router for repos — ``/api/repos``.  Endpoints --------- GET   /api/repos, Exclude a repo: add to ``excluded_github_repos`` settings and remove from DB., Body for PATCH /api/repos/{owner}/{repo}., Return all repos matching the given filters.      Query parameters     --------- (+3 more)

### Community 24 - "Community 24"
Cohesion: 0.33
Nodes (8): arxiv_from_url(), decision_key(), manual_for_paper(), manual_row_to_paper(), norm_title(), paper_key(), parse_date(), pm_row_to_paper()

### Community 25 - "Community 25"
Cohesion: 0.48
Nodes (2): _build_low_confidence_list(), TestLowConfidenceDedup

### Community 26 - "Community 26"
Cohesion: 0.33
Nodes (1): Validates that every *.txt fixture has a matching *.expected.json.

### Community 27 - "Community 27"
Cohesion: 0.4
Nodes (1): NDIF Citation Tracking Pipeline.

### Community 29 - "Community 29"
Cohesion: 1.0
Nodes (1): Allow running as: python -m ndif_citations

### Community 32 - "Community 32"
Cohesion: 1.0
Nodes (1): True iff repo_type is 'course'. The slim JSON exposes only this boolean.

## Knowledge Gaps
- **223 isolated node(s):** `Tests for README parsing utilities (utils.py).`, `Tests for extract_ndif_context (utils.py).`, `Write text to a temp file and return its Path (won't be auto-deleted).`, `Tests for the papers read API — Task 4.1.  Covers:   GET /api/papers`, `Return a TestClient with get_output_dir overridden to fixture_state.` (+218 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 25`** (7 nodes): `_build_low_confidence_list()`, `TestLowConfidenceDedup`, `.test_high_confidence_excluded()`, `.test_mixed_list()`, `.test_non_unclassified_low_confidence_included()`, `.test_unclassified_excluded_from_low_confidence()`, `test_output_report.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 26`** (6 nodes): `Validates that every *.txt fixture has a matching *.expected.json.`, `test_every_txt_has_expected_json()`, `test_expected_json_schema()`, `test_fixtures_are_not_empty()`, `test_required_scenarios_present()`, `test_fixtures_loader.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 27`** (5 nodes): `NDIF Citation Tracking Pipeline.`, `__init__.py`, `__init__.py`, `__init__.py`, `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 29`** (2 nodes): `Allow running as: python -m ndif_citations`, `__main__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 32`** (1 nodes): `True iff repo_type is 'course'. The slim JSON exposes only this boolean.`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `DiscoveredPaper` connect `Community 0` to `Community 1`, `Community 2`, `Community 3`, `Community 4`, `Community 5`, `Community 8`, `Community 9`, `Community 24`, `Community 25`?**
  _High betweenness centrality (0.145) - this node is a cross-community bridge._
- **Why does `make_paper()` connect `Community 1` to `Community 0`, `Community 3`, `Community 4`, `Community 5`, `Community 8`, `Community 11`, `Community 25`?**
  _High betweenness centrality (0.116) - this node is a cross-community bridge._
- **Why does `Category` connect `Community 0` to `Community 1`, `Community 2`, `Community 3`, `Community 5`, `Community 8`, `Community 9`, `Community 11`, `Community 16`, `Community 17`, `Community 24`, `Community 25`?**
  _High betweenness centrality (0.112) - this node is a cross-community bridge._
- **Are the 262 inferred relationships involving `DiscoveredPaper` (e.g. with `TestUpdateExisting` and `TestMergePapers`) actually correct?**
  _`DiscoveredPaper` has 262 INFERRED edges - model-reasoned connections that need verification._
- **Are the 260 inferred relationships involving `Category` (e.g. with `TestClassifyBandPath` and `Integration tests: classify_category emits the correct Confidence band across al`) actually correct?**
  _`Category` has 260 INFERRED edges - model-reasoned connections that need verification._
- **Are the 220 inferred relationships involving `make_paper()` (e.g. with `.test_no_pdf_no_keywords_in_abstract_is_none()` and `.test_no_pdf_no_abstract_is_none()`) actually correct?**
  _`make_paper()` has 220 INFERRED edges - model-reasoned connections that need verification._
- **Are the 190 inferred relationships involving `Bucket` (e.g. with `TestUpdateExisting` and `TestMergePapers`) actually correct?**
  _`Bucket` has 190 INFERRED edges - model-reasoned connections that need verification._