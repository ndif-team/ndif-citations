"""Unit tests for the unified canonical schema in data/known_venues.json.

Verifies:
    1. The new schema loads via config.KNOWN_VENUES.
    2. The backwards-compat derived accessors (conferences/journals/
       preprint_servers/acronym_map) are well-formed.
    3. Workshop parent references resolve to existing venue keys.
    4. Each known canonical venue is reachable through every consumer path that
       existing code relies on (has_known_venue_token, detect_venue_type-ish).
"""
from __future__ import annotations

from ndif_citations import config
from ndif_citations.venue import has_known_venue_token


class TestSchemaShape:
    def test_venues_key_present(self):
        assert "venues" in config.KNOWN_VENUES
        assert isinstance(config.KNOWN_VENUES["venues"], dict)
        assert len(config.KNOWN_VENUES["venues"]) > 0

    def test_each_entry_has_type(self):
        for name, entry in config.KNOWN_VENUES["venues"].items():
            assert "type" in entry, f"{name} missing 'type'"
            assert entry["type"] in {"conference", "workshop", "journal", "preprint"}, (
                f"{name} has invalid type {entry['type']!r}"
            )

    def test_workshops_have_existing_parent(self):
        venues = config.KNOWN_VENUES["venues"]
        for name, entry in venues.items():
            if entry["type"] == "workshop":
                parent = entry.get("parent")
                assert parent, f"workshop {name} missing parent"
                assert parent in venues, f"workshop {name} parent {parent} not in venues"
                assert venues[parent]["type"] == "conference", (
                    f"workshop {name} parent {parent} is not a conference"
                )


class TestDerivedAccessors:
    def test_conferences_bundles_workshops(self):
        confs = config.KNOWN_VENUES["conferences"]
        assert "ICML" in confs
        assert "NeurIPS" in confs
        # Workshops bundle in (preserves legacy detect_peer_review behavior).
        assert "BlackboxNLP" in confs
        assert "Findings of EMNLP" in confs

    def test_journals_excludes_conferences(self):
        journals = config.KNOWN_VENUES["journals"]
        assert "JMLR" in journals
        assert "TMLR" in journals
        # Newly-added per migration.
        assert "Pattern Recognition" in journals
        assert "Nature Methods" in journals
        # No conferences leaked in.
        assert "ICML" not in journals
        assert "BlackboxNLP" not in journals

    def test_preprint_servers_holds_canonical_keys(self):
        preprints = config.KNOWN_VENUES["preprint_servers"]
        assert "ArXiv" in preprints
        assert "OpenReview" in preprints
        # Aliases live in acronym_map, NOT preprint_servers.
        assert "arXiv (Cornell University)" not in preprints

    def test_acronym_map_contains_known_long_forms(self):
        am = config.KNOWN_VENUES["acronym_map"]
        assert am["International Conference on Machine Learning"] == "ICML"
        assert am["Neural Information Processing Systems"] == "NeurIPS"
        assert am["Advances in Neural Information Processing Systems"] == "NeurIPS"
        assert am["Conference on Empirical Methods in Natural Language Processing"] == "EMNLP"
        # CPVR is the explicit misspelling alias for CVPR.
        assert am["CPVR"] == "CVPR"


class TestEndToEndRecognition:
    """has_known_venue_token must still recognize venues that previously matched."""

    def test_recognizes_canonical_acronyms(self):
        assert has_known_venue_token("ICML 2025")
        assert has_known_venue_token("NeurIPS 2024")
        assert has_known_venue_token("FAccT 2025")

    def test_recognizes_long_form_via_acronym_map(self):
        assert has_known_venue_token("Annual Meeting of the Association for Computational Linguistics 2025")
        assert has_known_venue_token("International Conference on Machine Learning")

    def test_recognizes_workshops(self):
        assert has_known_venue_token("BlackboxNLP 2025")
        assert has_known_venue_token("Findings of EMNLP 2024")

    def test_recognizes_new_journal_additions(self):
        assert has_known_venue_token("Pattern Recognition")
        assert has_known_venue_token("Nature Methods")

    def test_recognizes_misspelling_alias(self):
        # CPVR is encoded as an explicit alias of CVPR.
        assert has_known_venue_token("CPVR 2024")

    def test_rejects_unknown_venue(self):
        assert not has_known_venue_token("Some Random Venue 2025")
