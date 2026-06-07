from ndif_citations.models import DiscoveredPaper


def test_enrichment_provenance_defaults_empty_and_roundtrips():
    p = DiscoveredPaper(title="X")
    assert p.enrichment_provenance == {}
    p.enrichment_provenance["abstract"] = "openalex"
    dumped = p.model_dump(mode="json")
    assert dumped["enrichment_provenance"] == {"abstract": "openalex"}
    restored = DiscoveredPaper.model_validate(dumped)
    assert restored.enrichment_provenance == {"abstract": "openalex"}
