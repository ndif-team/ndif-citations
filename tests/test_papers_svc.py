"""Tests for ndif_citations.server.services.papers_svc."""


def test_set_bucket_deletes_manual_add_on_discard(tmp_path):
    from ndif_citations.models import DiscoverySource, Bucket, PipelineRun
    from ndif_citations.output import write_outputs, load_existing_papers
    from ndif_citations.server.services import papers_svc
    from tests.conftest import make_paper

    p = make_paper(title="Manual Throwaway", arxiv_id=None,
                   source=DiscoverySource.MANUAL_ADD, bucket=Bucket.PENDING)
    write_outputs([p], tmp_path, PipelineRun())

    res = papers_svc.set_bucket(tmp_path, p.merge_key(), "discarded", "manual_discard", None)
    assert res == {"deleted": True, "merge_key": p.merge_key()}
    assert load_existing_papers(tmp_path) == []  # fully removed, not parked


def test_set_bucket_keeps_nonmanual_on_discard(tmp_path):
    from ndif_citations.models import DiscoverySource, Bucket, PipelineRun
    from ndif_citations.output import write_outputs, load_existing_papers
    from ndif_citations.server.services import papers_svc
    from tests.conftest import make_paper

    p = make_paper(title="Pipeline Paper", arxiv_id=None,
                   source=DiscoverySource.S2_CITATION, bucket=Bucket.PENDING)
    write_outputs([p], tmp_path, PipelineRun())

    res = papers_svc.set_bucket(tmp_path, p.merge_key(), "discarded", "manual_discard", None)
    assert "deleted" not in res                       # returns to_full_dict, not a delete marker
    papers = load_existing_papers(tmp_path)
    assert len(papers) == 1 and papers[0].bucket == Bucket.DISCARDED
