from pulse.core.models import (
    EmbeddingDocument,
    EmbeddingsResponse,
    UsageRecord,
    UsageReport,
)


def test_usage_helpers():
    usage = UsageReport(total=2, records=[UsageRecord(feature="embeddings", units=2)])
    resp = EmbeddingsResponse(
        embeddings=[EmbeddingDocument(text="a", vector=[1.0])],
        requestId="r1",
        usage=usage,
    )
    assert resp.usage_total == 2
    records = resp.usage_records_by_feature()
    assert "embeddings" in records
    assert records["embeddings"].units == 2
