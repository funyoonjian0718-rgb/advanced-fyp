from pathlib import Path

from app.core.config import settings
from app.services.evidence import Evidence
from app.services import vector_store


class FakeEmbeddingModel:
    def get_sentence_embedding_dimension(self) -> int:
        return 3


def test_local_qdrant_indexes_searches_and_replaces_source(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(settings, "qdrant_path", tmp_path / "qdrant")
    monkeypatch.setattr(vector_store, "get_embedding_model", FakeEmbeddingModel)
    monkeypatch.setattr(
        vector_store,
        "embed_texts",
        lambda texts: [[1.0, 0.0, 0.0] for _ in texts],
    )
    vector_store.get_vector_client.cache_clear()

    try:
        first = Evidence(
            evidence_id="chunk-1",
            text="Dietary guideline evidence.",
            source_id=1,
            source_title="Guideline",
            source_url="https://example.gov.my/guideline",
            source_type="government_guideline",
            authority_score=0.9,
            kind="document",
        )
        replacement = Evidence(
            evidence_id="chunk-2",
            text="Updated dietary guideline evidence.",
            source_id=1,
            source_title="Guideline",
            source_url="https://example.gov.my/guideline",
            source_type="government_guideline",
            authority_score=0.9,
            kind="document",
        )

        vector_store.replace_source_vectors(1, [first])
        assert vector_store.search_dense("dietary guideline", 5)[0][0].evidence_id == "chunk-1"

        vector_store.replace_source_vectors(1, [replacement])
        results = vector_store.search_dense("dietary guideline", 5)
        assert [item.evidence_id for item, _ in results] == ["chunk-2"]
    finally:
        vector_store.get_vector_client().close()
        vector_store.get_vector_client.cache_clear()
