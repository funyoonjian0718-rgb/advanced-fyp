from pathlib import Path

from app.core.config import settings
from app.services.evidence import Evidence, evidence_from_payload
from app.services import vector_store


class FakeEmbeddingModel:
    def get_sentence_embedding_dimension(self) -> int:
        return 3


def test_old_payload_verified_flags_require_explicit_review_attestation() -> None:
    evidence = evidence_from_payload(
        {
            "evidence_id": "food-1",
            "text": "Food: Lentil patty",
            "source_id": 1,
            "source_title": "Food records",
            "source_url": None,
            "source_type": "user_uploaded_csv",
            "authority_score": 0.3,
            "kind": "food",
            "metadata": {
                "allergens": [],
                "dietary_labels": ["vegetarian"],
                "allergen_data_verified": True,
                "dietary_data_verified": True,
            },
        }
    )

    assert evidence.metadata["allergen_data_verified"] is False
    assert evidence.metadata["dietary_data_verified"] is False


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

        vector_store.reset_collection()
        assert vector_store.search_dense("dietary guideline", 5) == []

        vector_store.replace_source_vectors(1, [first])
        assert vector_store.search_dense("dietary guideline", 5)[0][0].evidence_id == "chunk-1"
    finally:
        vector_store.get_vector_client().close()
        vector_store.get_vector_client.cache_clear()


def test_dense_search_can_filter_by_evidence_kind(
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
        document = Evidence(
            evidence_id="chunk-1",
            text="Guideline content.",
            source_id=1,
            source_title="Guideline",
            source_url=None,
            source_type="guideline",
            authority_score=0.8,
            kind="document",
        )
        food = Evidence(
            evidence_id="food-1",
            text="Nutrition record.",
            source_id=1,
            source_title="Food dataset",
            source_url=None,
            source_type="user_uploaded_csv",
            authority_score=0.3,
            kind="food",
        )
        vector_store.replace_source_vectors(1, [food, document])

        results = vector_store.search_dense("guideline", 5, kind="document")

        assert [item.kind for item, _ in results] == ["document"]
        assert [item.evidence_id for item, _ in results] == ["chunk-1"]
    finally:
        vector_store.get_vector_client().close()
        vector_store.get_vector_client.cache_clear()
