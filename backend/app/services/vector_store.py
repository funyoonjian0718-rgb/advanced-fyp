from functools import lru_cache
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from app.core.config import settings
from app.services.embeddings import embed_texts, get_embedding_model
from app.services.evidence import Evidence, evidence_from_payload, evidence_to_payload

COLLECTION_NAME = "dietary_evidence"


@lru_cache(maxsize=1)
def get_vector_client() -> QdrantClient:
    Path(settings.qdrant_path).mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(settings.qdrant_path))


def close_vector_client() -> None:
    if get_vector_client.cache_info().currsize:
        get_vector_client().close()
        get_vector_client.cache_clear()


def ensure_collection() -> None:
    client = get_vector_client()
    embedding_size = get_embedding_model().get_sentence_embedding_dimension()

    if client.collection_exists(COLLECTION_NAME):
        collection = client.get_collection(COLLECTION_NAME)
        vectors = collection.config.params.vectors
        vector_size = vectors.size if isinstance(vectors, models.VectorParams) else None
        if vector_size != embedding_size:
            raise RuntimeError(
                "The existing Qdrant collection uses a different embedding size. "
                "Rebuild the local vector index after changing EMBEDDING_MODEL_ID."
            )
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=models.VectorParams(
            size=embedding_size,
            distance=models.Distance.COSINE,
        ),
    )


def replace_source_vectors(source_id: int, evidence: list[Evidence]) -> None:
    client = get_vector_client()
    if not client.collection_exists(COLLECTION_NAME):
        ensure_collection()

    client.delete(
        collection_name=COLLECTION_NAME,
        points_selector=models.FilterSelector(
            filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="source_id",
                        match=models.MatchValue(value=source_id),
                    )
                ]
            )
        ),
        wait=True,
    )
    if not evidence:
        return

    vectors = embed_texts([item.text for item in evidence])
    client.upsert(
        collection_name=COLLECTION_NAME,
        points=[
            models.PointStruct(
                id=str(uuid5(NAMESPACE_URL, item.evidence_id)),
                vector=vector,
                payload=evidence_to_payload(item),
            )
            for item, vector in zip(evidence, vectors, strict=True)
        ],
        wait=True,
    )


def search_dense(query: str, limit: int) -> list[tuple[Evidence, float]]:
    client = get_vector_client()
    if not client.collection_exists(COLLECTION_NAME):
        return []
    vector = embed_texts([query])[0]
    result = client.query_points(
        collection_name=COLLECTION_NAME,
        query=vector,
        limit=limit,
        with_payload=True,
    )
    matches: list[tuple[Evidence, float]] = []
    for point in result.points:
        if point.payload is not None:
            matches.append(
                (evidence_from_payload(point.payload), float(point.score))
            )
    return matches
