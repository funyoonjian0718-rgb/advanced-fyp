import os
from functools import lru_cache
from pathlib import Path

from app.core.config import settings

os.environ.setdefault("HF_HOME", str(settings.model_cache_path))
os.environ.setdefault("SENTENCE_TRANSFORMERS_HOME", str(settings.model_cache_path))


@lru_cache(maxsize=1)
def get_embedding_model():
    from sentence_transformers import SentenceTransformer

    Path(settings.model_cache_path).mkdir(parents=True, exist_ok=True)
    return SentenceTransformer(
        settings.embedding_model_id,
        cache_folder=str(settings.model_cache_path),
        device="cpu",
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    vectors = get_embedding_model().encode(
        texts,
        batch_size=32,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )
    return vectors.tolist()
