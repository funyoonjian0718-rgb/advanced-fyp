from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    database_url: str
    secret_key: str
    embedding_model_id: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    reranker_model_id: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    llm_model_id: str = "Qwen/Qwen2.5-0.5B-Instruct"
    llm_adapter_path: Path = BASE_DIR / "backend" / "models" / "dietary-assistant"
    qdrant_path: Path = BASE_DIR / "backend" / "data" / "qdrant"
    model_cache_path: Path = BASE_DIR / "backend" / ".model-cache"
    answer_confidence_threshold: float = 0.45
    max_answer_tokens: int = 512

    model_config = SettingsConfigDict(
        env_file=BASE_DIR / ".env",
        extra="ignore",
    )


settings = Settings()