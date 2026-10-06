import json
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.core.database import Base
from app.models.knowledge import DocumentChunk, FoodNutrition, KnowledgeSource
from app.services import ingestion


def test_nutrition_ingestion_keeps_source_quality_and_updates_idempotently(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset = tmp_path / "food.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "tempeh-1",
                    "name": "Tempeh",
                    "serving_size": "100 g",
                    "protein": 20.3,
                    "source_type": "generated_estimate",
                    "source_url": "https://myfcd.moh.gov.my/",
                }
            ]
        ),
        encoding="utf-8",
    )
    indexed: list[tuple[int, list]] = []
    monkeypatch.setattr(
        ingestion,
        "replace_source_vectors",
        lambda source_id, evidence: indexed.append((source_id, evidence)),
    )
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)

    try:
        with Session(engine) as db:
            assert ingestion.ingest_nutrition_dataset(db, dataset) == 1
            assert ingestion.ingest_nutrition_dataset(db, dataset) == 1
            foods = db.scalars(select(FoodNutrition)).all()
            sources = db.scalars(select(KnowledgeSource)).all()

            assert len(foods) == 1
            assert foods[0].protein_g == 20.3
            assert foods[0].allergens is None
            assert foods[0].dietary_labels is None
            assert len(sources) == 1
            assert sources[0].authority_score == 0.2
            assert len(indexed) == 2
    finally:
        engine.dispose()


def test_document_chunks_obey_size_limit_and_keep_overlap() -> None:
    source = "important dietary evidence. " * 100

    chunks = ingestion._chunk_text(source)

    assert len(chunks) > 1
    assert all(len(chunk) <= ingestion.CHUNK_SIZE for chunk in chunks)
    assert chunks[0][-60:] in chunks[1]


def test_user_uploaded_records_keep_low_authority_even_with_homepage_url() -> None:
    assert (
        ingestion._source_authority(
            "user_uploaded_csv",
            "https://myfcd.moh.gov.my/",
        )
        == 0.3
    )


def test_guideline_snippet_is_ingested_as_document_not_zero_nutrition_food(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset = tmp_path / "guidelines.json"
    dataset.write_text(
        json.dumps(
            [
                {
                    "id": "guideline-1",
                    "name": "Fruit guidance",
                    "category": "Guideline",
                    "notes": "Eat whole fruit rather than juice.",
                    "calories": 0,
                    "protein": 0,
                    "source_type": "guideline_snippet",
                    "source_url": "user uploaded knowledge base",
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        ingestion,
        "replace_source_vectors",
        lambda source_id, evidence: None,
    )
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)

    try:
        with Session(engine) as db:
            ingestion.ingest_nutrition_dataset(db, dataset)

            assert db.scalars(select(FoodNutrition)).all() == []
            chunks = db.scalars(select(DocumentChunk)).all()
            assert len(chunks) == 1
            assert "whole fruit" in chunks[0].content
    finally:
        engine.dispose()
