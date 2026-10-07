import json
from pathlib import Path

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.core.database import Base
from app.models.knowledge import DocumentChunk, FoodNutrition, KnowledgeSource
from app.services import ingestion


def test_food_ingestion_keeps_source_quality_and_updates_idempotently(
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
            assert ingestion.ingest_food_dataset(db, dataset) == 1
            assert ingestion.ingest_food_dataset(db, dataset) == 1
            foods = db.scalars(select(FoodNutrition)).all()
            sources = db.scalars(select(KnowledgeSource)).all()

            assert len(foods) == 1
            assert foods[0].protein_g == 20.3
            assert foods[0].allergens is None
            assert foods[0].dietary_labels is None
            assert len(sources) == 1
            assert sources[0].authority_score == 0.2
            assert indexed == []
            assert db.scalars(select(DocumentChunk)).all() == []
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


def test_food_csv_ingestion_normalizes_v2_fields(tmp_path: Path) -> None:
    dataset = tmp_path / "foods_v2.csv"
    dataset.write_text(
        "food_id,food_name,serving,protein_g,calories,source_type\n"
        "tempeh-v2,Tempeh,100 g,18,190,user_uploaded_csv\n",
        encoding="utf-8",
    )
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)

    try:
        with Session(engine) as db:
            assert ingestion.ingest_food_dataset(db, dataset) == 1
            food = db.scalar(select(FoodNutrition))
            assert food is not None
            assert food.source_record_id == "tempeh-v2"
            assert food.name == "Tempeh"
            assert food.serving_size == "100 g"
            assert food.protein_g == 18
            assert food.calories == 190
    finally:
        engine.dispose()


def test_rag_chunk_ingestion_creates_documents_without_food_records(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset = tmp_path / "food_multichunk.jsonl"
    records = [
        {
            "chunk_id": "food-1-overview",
            "parent_id": 1,
            "chunk_type": "overview",
            "record_type": "food",
            "food_name": "Nasi Lemak",
            "text": (
                "Nasi Lemak is a Malaysian food item in the Rice Dish category. "
                "The listed serving size is 1 plate. Aliases: none listed. "
                "General notes: Rice dish."
            ),
            "source_type": "user_uploaded_csv",
        },
        {
            "chunk_id": "food-1-nutrition",
            "parent_id": 1,
            "chunk_type": "nutrition",
            "record_type": "food",
            "food_name": "Nasi Lemak",
            "text": (
                "Nasi Lemak nutrition per 1 plate: 520 kcal; carbohydrates "
                "62.0 g; protein 16.0 g; fat 24.0 g; fibre 4.0 g; sugar 6.0 g; "
                "sodium 780 mg; potassium 390 mg; glycemic index about 72."
            ),
            "source_type": "user_uploaded_csv",
        },
        {
            "chunk_id": "guideline-1",
            "parent_id": 2,
            "chunk_type": "guideline",
            "record_type": "guideline",
            "food_name": "Plate guide",
            "text": "Fill half the plate with fruit and vegetables.",
            "source_type": "user_uploaded_csv",
        },
    ]
    dataset.write_text(
        "\n".join(json.dumps(record) for record in records),
        encoding="utf-8",
    )
    indexed: list[list] = []
    monkeypatch.setattr(
        ingestion,
        "replace_source_vectors",
        lambda source_id, evidence: indexed.append(evidence),
    )
    monkeypatch.setattr(ingestion.settings, "rag_chunks_path", dataset)
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)

    try:
        with Session(engine) as db:
            legacy_source = KnowledgeSource(
                title=(
                    f"{ingestion.LEGACY_DEFAULT_DATASET_STEM} "
                    "(user_uploaded_csv)"
                ),
                source_type="user_uploaded_csv",
                authority_score=0.3,
            )
            db.add(legacy_source)
            db.flush()
            db.add(
                FoodNutrition(
                    source_id=legacy_source.id,
                    source_record_id="old-1",
                    name="Old duplicate",
                )
            )
            rag_source = KnowledgeSource(
                title=f"{dataset.stem} (user_uploaded_csv)",
                source_type="user_uploaded_csv",
                authority_score=0.3,
            )
            db.add(rag_source)
            db.flush()
            db.add(
                FoodNutrition(
                    source_id=rag_source.id,
                    source_record_id="previous-derived-food",
                    name="Preserved structured record",
                )
            )
            db.commit()

            assert ingestion.ingest_rag_chunks(db, dataset) == 3
            foods = db.scalars(select(FoodNutrition)).all()
            chunks = db.scalars(select(DocumentChunk)).all()
            sources = db.scalars(select(KnowledgeSource)).all()

            assert {food.name for food in foods} == {
                "Old duplicate",
                "Preserved structured record",
            }
            assert len(chunks) == 3
            assert len(indexed[0]) == 3
            assert indexed[-1] == []
            assert len(sources) == 2

            legacy_source = KnowledgeSource(
                title=(
                    f"{ingestion.LEGACY_DEFAULT_DATASET_STEM} "
                    "(user_uploaded_csv)"
                ),
                source_type="user_uploaded_csv",
                authority_score=0.3,
            )
            db.add(legacy_source)
            db.commit()
            preserved_legacy_id = legacy_source.id

            custom_dataset = tmp_path / "custom_multichunk.jsonl"
            custom_dataset.write_text(
                json.dumps(
                    {
                        "chunk_id": "custom-guideline",
                        "parent_id": 9,
                        "chunk_type": "guideline",
                        "record_type": "guideline",
                        "food_name": "Custom guideline",
                        "text": "A custom dietary guideline.",
                        "source_type": "user_uploaded_csv",
                    }
                ),
                encoding="utf-8",
            )
            assert ingestion.ingest_rag_chunks(db, custom_dataset) == 1
            assert db.get(KnowledgeSource, preserved_legacy_id) is not None
    finally:
        engine.dispose()


def test_guideline_rag_chunk_is_not_ingested_as_food(
    tmp_path: Path,
    monkeypatch,
) -> None:
    dataset = tmp_path / "guidelines.jsonl"
    dataset.write_text(
        json.dumps(
            {
                "chunk_id": "guideline-1",
                "parent_id": "guideline-1",
                "chunk_type": "guideline",
                "record_type": "guideline",
                "food_name": "Fruit guidance",
                "text": "Eat whole fruit rather than juice.",
                "source_type": "guideline_snippet",
            }
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
            ingestion.ingest_rag_chunks(db, dataset)

            assert db.scalars(select(FoodNutrition)).all() == []
            chunks = db.scalars(select(DocumentChunk)).all()
            assert len(chunks) == 1
            assert "whole fruit" in chunks[0].content
    finally:
        engine.dispose()
