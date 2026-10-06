import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from pypdf import PdfReader
from sqlalchemy import delete, select
from sqlalchemy.orm import Session, joinedload

from app.core.config import BASE_DIR
from app.models.knowledge import DocumentChunk, FoodNutrition, KnowledgeSource
from app.services.evidence import chunk_evidence, food_evidence
from app.services.vector_store import replace_source_vectors

DEFAULT_NUTRITION_DATASET = (
    BASE_DIR / "database" / "malaysian_food_rag_dataset_EXPANDED_full.json"
)
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, list):
        value = ", ".join(str(item) for item in value)
    result = str(value).strip()
    return result or None


def _as_float(value: Any, *, field: str, record_id: str) -> float | None:
    if value is None or str(value).strip().casefold() in {
        "",
        "n/a",
        "na",
        "none",
        "null",
        "-",
    }:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Record {record_id!r} has invalid numeric field {field!r}: {value!r}"
        ) from error
    if number < 0:
        raise ValueError(
            f"Record {record_id!r} has negative numeric field {field!r}: {value!r}"
        )
    return number


def _as_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        items = [str(item).strip() for item in value if str(item).strip()]
        return items
    text = str(value).strip()
    if not text:
        return None
    return [item.strip() for item in re.split(r"[,;|]", text) if item.strip()]


def _valid_source_url(value: Any) -> str | None:
    text = _as_text(value)
    if not text:
        return None
    parsed = urlparse(text)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        return None
    return text[:1000]


def _source_authority(source_type: str, source_url: str | None) -> float:
    normalized_type = source_type.casefold()
    if normalized_type.startswith("generated_"):
        return 0.2
    if "user_uploaded" in normalized_type or "user upload" in normalized_type:
        return 0.3
    if "guideline" in normalized_type or "government" in normalized_type:
        return 0.7 if source_url else 0.4
    if source_url and urlparse(source_url).hostname == "myfcd.moh.gov.my":
        return 0.65
    return 0.4


def _read_nutrition_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Nutrition dataset was not found: {path}")
    suffix = path.suffix.casefold()
    if suffix == ".json":
        with path.open("r", encoding="utf-8-sig") as dataset_file:
            records = json.load(dataset_file)
        if not isinstance(records, list) or not all(
            isinstance(record, dict) for record in records
        ):
            raise ValueError("Nutrition JSON must contain an array of objects.")
        return records
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as dataset_file:
            return [dict(row) for row in csv.DictReader(dataset_file)]
    raise ValueError("Nutrition dataset must be a .csv or .json file.")


def _get_or_create_source(
    db: Session,
    *,
    title: str,
    source_type: str,
    source_url: str | None,
    authority_score: float,
) -> KnowledgeSource:
    source = db.scalar(
        select(KnowledgeSource).where(
            KnowledgeSource.title == title,
            KnowledgeSource.source_type == source_type,
            KnowledgeSource.source_url == source_url,
        )
    )
    if source is None:
        source = KnowledgeSource(
            title=title,
            source_type=source_type,
            source_url=source_url,
            authority_score=authority_score,
        )
        db.add(source)
        db.flush()
    else:
        source.authority_score = authority_score
    return source


def ingest_nutrition_dataset(db: Session, path: Path) -> int:
    records = _read_nutrition_records(path)
    groups: dict[tuple[str, str | None], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        source_type = _as_text(record.get("source_type")) or "user_uploaded_dataset"
        source_url = _valid_source_url(record.get("source_url"))
        groups[(source_type, source_url)].append(record)

    indexed_source_ids: list[int] = []
    for (source_type, source_url), source_records in groups.items():
        source = _get_or_create_source(
            db,
            title=f"{path.stem} ({source_type})",
            source_type=source_type,
            source_url=source_url,
            authority_score=_source_authority(source_type, source_url),
        )
        indexed_source_ids.append(source.id)
        current_food_ids: set[str] = set()
        seen_record_ids: set[str] = set()
        db.execute(
            delete(DocumentChunk).where(DocumentChunk.source_id == source.id)
        )
        db.flush()
        chunk_index = 0

        for row_number, record in enumerate(source_records, start=1):
            record_id = _as_text(record.get("id")) or str(row_number)
            if record_id in seen_record_ids:
                raise ValueError(
                    f"Duplicate record id {record_id!r} "
                    f"for source type {source_type!r}."
                )
            seen_record_ids.add(record_id)
            category = _as_text(record.get("category")) or ""
            name = _as_text(record.get("name"))

            if (
                "guideline" in source_type.casefold()
                or category.casefold() == "guideline"
            ):
                if not name:
                    raise ValueError(
                        f"Guideline record {record_id!r} is missing its title."
                    )
                document_text = "\n".join(
                    value
                    for value in (
                        name,
                        _as_text(record.get("aliases")),
                        _as_text(record.get("notes")),
                    )
                    if value
                )
                if not document_text.strip():
                    document_text = _as_text(record.get("rag_text")) or ""
                if not document_text.strip():
                    raise ValueError(
                        f"Guideline record {record_id!r} contains no text."
                    )
                for piece in _chunk_text(document_text):
                    db.add(
                        DocumentChunk(
                            source_id=source.id,
                            chunk_index=chunk_index,
                            content=piece,
                            extra_data={
                                "record_id": record_id,
                                "category": category,
                            },
                        )
                    )
                    chunk_index += 1
                continue

            current_food_ids.add(record_id)
            food = db.scalar(
                select(FoodNutrition).where(
                    FoodNutrition.source_id == source.id,
                    FoodNutrition.source_record_id == record_id,
                )
            )
            if food is None:
                food = FoodNutrition(
                    source_id=source.id,
                    source_record_id=record_id,
                    name="",
                )
                db.add(food)

            if not name:
                raise ValueError(
                    f"Nutrition record {record_id!r} is missing its food name."
                )
            food.name = name
            food.aliases = _as_text(record.get("aliases"))
            food.category = _as_text(record.get("category"))
            food.serving_size = _as_text(record.get("serving_size"))
            food.ingredients = _as_text(record.get("ingredients"))
            food.allergens = _as_list(record.get("allergens"))
            food.dietary_labels = _as_list(
                record.get("dietary_labels", record.get("dietary_preference"))
            )
            food.calories = _as_float(
                record.get("calories"), field="calories", record_id=record_id
            )
            food.carbohydrates_g = _as_float(
                record.get("carbohydrates"),
                field="carbohydrates",
                record_id=record_id,
            )
            food.protein_g = _as_float(
                record.get("protein"), field="protein", record_id=record_id
            )
            food.fat_g = _as_float(
                record.get("fat"), field="fat", record_id=record_id
            )
            food.fiber_g = _as_float(
                record.get("fiber", record.get("fibre")),
                field="fiber",
                record_id=record_id,
            )
            food.sugar_g = _as_float(
                record.get("sugar"), field="sugar", record_id=record_id
            )
            food.sodium_mg = _as_float(
                record.get("sodium"), field="sodium", record_id=record_id
            )
            food.potassium_mg = _as_float(
                record.get("potassium"), field="potassium", record_id=record_id
            )
            food.glycemic_index = _as_float(
                record.get("glycemic_index"),
                field="glycemic_index",
                record_id=record_id,
            )
            food.diabetes_rating = _as_text(record.get("diabetes_rating"))
            food.health_tags = _as_text(record.get("health_tags"))
            food.notes = _as_text(record.get("notes")) or _as_text(
                record.get("rag_text")
            )

        old_records = db.scalars(
            select(FoodNutrition).where(
                FoodNutrition.source_id == source.id,
                FoodNutrition.source_record_id.not_in(current_food_ids),
            )
        ).all()
        for old_record in old_records:
            db.delete(old_record)

    db.flush()
    evidence_by_source: dict[int, list] = {
        source_id: [] for source_id in indexed_source_ids
    }
    for chunk in db.scalars(
        select(DocumentChunk)
        .options(joinedload(DocumentChunk.source))
        .where(DocumentChunk.source_id.in_(indexed_source_ids))
        .order_by(DocumentChunk.id)
    ).all():
        evidence_by_source[chunk.source_id].append(chunk_evidence(chunk))
    for food in db.scalars(
        select(FoodNutrition)
        .options(joinedload(FoodNutrition.source))
        .where(FoodNutrition.source_id.in_(indexed_source_ids))
        .order_by(FoodNutrition.id)
    ).all():
        evidence_by_source[food.source_id].append(food_evidence(food))

    db.commit()

    for source_id in indexed_source_ids:
        replace_source_vectors(source_id, evidence_by_source[source_id])
    return len(records)


def _read_document(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(f"Knowledge document was not found: {path}")
    suffix = path.suffix.casefold()
    if suffix == ".pdf":
        return "\n".join(page.extract_text() or "" for page in PdfReader(path).pages)
    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8-sig")
    raise ValueError("Knowledge documents must be .pdf, .txt, or .md files.")


def _chunk_text(text: str) -> list[str]:
    normalized = re.sub(r"[ \t]+", " ", text)
    paragraphs = [
        re.sub(r"\s+", " ", part).strip()
        for part in re.split(r"\n\s*\n", normalized)
        if part.strip()
    ]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        while len(paragraph) > CHUNK_SIZE:
            if current:
                chunks.append(current)
                current = ""
            chunks.append(paragraph[:CHUNK_SIZE].strip())
            paragraph = paragraph[CHUNK_SIZE - CHUNK_OVERLAP :].strip()
        candidate = f"{current}\n{paragraph}".strip() if current else paragraph
        if len(candidate) > CHUNK_SIZE and current:
            chunks.append(current)
            overlap = current[-CHUNK_OVERLAP:]
            current = f"{overlap} {paragraph}".strip()
        else:
            current = candidate
    if current:
        chunks.append(current)
    return chunks


def ingest_document(
    db: Session,
    path: Path,
    *,
    source_type: str,
    source_url: str | None = None,
    authority_score: float | None = None,
) -> int:
    if not 0 <= (authority_score if authority_score is not None else 0) <= 1:
        raise ValueError("Source authority must be between 0 and 1.")
    text = _read_document(path).strip()
    if not text:
        raise ValueError(f"Knowledge document contains no extractable text: {path}")
    pieces = _chunk_text(text)
    url = _valid_source_url(source_url)
    authority = (
        authority_score
        if authority_score is not None
        else _source_authority(source_type, url)
    )
    source = _get_or_create_source(
        db,
        title=path.name,
        source_type=source_type,
        source_url=url,
        authority_score=authority,
    )
    db.execute(delete(DocumentChunk).where(DocumentChunk.source_id == source.id))
    db.flush()
    chunks = [
        DocumentChunk(
            source_id=source.id,
            chunk_index=index,
            content=piece,
            extra_data={"page_or_section": index + 1},
        )
        for index, piece in enumerate(pieces)
    ]
    db.add_all(chunks)
    db.flush()
    evidence = [chunk_evidence(chunk) for chunk in chunks]
    db.commit()
    replace_source_vectors(source.id, evidence)
    return len(chunks)


def ingest_default_dataset(db: Session) -> int:
    return ingest_nutrition_dataset(db, DEFAULT_NUTRITION_DATASET)
