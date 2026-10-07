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

from app.core.config import settings
from app.models.knowledge import DocumentChunk, FoodNutrition, KnowledgeSource
from app.services.evidence import chunk_evidence
from app.services.vector_store import replace_source_vectors

DEFAULT_RAG_CHUNKS = settings.rag_chunks_path
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 180
LEGACY_DEFAULT_DATASET_STEM = "malaysian_food_rag_dataset_EXPANDED_full"
FIELD_ALIASES = {
    "id": ("record_id", "food_id", "uuid"),
    "name": ("food_name", "food", "item", "title"),
    "aliases": ("alternate_names", "alternative_names", "synonyms"),
    "category": ("food_category", "food_group"),
    "serving_size": ("serving", "portion", "portion_size", "serving_description"),
    "ingredients": ("ingredient_list",),
    "allergens": ("allergen", "allergen_list", "allergy_information"),
    "dietary_labels": (
        "dietary_preference",
        "diet_labels",
        "dietary",
        "diet",
    ),
    "calories": ("energy_kcal", "calories_kcal", "kcal"),
    "carbohydrates": (
        "carbs",
        "carbohydrate",
        "carbohydrates_g",
        "carbohydrate_g",
        "carbs_g",
    ),
    "protein": ("protein_g",),
    "fat": ("fat_g",),
    "fiber": ("fibre", "fiber_g", "fibre_g"),
    "sugar": ("sugar_g",),
    "sodium": ("sodium_mg",),
    "potassium": ("potassium_mg",),
    "glycemic_index": ("glycaemic_index", "gi"),
    "diabetes_rating": ("diabetes_suitability",),
    "health_tags": ("tags", "health_labels"),
    "notes": ("description", "remarks", "advice"),
    "rag_text": ("text", "content"),
    "source_type": ("provenance", "provenance_type", "record_source_type"),
    "source_url": ("source_link", "reference_url", "url"),
}


def _canonicalize_record(record: dict[str, Any]) -> dict[str, Any]:
    normalized = {
        re.sub(r"[^a-z0-9]", "", str(key).casefold()): value
        for key, value in record.items()
    }
    for nested_key in ("nutrition", "nutrients", "nutrition_facts"):
        nutrition = record.get(nested_key)
        if isinstance(nutrition, dict):
            normalized = {
                **{
                    re.sub(r"[^a-z0-9]", "", str(key).casefold()): value
                    for key, value in nutrition.items()
                },
                **normalized,
            }

    result = dict(record)
    for canonical, aliases in FIELD_ALIASES.items():
        if canonical in result and result[canonical] is not None:
            continue
        for key in (canonical, *aliases):
            normalized_key = re.sub(r"[^a-z0-9]", "", key.casefold())
            value = normalized.get(normalized_key)
            if value is not None:
                result[canonical] = value
                break
    return result


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


def _read_food_records(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Food dataset was not found: {path}")
    suffix = path.suffix.casefold()
    if suffix == ".json":
        with path.open("r", encoding="utf-8-sig") as dataset_file:
            records = json.load(dataset_file)
        if isinstance(records, dict):
            records = records.get("records", records.get("data"))
        if not isinstance(records, list) or not all(
            isinstance(record, dict) for record in records
        ):
            raise ValueError("Food JSON must contain an array of objects.")
        if not records:
            raise ValueError("Food dataset must contain at least one record.")
        return [_canonicalize_record(record) for record in records]
    if suffix == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as dataset_file:
            records = [
                _canonicalize_record(dict(row))
                for row in csv.DictReader(dataset_file)
            ]
        if not records:
            raise ValueError("Food dataset must contain at least one record.")
        return records
    raise ValueError("Food dataset must be a .csv or .json file.")


def _read_rag_chunks(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"RAG chunk dataset was not found: {path}")
    if path.suffix.casefold() != ".jsonl":
        raise ValueError("RAG chunk dataset must be a .jsonl file.")

    records = []
    with path.open("r", encoding="utf-8-sig") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"RAG JSONL line {line_number} is not valid JSON."
                ) from error
            if not isinstance(record, dict):
                raise ValueError(
                    f"RAG JSONL line {line_number} must contain an object."
                )
            records.append(record)
    if not records:
        raise ValueError("RAG chunk dataset must contain at least one record.")
    return records


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

def ingest_food_dataset(db: Session, path: Path) -> int:
    records = _read_food_records(path)

    # The FoodNutrition table should contain only actual food records.
    # Guideline records belong to the RAG/document evidence pipeline.
    records = [
        record
        for record in records
        if record.get("record_type", "food") == "food"
    ]

    groups: dict[tuple[str, str | None], list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        source_type = _as_text(record.get("source_type")) or "user_uploaded_dataset"
        source_url = _valid_source_url(record.get("source_url"))

        groups[(source_type, source_url)].append(record)

    for (source_type, source_url), source_records in groups.items():
        source = _get_or_create_source(
            db,
            title=f"{path.stem} ({source_type})",
            source_type=source_type,
            source_url=source_url,
            authority_score=_source_authority(source_type, source_url),
        )

        current_food_ids: set[str] = set()

        for row_number, record in enumerate(source_records, start=1):
            record_id = _as_text(record.get("id")) or str(row_number)

            if record_id in current_food_ids:
                raise ValueError(
                    f"Duplicate food record id {record_id!r} "
                    f"for source type {source_type!r}."
                )

            current_food_ids.add(record_id)

            if (
                "guideline" in source_type.casefold()
                or (
                    (_as_text(record.get("category")) or "").casefold()
                    == "guideline"
                )
            ):
                raise ValueError(
                    "Structured food datasets cannot contain guideline records."
                )

            name = _as_text(record.get("name"))

            if not name:
                raise ValueError(
                    f"Food record {record_id!r} is missing its name."
                )

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
                    name=name,
                )
                db.add(food)

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

    db.commit()
    from app.services.retrieval import invalidate_retrieval_cache

    invalidate_retrieval_cache()
    return len(records)


def ingest_rag_chunks(
    db: Session,
    path: Path,
) -> int:
    records = _read_rag_chunks(path)
    grouped: dict[tuple[str, str | None], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        source_type = _as_text(record.get("source_type")) or "user_uploaded_dataset"
        source_url = _valid_source_url(record.get("source_url"))
        grouped[(source_type, source_url)].append(record)

    source_ids: list[int] = []
    for (source_type, source_url), source_records in grouped.items():
        source = _get_or_create_source(
            db,
            title=f"{path.stem} ({source_type})",
            source_type=source_type,
            source_url=source_url,
            authority_score=_source_authority(source_type, source_url),
        )
        source_ids.append(source.id)
        db.execute(delete(DocumentChunk).where(DocumentChunk.source_id == source.id))
        db.flush()

        seen_chunk_ids: set[str] = set()
        for record in source_records:
            chunk_id = _as_text(record.get("chunk_id"))
            parent_id = _as_text(record.get("parent_id"))
            content = _as_text(record.get("text"))
            if not chunk_id or not parent_id or not content:
                raise ValueError(
                    "Every multichunk JSONL record must include chunk_id, "
                    "parent_id, and non-empty text."
                )
            if chunk_id in seen_chunk_ids:
                raise ValueError(
                    f"Duplicate chunk_id {chunk_id!r} in {path.name}."
                )
            seen_chunk_ids.add(chunk_id)
            db.add(
                DocumentChunk(
                    source_id=source.id,
                    chunk_index=len(seen_chunk_ids) - 1,
                    content=content,
                    extra_data={
                        "chunk_id": chunk_id,
                        "parent_id": parent_id,
                        "chunk_type": _as_text(record.get("chunk_type")) or "",
                        "record_type": _as_text(record.get("record_type")) or "",
                        "food_name": _as_text(record.get("food_name")) or "",
                        "evidence_level": _as_text(record.get("evidence_level")) or "",
                    },
                )
            )

    legacy_source_ids: list[int] = []
    if path.resolve() == settings.rag_chunks_path.resolve():
        legacy_source_ids = db.scalars(
            select(KnowledgeSource.id).where(
                KnowledgeSource.title.like(
                    f"{LEGACY_DEFAULT_DATASET_STEM} (%)"
                )
            )
        ).all()

    db.flush()
    evidence_by_source: dict[int, list] = {source_id: [] for source_id in source_ids}
    for chunk in db.scalars(
        select(DocumentChunk)
        .options(joinedload(DocumentChunk.source))
        .where(DocumentChunk.source_id.in_(source_ids))
        .order_by(DocumentChunk.id)
    ).all():
        evidence_by_source[chunk.source_id].append(chunk_evidence(chunk))
    db.commit()
    from app.services.retrieval import invalidate_retrieval_cache

    invalidate_retrieval_cache()
    for source_id in source_ids:
        replace_source_vectors(source_id, evidence_by_source[source_id])
    for source_id in legacy_source_ids:
        replace_source_vectors(source_id, [])
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
    from app.services.retrieval import invalidate_retrieval_cache

    invalidate_retrieval_cache()
    replace_source_vectors(source.id, evidence)
    return len(chunks)


def ingest_default_rag_chunks(db: Session) -> int:
    return ingest_rag_chunks(db, DEFAULT_RAG_CHUNKS)
