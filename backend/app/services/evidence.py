from dataclasses import dataclass, field

from app.models.knowledge import DocumentChunk, FoodNutrition, KnowledgeSource


ESTIMATE_SOURCE_TYPES = {
    "generated_estimate",
    "generated_portion_variant",
    "generated_less_sugar_variant",
}


@dataclass
class Evidence:
    evidence_id: str
    text: str
    source_id: int
    source_title: str
    source_url: str | None
    source_type: str
    authority_score: float
    kind: str
    metadata: dict[str, object] = field(default_factory=dict)
    retrieval_score: float = 0.0
    reranker_score: float = 0.0

    @property
    def is_estimate(self) -> bool:
        return self.source_type in ESTIMATE_SOURCE_TYPES


def _format_number(value: float | None) -> str | None:
    if value is None:
        return None
    return f"{value:g}"


def food_evidence(food: FoodNutrition) -> Evidence:
    source = food.source
    facts = [
        f"Food: {food.name}",
        f"Category: {food.category}" if food.category else None,
        f"Serving: {food.serving_size}" if food.serving_size else None,
        f"Calories: {_format_number(food.calories)} kcal"
        if food.calories is not None
        else None,
        f"Carbohydrates: {_format_number(food.carbohydrates_g)} g"
        if food.carbohydrates_g is not None
        else None,
        f"Protein: {_format_number(food.protein_g)} g"
        if food.protein_g is not None
        else None,
        f"Fat: {_format_number(food.fat_g)} g"
        if food.fat_g is not None
        else None,
        f"Fibre: {_format_number(food.fiber_g)} g"
        if food.fiber_g is not None
        else None,
        f"Sugar: {_format_number(food.sugar_g)} g"
        if food.sugar_g is not None
        else None,
        f"Sodium: {_format_number(food.sodium_mg)} mg"
        if food.sodium_mg is not None
        else None,
        f"Potassium: {_format_number(food.potassium_mg)} mg"
        if food.potassium_mg is not None
        else None,
        f"Glycemic index: {_format_number(food.glycemic_index)}"
        if food.glycemic_index is not None
        else None,
        f"Diabetes rating: {food.diabetes_rating}"
        if food.diabetes_rating
        else None,
        f"Ingredients: {food.ingredients}" if food.ingredients else None,
        f"Allergens: {', '.join(food.allergens)}"
        if food.allergens is not None
        else None,
        f"Dietary labels: {', '.join(food.dietary_labels)}"
        if food.dietary_labels is not None
        else None,
        f"Health tags: {food.health_tags}" if food.health_tags else None,
        f"Notes: {food.notes}" if food.notes else None,
    ]
    text = "\n".join(fact for fact in facts if fact)

    return Evidence(
        evidence_id=f"food-{food.id}",
        text=text,
        source_id=source.id,
        source_title=source.title,
        source_url=source.source_url,
        source_type=source.source_type,
        authority_score=source.authority_score,
        kind="food",
        metadata={
            "food_id": food.id,
            "name": food.name,
            "aliases": food.aliases or "",
            "category": food.category or "",
            "allergens": food.allergens,
            "dietary_labels": food.dietary_labels,
            "ingredients": food.ingredients,
            "allergen_data_verified": food.allergens is not None,
            "dietary_data_verified": food.dietary_labels is not None,
        },
    )


def chunk_evidence(chunk: DocumentChunk) -> Evidence:
    source: KnowledgeSource = chunk.source
    return Evidence(
        evidence_id=f"chunk-{chunk.id}",
        text=chunk.content,
        source_id=source.id,
        source_title=source.title,
        source_url=source.source_url,
        source_type=source.source_type,
        authority_score=source.authority_score,
        kind="document",
        metadata=dict(chunk.extra_data),
    )


def evidence_to_payload(evidence: Evidence) -> dict[str, object]:
    return {
        "evidence_id": evidence.evidence_id,
        "text": evidence.text,
        "source_id": evidence.source_id,
        "source_title": evidence.source_title,
        "source_url": evidence.source_url,
        "source_type": evidence.source_type,
        "authority_score": evidence.authority_score,
        "kind": evidence.kind,
        "metadata": evidence.metadata,
    }


def evidence_from_payload(payload: dict[str, object]) -> Evidence:
    metadata = payload.get("metadata")
    return Evidence(
        evidence_id=str(payload["evidence_id"]),
        text=str(payload["text"]),
        source_id=int(payload["source_id"]),
        source_title=str(payload["source_title"]),
        source_url=(
            str(payload["source_url"])
            if payload.get("source_url") is not None
            else None
        ),
        source_type=str(payload["source_type"]),
        authority_score=float(payload["authority_score"]),
        kind=str(payload["kind"]),
        metadata=metadata if isinstance(metadata, dict) else {},
    )
