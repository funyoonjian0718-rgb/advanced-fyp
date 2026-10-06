import argparse
import json
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import joinedload

from app.core.config import BASE_DIR
from app.core.database import SessionLocal
from app.models.knowledge import DocumentChunk, FoodNutrition
from app.services.evidence import chunk_evidence, food_evidence

DEFAULT_OUTPUT = (
    BASE_DIR / "backend" / "data" / "training" / "dietary_assistant.jsonl"
)


def _food_example(food: FoodNutrition) -> dict[str, str] | None:
    evidence = food_evidence(food)
    serving = food.serving_size or "the listed serving"
    claims: list[dict[str, object]] = []
    if food.calories is not None:
        claims.append(
            {
                "text": (
                    f"{food.name} is listed with {food.calories:g} kcal per "
                    f"serving of {serving}."
                ),
                "evidence_ids": [evidence.evidence_id],
            }
        )
    if food.protein_g is not None:
        claims.append(
            {
                "text": (
                    f"{food.name} is listed with {food.protein_g:g} g protein "
                    f"per serving of {serving}."
                ),
                "evidence_ids": [evidence.evidence_id],
            }
        )
    if food.sodium_mg is not None:
        claims.append(
            {
                "text": (
                    f"{food.name} is listed with {food.sodium_mg:g} mg sodium "
                    f"per serving of {serving}."
                ),
                "evidence_ids": [evidence.evidence_id],
            }
        )
    if not claims:
        return None
    prompt = (
        f"Question: Give the recorded nutrition values for {food.name}.\n"
        "Use only the evidence below. Return JSON with a claims array; each claim "
        "must cite the exact evidence ID.\n\n"
        f"[{evidence.evidence_id}] source type={evidence.source_type}; "
        f"authority={evidence.authority_score:.2f}\n{evidence.text}"
    )
    return {
        "instruction": prompt,
        "output": json.dumps({"claims": claims}, ensure_ascii=False),
    }


def _document_example(chunk: DocumentChunk) -> dict[str, str]:
    evidence = chunk_evidence(chunk)
    excerpt = evidence.text[:900].strip()
    return {
        "instruction": (
            "Question: What does the supplied source say?\n"
            "Use only the evidence below. Return JSON with a claims array; each "
            "claim must cite the exact evidence ID.\n\n"
            f"[{evidence.evidence_id}] source type={evidence.source_type}; "
            f"authority={evidence.authority_score:.2f}\n{evidence.text}"
        ),
        "output": json.dumps(
            {
                "claims": [
                    {
                        "text": excerpt,
                        "evidence_ids": [evidence.evidence_id],
                    }
                ]
            },
            ensure_ascii=False,
        ),
    }


def build_training_data(output: Path) -> int:
    output.parent.mkdir(parents=True, exist_ok=True)
    with SessionLocal() as db:
        foods = db.scalars(
            select(FoodNutrition)
            .options(joinedload(FoodNutrition.source))
            .order_by(FoodNutrition.id)
        ).all()
        chunks = db.scalars(
            select(DocumentChunk)
            .options(joinedload(DocumentChunk.source))
            .order_by(DocumentChunk.id)
        ).all()
        examples = [
            example
            for food in foods
            if (example := _food_example(food)) is not None
        ]
        examples.extend(_document_example(chunk) for chunk in chunks)

    if len(examples) < 4:
        raise ValueError(
            "At least four usable nutrition or document records are required "
            "to build the starter fine-tuning dataset."
        )

    with output.open("w", encoding="utf-8", newline="\n") as output_file:
        for example in examples:
            output_file.write(json.dumps(example, ensure_ascii=False) + "\n")
    return len(examples)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a starter JSONL fine-tuning set from the ingested knowledge "
            "base. The source values retain their original authority warnings."
        )
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    count = build_training_data(args.output)
    print(f"Wrote {count} grounded training examples to {args.output}.")


if __name__ == "__main__":
    main()
