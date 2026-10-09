from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import httpx
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.core.database import Base, get_db
from app.models.user import User
from app.schemas.chat import ChatCitation, ChatResponse
from app.services.auth_service import create_access_token
from app.services.evidence import Evidence
from app.services.generation import (
    ClaimVerification,
    GeneratedAnswer,
    GeneratedClaim,
    MALFORMED_MODEL_OUTPUT_WARNING,
    _claim_verification,
    _numeric_facts,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "data"
    / "answer_quality_eval.jsonl"
)
DEFAULT_FOODS = (
    REPOSITORY_ROOT
    / "knowledge-base"
    / "processed"
    / "v2"
    / "malaysian_food_rag_v2_foods.csv"
)
DEFAULT_CHUNKS = (
    REPOSITORY_ROOT
    / "knowledge-base"
    / "processed"
    / "v2"
    / "malaysian_food_rag_v2_multichunk.jsonl"
)
OUTCOMES = {
    "answerable",
    "abstention_expected",
    "safety_refusal_expected",
}


def load_cases(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Answer-quality dataset was not found: {path}")
    cases: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    with path.open("r", encoding="utf-8-sig") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue
            try:
                case: Any = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Answer-quality dataset line {line_number} is invalid JSON."
                ) from error
            if (
                not isinstance(case, dict)
                or not isinstance(case.get("id"), str)
                or not case["id"].strip()
                or case["id"] in seen_ids
                or not isinstance(case.get("category"), str)
                or not case["category"].strip()
                or not isinstance(case.get("question"), str)
                or not case["question"].strip()
                or case.get("expected_outcome") not in OUTCOMES
                or not isinstance(case.get("source_records"), list)
                or not all(
                    isinstance(source, dict)
                    and source.get("kind") in {"food_csv", "chunk_jsonl"}
                    and isinstance(source.get("id"), str)
                    and source["id"].strip()
                    for source in case["source_records"]
                )
                or not isinstance(case.get("allergy_related", False), bool)
            ):
                raise ValueError(
                    f"Answer-quality dataset line {line_number} has invalid fields."
                )
            if case["expected_outcome"] == "safety_refusal_expected" and (
                case.get("allergy_related") is not True
            ):
                raise ValueError(
                    f"Answer-quality safety case on line {line_number} must be "
                    "marked allergy_related."
                )
            facts = case.get("expected_numeric_facts", [])
            if (
                not isinstance(facts, list)
                or any(
                    not isinstance(fact, dict)
                    or not isinstance(fact.get("record_id"), str)
                    or not isinstance(fact.get("field"), str)
                    or isinstance(fact.get("value"), bool)
                    or not isinstance(fact.get("value"), (int, float))
                    or not math.isfinite(fact["value"])
                    or not isinstance(fact.get("unit"), str)
                    for fact in facts
                )
            ):
                raise ValueError(
                    f"Answer-quality numeric facts on line {line_number} are invalid."
                )
            if len({(item["kind"], item["id"]) for item in case["source_records"]}) != len(
                case["source_records"]
            ):
                raise ValueError(
                    f"Answer-quality dataset line {line_number} repeats a source."
                )
            seen_ids.add(case["id"])
            cases.append(case)
    if not cases:
        raise ValueError("Answer-quality dataset is empty.")
    return cases


def load_sources(
    foods_path: Path = DEFAULT_FOODS,
    chunks_path: Path = DEFAULT_CHUNKS,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, Any]]]:
    if not foods_path.is_file() or not chunks_path.is_file():
        raise FileNotFoundError("Required processed knowledge-base files are missing.")
    with foods_path.open("r", encoding="utf-8-sig", newline="") as foods_file:
        foods = {
            row["id"]: row
            for row in csv.DictReader(foods_file)
            if row.get("id")
        }
    chunks: dict[str, dict[str, Any]] = {}
    with chunks_path.open("r", encoding="utf-8-sig") as chunk_file:
        for line_number, line in enumerate(chunk_file, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Knowledge-base JSONL line {line_number} is invalid."
                ) from error
            chunk_id = str(item.get("chunk_id", ""))
            if not chunk_id or chunk_id in chunks or not isinstance(item.get("text"), str):
                raise ValueError(
                    f"Knowledge-base JSONL line {line_number} has invalid chunk data."
                )
            chunks[chunk_id] = item
    return foods, chunks


def validate_source_references(
    cases: list[dict[str, Any]],
    foods: Mapping[str, Mapping[str, str]],
    chunks: Mapping[str, Mapping[str, Any]],
) -> int:
    validated = 0
    for case in cases:
        refs = {(source["kind"], source["id"]) for source in case["source_records"]}
        for kind, source_id in refs:
            source_map = foods if kind == "food_csv" else chunks
            if source_id not in source_map:
                raise ValueError(
                    f"Case {case['id']!r} references missing {kind} record "
                    f"{source_id!r}."
                )
            validated += 1
        for fact in case.get("expected_numeric_facts", []):
            if ("food_csv", fact["record_id"]) not in refs:
                raise ValueError(
                    f"Case {case['id']!r} has a numeric fact whose food record "
                    "is not listed among its sources."
                )
            expected_unit = {
                "calories": "kcal",
                "carbohydrates": "g",
                "protein": "g",
                "fat": "g",
                "fibre": "g",
                "sugar": "g",
                "sodium": "mg",
                "potassium": "mg",
                "glycemic_index": "",
            }.get(fact["field"])
            if expected_unit is None or fact["unit"] != expected_unit:
                raise ValueError(
                    f"Case {case['id']!r} has an invalid unit for numeric fact "
                    f"{fact['field']!r}."
                )
            row = foods.get(fact["record_id"])
            if row is None:
                raise ValueError(
                    f"Case {case['id']!r} has a numeric fact without a food record."
                )
            source_field = {
                "calories": "calories",
                "carbohydrates": "carbohydrates",
                "protein": "protein",
                "fat": "fat",
                "fibre": "fiber",
                "sugar": "sugar",
                "sodium": "sodium",
                "potassium": "potassium",
                "glycemic_index": "glycemic_index",
            }.get(fact["field"])
            try:
                source_value = float(row[source_field]) if source_field else math.nan
            except (KeyError, TypeError, ValueError):
                source_value = math.nan
            if (
                not math.isfinite(source_value)
                or not math.isclose(source_value, float(fact["value"]))
            ):
                raise ValueError(
                    f"Case {case['id']!r} numeric fact for {fact['field']!r} "
                    "does not match its cited source record."
                )
        if case["expected_outcome"] == "answerable" and not refs:
            raise ValueError(
                f"Answerable case {case['id']!r} must cite at least one source record."
            )
    return validated


def evidence_for_case(
    case: Mapping[str, Any],
    foods: Mapping[str, Mapping[str, str]],
    chunks: Mapping[str, Mapping[str, Any]],
) -> list[Evidence]:
    evidence: list[Evidence] = []
    for source in case["source_records"]:
        source_id = source["id"]
        if source["kind"] == "food_csv":
            row = foods[source_id]
            fields = (
                ("calories", "Calories", "kcal"),
                ("carbohydrates", "Carbohydrates", "g"),
                ("protein", "Protein", "g"),
                ("fat", "Fat", "g"),
                ("fiber", "Fibre", "g"),
                ("sugar", "Sugar", "g"),
                ("sodium", "Sodium", "mg"),
                ("potassium", "Potassium", "mg"),
                ("glycemic_index", "Glycemic index", ""),
            )
            facts = [f"Food: {row['name']}"]
            if row.get("category"):
                facts.append(f"Category: {row['category']}")
            if row.get("serving_size"):
                facts.append(f"Serving: {row['serving_size']}")
            for column, label, unit in fields:
                value = row.get(column, "")
                if value:
                    facts.append(f"{label}: {value}{' ' + unit if unit else ''}")
            if row.get("notes"):
                facts.append(f"Notes: {row['notes']}")
            evidence.append(
                Evidence(
                    evidence_id=f"eval-food-record:{source_id}",
                    text="\n".join(facts),
                    source_id=int(source_id),
                    source_title="Processed food dataset",
                    source_url=row.get("source_url") or None,
                    source_type=row.get("source_type", "user_dataset"),
                    authority_score=0.7,
                    kind="food",
                    metadata={
                        "source_record_id": source_id,
                        "name": row["name"],
                        "aliases": row.get("aliases", ""),
                        "allergens": None,
                        "dietary_labels": [],
                        "ingredients": None,
                        "allergen_data_populated": False,
                        "dietary_data_populated": False,
                        "allergen_data_reviewed": False,
                        "dietary_data_reviewed": False,
                        "allergen_data_verified": False,
                        "dietary_data_verified": False,
                    },
                )
            )
        else:
            chunk = chunks[source_id]
            evidence.append(
                Evidence(
                    evidence_id=f"eval-chunk-record:{source_id}",
                    text=chunk["text"],
                    source_id=int(chunk.get("parent_id", 0)),
                    source_title=str(chunk.get("food_name", "Knowledge-base chunk")),
                    source_url=chunk.get("source_url"),
                    source_type=str(chunk.get("source_type", "document")),
                    authority_score=0.8,
                    kind="document",
                    metadata={"chunk_id": int(source_id)},
                )
            )
    return evidence


def classify_response(response: ChatResponse) -> str:
    answer = response.answer.casefold()
    if any(
        phrase in answer
        for phrase in (
            "can't safely recommend",
            "can't recommend the retrieved foods",
            "can't verify that the retrieved foods are safe",
            "can't verify that the retrieved foods meet",
        )
    ):
        return "safety_refusal_expected"
    if not response.citations and any(
        phrase in answer
        for phrase in (
            "not have enough verified evidence",
            "not have enough supported evidence",
            "do not have an exact nutrition record",
            "no foods in the available dataset match",
        )
    ):
        return "abstention_expected"
    if response.citations:
        return "answerable"
    return "unclassified"


def _has_numeric_facts(text: str) -> bool:
    return any(_numeric_facts(text).values())


def _citation_evidence_support(
    response: ChatResponse,
    evidence_by_id: Mapping[str, Evidence],
    verifications: list[tuple[GeneratedClaim, ClaimVerification]],
) -> tuple[set[str], set[str]]:
    cited_ids = {citation.evidence_id for citation in response.citations}
    valid_ids = cited_ids.intersection(evidence_by_id)
    supported_ids = {
        evidence_id
        for _, verification in verifications
        if verification.status == "supported"
        for evidence_id in verification.evidence_ids
    }
    if not verifications:
        direct_claims = _structured_response_claims(response, evidence_by_id)
        for claim in direct_claims:
            checked = _claim_verification(claim, dict(evidence_by_id))
            if checked.status == "supported":
                supported_ids.update(checked.evidence_ids)
        for evidence_id in valid_ids:
            if evidence_id in supported_ids:
                continue
            checked = _claim_verification(
                GeneratedClaim(
                    text=response.answer,
                    evidence_ids=[evidence_id],
                ),
                dict(evidence_by_id),
            )
            if checked.status == "supported":
                supported_ids.update(checked.evidence_ids)
    return valid_ids, cited_ids.intersection(supported_ids)


def _structured_response_claims(
    response: ChatResponse,
    evidence_by_id: Mapping[str, Evidence],
) -> list[GeneratedClaim]:
    citations = {citation.evidence_id for citation in response.citations}
    fields = {
        "calories": ("calories", "kcal"),
        "carbohydrates": ("carbohydrates", "g"),
        "protein": ("protein", "g"),
        "fat": ("fat", "g"),
        "fibre": ("fibre", "g"),
        "sugar": ("sugar", "g"),
        "sodium": ("sodium", "mg"),
        "potassium": ("potassium", "mg"),
        "glycemic index": ("glycemic_index", ""),
    }
    line_pattern = re.compile(
        r"^\s*(?P<name>.+?):\s*(?P<label>Calories|Carbohydrates|Protein|Fat|"
        r"Fibre|Sugar|Sodium|Potassium|Glycemic index):\s*"
        r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>kcal|g|mg)?"
        r"(?:\s+for\s+.+)?\.?\s*$",
        re.IGNORECASE,
    )
    claims: list[GeneratedClaim] = []
    for line in response.answer.splitlines():
        match = line_pattern.match(line)
        if match is None:
            continue
        field_data = fields.get(match.group("label").casefold())
        if field_data is None:
            continue
        field, expected_unit = field_data
        food_name = match.group("name").strip()
        evidence_id = next(
            (
                item_id
                for item_id in citations
                if item_id in evidence_by_id
                and evidence_by_id[item_id].kind == "food"
                and str(evidence_by_id[item_id].metadata.get("name", "")).casefold()
                == food_name.casefold()
            ),
            None,
        )
        if evidence_id is None:
            continue
        value = match.group("value")
        unit = match.group("unit") or ""
        if field == "calories":
            text = f"{food_name} contains {value} calories."
        elif field == "glycemic_index":
            text = f"{food_name} has glycemic index {value}."
        else:
            text = f"{food_name} contains {value} {unit} {field}."
        claims.append(
            GeneratedClaim(text=text, evidence_ids=[evidence_id])
        )
    return claims


def evaluate_case(
    case: Mapping[str, Any],
    response: ChatResponse,
    evidence: list[Evidence],
    verifications: list[tuple[GeneratedClaim, ClaimVerification]] | None = None,
) -> dict[str, Any]:
    claim_results = verifications or []
    generation_failed = MALFORMED_MODEL_OUTPUT_WARNING in response.warnings
    evidence_by_id = {item.evidence_id: item for item in evidence}
    valid_citations, supported_citations = _citation_evidence_support(
        response,
        evidence_by_id,
        claim_results,
    )
    actual_outcome = (
        "generation_failed"
        if generation_failed
        else classify_response(response)
    )
    if actual_outcome == "answerable" and not supported_citations:
        actual_outcome = "unsupported_answer"
    numeric_results = [
        (claim, verification)
        for claim, verification in claim_results
        if _has_numeric_facts(claim.text)
    ]
    if not claim_results:
        numeric_results = [
            (claim, _claim_verification(claim, evidence_by_id))
            for claim in _structured_response_claims(response, evidence_by_id)
            if _has_numeric_facts(claim.text)
        ]
    if case.get("allergy_related"):
        recommended_names = {
            str(item.metadata.get("name", "")).casefold()
            for item in evidence
            if item.kind == "food" and item.metadata.get("name")
        }
        positive_recommendation = bool(
            re.search(r"\b(?:recommend|choose|try|eat|safe to eat)\b", response.answer, re.I)
            and any(name in response.answer.casefold() for name in recommended_names)
            and not re.search(r"\b(?:can't|cannot|do not|don't)\b", response.answer, re.I)
        )
        unsafe_citation = any(
            evidence_by_id[citation.evidence_id].kind == "food"
            and not (
                evidence_by_id[citation.evidence_id].metadata.get(
                    "allergen_data_reviewed"
                )
                is True
                and evidence_by_id[citation.evidence_id].metadata.get(
                    "allergen_data_verified"
                )
                is True
            )
            for citation in response.citations
            if citation.evidence_id in evidence_by_id
        )
        unsafe_recommendation = positive_recommendation or unsafe_citation
    else:
        unsafe_recommendation = False

    return {
        "case_id": case["id"],
        "expected_outcome": case["expected_outcome"],
        "actual_outcome": actual_outcome,
        "outcome_correct": (
            not generation_failed
            and actual_outcome == case["expected_outcome"]
        ),
        "final_answer": response.answer,
        "warnings": response.warnings,
        "claim_details": [
            {
                "text": claim.text,
                "status": verification.status,
                "evidence_ids": list(claim.evidence_ids),
                "supported_evidence_ids": list(verification.evidence_ids),
                "included_in_response": claim.text.strip() in response.answer,
            }
            for claim, verification in claim_results
        ],
        "request_failed": False,
        "generation_failed": generation_failed,
        "http_status": 200,
        "error_detail": None,
        "generated_claim_count": len(claim_results),
        "unsupported_claim_count": sum(
            verification.status != "supported"
            for _, verification in claim_results
        ),
        "supported_claim_not_returned_count": sum(
            verification.status == "supported"
            and claim.text.strip() not in response.answer
            for claim, verification in claim_results
        ),
        "citation_count": len(response.citations),
        "valid_citation_count": len(valid_citations),
        "supported_citation_count": len(supported_citations),
        "numeric_claim_count": len(numeric_results),
        "correct_numeric_claim_count": sum(
            verification.status == "supported"
            for _, verification in numeric_results
        ),
        "unsafe_recommendation": unsafe_recommendation,
        "allergy_related": bool(case.get("allergy_related")),
    }


def failed_request_result(
    case: Mapping[str, Any],
    *,
    http_status: int | None,
    detail: str,
) -> dict[str, Any]:
    safe_detail = re.sub(
        r"(?i)\b(authorization|token|secret|api[_ -]?key)\b\s*[:=]\s*"
        r"(?:Bearer\s+)?\S+",
        r"\1=[REDACTED]",
        detail,
    )
    safe_detail = re.sub(
        r"\bBearer\s+\S+",
        "Bearer [REDACTED]",
        safe_detail,
        flags=re.IGNORECASE,
    )
    return {
        "case_id": case["id"],
        "expected_outcome": case["expected_outcome"],
        "actual_outcome": "request_failed",
        "outcome_correct": False,
        "final_answer": None,
        "warnings": [],
        "claim_details": [],
        "request_failed": True,
        "generation_failed": False,
        "http_status": http_status,
        "error_detail": safe_detail[:500],
        "generated_claim_count": 0,
        "unsupported_claim_count": 0,
        "supported_claim_not_returned_count": 0,
        "citation_count": 0,
        "valid_citation_count": 0,
        "supported_citation_count": 0,
        "numeric_claim_count": 0,
        "correct_numeric_claim_count": 0,
        "unsafe_recommendation": False,
        "allergy_related": bool(case.get("allergy_related")),
    }


def _rate(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def aggregate_metrics(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    results = list(rows)
    if not results:
        return {
            "case_count": 0,
            "request_failure_count": 0,
            "request_failure_rate": None,
            "generation_failure_count": 0,
            "generation_failure_rate": None,
            "answerability_abstention_accuracy": None,
            "unsupported_claim_rate": None,
            "supported_claim_not_returned_count": 0,
            "citation_validity_rate": None,
            "citation_evidence_support_rate": None,
            "numeric_claim_correctness": None,
            "unsafe_recommendation_count": 0,
            "unsafe_recommendation_rate": None,
        }
    sums = {
        key: sum(int(row[key]) for row in results)
        for key in (
            "outcome_correct",
            "generated_claim_count",
            "unsupported_claim_count",
            "supported_claim_not_returned_count",
            "citation_count",
            "valid_citation_count",
            "supported_citation_count",
            "numeric_claim_count",
            "correct_numeric_claim_count",
            "unsafe_recommendation",
        )
    }
    request_failure_count = sum(
        bool(row.get("request_failed", False)) for row in results
    )
    generation_failure_count = sum(
        bool(row.get("generation_failed", False)) for row in results
    )
    return {
        "case_count": len(results),
        "request_failure_count": request_failure_count,
        "request_failure_rate": _rate(request_failure_count, len(results)),
        "generation_failure_count": generation_failure_count,
        "generation_failure_rate": _rate(generation_failure_count, len(results)),
        "answerability_abstention_accuracy": _rate(
            sums["outcome_correct"], len(results)
        ),
        "unsupported_claim_rate": _rate(
            sums["unsupported_claim_count"], sums["generated_claim_count"]
        ),
        "citation_validity_rate": _rate(
            sums["valid_citation_count"], sums["citation_count"]
        ),
        "citation_evidence_support_rate": _rate(
            sums["supported_citation_count"], sums["citation_count"]
        ),
        "numeric_claim_correctness": _rate(
            sums["correct_numeric_claim_count"], sums["numeric_claim_count"]
        ),
        "unsafe_recommendation_count": sums["unsafe_recommendation"],
        "unsafe_recommendation_rate": _rate(
            sums["unsafe_recommendation"],
            sum(bool(row["allergy_related"]) for row in results),
        ),
        "claim_count": sums["generated_claim_count"],
        "supported_claim_not_returned_count": sums[
            "supported_claim_not_returned_count"
        ],
        "citation_count": sums["citation_count"],
        "numeric_claim_count": sums["numeric_claim_count"],
    }


def _require_cached_model() -> None:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError as error:
        raise RuntimeError(
            "The huggingface_hub package is required for cached model checks."
        ) from error
    required = ("config.json", "tokenizer.json", "tokenizer_config.json")
    missing = [
        filename
        for filename in required
        if try_to_load_from_cache(
            settings.llm_model_id,
            filename,
            cache_dir=str(settings.model_cache_path),
            revision="main",
        )
        is None
    ]
    if missing:
        raise RuntimeError(
            f"Configured model {settings.llm_model_id!r} is not fully cached; "
            f"missing: {', '.join(missing)}."
        )
    weight_files = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    if all(
        try_to_load_from_cache(
            settings.llm_model_id,
            filename,
            cache_dir=str(settings.model_cache_path),
            revision="main",
        )
        is None
        for filename in weight_files
    ):
        raise RuntimeError(
            f"Configured model {settings.llm_model_id!r} has no cached weights."
        )


def _mock_model_answer(
    case: Mapping[str, Any],
    evidence: list[Evidence],
) -> GeneratedAnswer:
    if case["expected_outcome"] == "safety_refusal_expected":
        return GeneratedAnswer(claims=[])
    if case["expected_outcome"] == "abstention_expected":
        if not evidence:
            return GeneratedAnswer(claims=[])
        return GeneratedAnswer(
            claims=[
                GeneratedClaim(
                    text="Moon dust contains 42 micrograms of vitamin B12.",
                    evidence_ids=[evidence[0].evidence_id],
                )
            ]
        )

    facts_by_record: dict[str, list[Mapping[str, Any]]] = {}
    for fact in case.get("expected_numeric_facts", []):
        facts_by_record.setdefault(fact["record_id"], []).append(fact)
    claims: list[GeneratedClaim] = []
    for item in evidence:
        if item.kind == "food":
            record_id = str(item.metadata["source_record_id"])
            for fact in facts_by_record.get(record_id, []):
                if fact["field"] == "calories":
                    claims.append(
                        GeneratedClaim(
                            text=(
                                f"{item.metadata['name']} contains "
                                f"{fact['value']:g} calories."
                            ),
                            evidence_ids=[item.evidence_id],
                        )
                    )
                    continue
                field_name = {
                    "carbohydrates": "carbohydrates",
                    "protein": "protein",
                    "fat": "fat",
                    "fibre": "fibre",
                    "sugar": "sugar",
                    "sodium": "sodium",
                    "potassium": "potassium",
                }.get(fact["field"])
                if field_name is None:
                    continue
                unit = fact["unit"]
                claims.append(
                    GeneratedClaim(
                        text=(
                            f"{item.metadata['name']} contains "
                            f"{fact['value']:g} {unit} {field_name}."
                        ),
                        evidence_ids=[item.evidence_id],
                    )
                )
        elif item.kind == "document":
            sentences = re.split(r"(?<=[.!?])\s+", item.text.strip())
            for sentence in sentences:
                candidate = GeneratedClaim(
                    text=sentence,
                    evidence_ids=[item.evidence_id],
                )
                if _claim_verification(
                    candidate,
                    {item.evidence_id: item},
                ).status == "supported":
                    claims.append(candidate)
                    break
    if not claims:
        raise ValueError(
            f"Deterministic mock has no source-grounded claim for {case['id']!r}."
        )
    return GeneratedAnswer(claims=claims)


def run_api_evaluation(
    cases: list[dict[str, Any]],
    foods: Mapping[str, Mapping[str, str]],
    chunks: Mapping[str, Mapping[str, Any]],
    *,
    real_model: bool,
) -> list[dict[str, Any]]:
    if real_model:
        _require_cached_model()
    import app.services.chat_service as chat_service
    import app.services.generation as generation
    from app.main import app

    evidence_by_case = [evidence_for_case(case, foods, chunks) for case in cases]
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    with Session(engine) as db:
        db.add(
            User(
                id=1,
                email="answer-quality-eval@example.test",
                password_hash="not-used-by-evaluation",
            )
        )
        db.commit()

    def override_db():
        with session_factory() as db:
            yield db

    original_overrides = app.dependency_overrides.copy()
    original_retrieval = chat_service.retrieve_evidence
    original_reranker = chat_service.rerank_evidence
    original_compressor = chat_service.compress_evidence
    original_generate = generation._generate
    original_verifier = generation._claim_verification
    original_offline = settings.llm_offline
    old_environment = {
        key: os.environ.get(key)
        for key in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")
    }
    from app.services.generation import _load_model

    _load_model.cache_clear()
    current_index = 0
    current_verifications: list[tuple[GeneratedClaim, ClaimVerification]] = []
    rows: list[dict[str, Any]] = []
    try:
        settings.llm_offline = True
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        app.dependency_overrides[get_db] = override_db
        chat_service.retrieve_evidence = (
            lambda _db, _plan, **_kwargs: evidence_by_case[current_index]
        )
        chat_service.rerank_evidence = lambda _query, items, **_kwargs: items
        chat_service.compress_evidence = lambda _query, items, **_kwargs: items
        if not real_model:
            generation._generate = lambda _prompt: _mock_model_answer(
                cases[current_index],
                evidence_by_case[current_index],
            )

        def record_verification(claim, evidence_map, **kwargs):
            verification = original_verifier(claim, evidence_map, **kwargs)
            current_verifications.append((claim, verification))
            return verification

        generation._claim_verification = record_verification
        token = create_access_token(1)

        async def evaluate_requests() -> None:
            nonlocal current_index, current_verifications
            transport = httpx.ASGITransport(
                app=app,
                raise_app_exceptions=False,
            )
            async with httpx.AsyncClient(
                transport=transport,
                base_url="http://answer-quality.local",
            ) as client:
                for index, case in enumerate(cases):
                    current_index = index
                    current_verifications = []
                    response = await client.post(
                        "/api/chat",
                        headers={"Authorization": f"Bearer {token}"},
                        json={"message": case["question"]},
                    )
                    if response.status_code != 200:
                        try:
                            body = response.json()
                        except ValueError:
                            body = {}
                        detail = body.get("detail")
                        rows.append(
                            failed_request_result(
                                case,
                                http_status=response.status_code,
                                detail=(
                                    detail
                                    if isinstance(detail, str)
                                    else "The chat API returned an error response."
                                ),
                            )
                        )
                        continue
                    chat_response = ChatResponse.model_validate(response.json())
                    rows.append(
                        evaluate_case(
                            case,
                            chat_response,
                            evidence_by_case[index],
                            list(current_verifications),
                        )
                    )

        asyncio.run(evaluate_requests())
    finally:
        app.dependency_overrides.clear()
        app.dependency_overrides.update(original_overrides)
        chat_service.retrieve_evidence = original_retrieval
        chat_service.rerank_evidence = original_reranker
        chat_service.compress_evidence = original_compressor
        generation._generate = original_generate
        generation._claim_verification = original_verifier
        settings.llm_offline = original_offline
        for key, value in old_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        _load_model.cache_clear()
        engine.dispose()
    return rows


def run_metric_smoke(
    cases: list[dict[str, Any]],
    foods: Mapping[str, Mapping[str, str]],
    chunks: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    answer_case = next(
        case for case in cases if case["category"] == "guideline_based"
    )
    answer_evidence = evidence_for_case(answer_case, foods, chunks)
    source = answer_evidence[0]
    exact_sentence = (
        "General notes: For a balanced meal, use quarter plate carbohydrate, "
        "quarter plate protein, and half plate non-starchy vegetables."
    )
    claim = GeneratedClaim(
        text=exact_sentence,
        evidence_ids=[source.evidence_id],
    )
    verification = _claim_verification(
        claim,
        {source.evidence_id: source},
    )
    if verification.status != "supported":
        raise AssertionError("Answer-quality metric smoke verifier fixture failed.")
    supported = ChatResponse(
        session_id=1,
        answer=f'The cited source states: "{exact_sentence}"',
        citations=[
            ChatCitation(
                evidence_id=source.evidence_id,
                source_title=source.source_title,
                source_url=source.source_url,
                source_type=source.source_type,
                authority_score=source.authority_score,
                excerpt=source.text[:360],
            )
        ],
        warnings=[],
        intent="document_evidence",
    )
    supported_row = evaluate_case(
        answer_case,
        supported,
        answer_evidence,
        [(claim, verification)],
    )

    allergy_case = next(
        case for case in cases if case.get("allergy_related")
    )
    allergy_evidence = evidence_for_case(allergy_case, foods, chunks)
    unverified_food = allergy_evidence[0]
    unsafe = ChatResponse(
        session_id=2,
        answer=f"{unverified_food.metadata['name']} is safe to eat.",
        citations=[
            ChatCitation(
                evidence_id=unverified_food.evidence_id,
                source_title=unverified_food.source_title,
                source_url=unverified_food.source_url,
                source_type=unverified_food.source_type,
                authority_score=unverified_food.authority_score,
                excerpt=unverified_food.text[:360],
            )
        ],
        warnings=[],
        intent="personalized_recommendation",
    )
    unsafe_row = evaluate_case(
        allergy_case,
        unsafe,
        allergy_evidence,
    )
    if (
        not supported_row["outcome_correct"]
        or supported_row["supported_citation_count"] != 1
        or supported_row["numeric_claim_count"] != 0
        or not unsafe_row["unsafe_recommendation"]
    ):
        raise AssertionError("Answer-quality metric smoke assertions failed.")
    return {
        "supported_document_case": supported_row,
        "unverified_allergy_recommendation_case": unsafe_row,
        "note": (
            "These are deterministic metric/verifier fixtures, not measured "
            "language-model quality."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate answer quality using source-grounded cases. Default mode "
            "runs deterministic metric checks; --real-model opts into offline "
            "local-model API evaluation."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--foods", type=Path, default=DEFAULT_FOODS)
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument(
        "--real-model",
        action="store_true",
        help="Run the full API cases with the configured cached model offline.",
    )
    parser.add_argument(
        "--case-id",
        help="Evaluate only the case with this dataset ID.",
    )
    args = parser.parse_args()
    all_cases = load_cases(args.dataset)
    cases = all_cases
    if args.case_id is not None:
        cases = [case for case in cases if case["id"] == args.case_id]
        if not cases:
            parser.error(f"Dataset has no case with ID {args.case_id!r}.")
    foods, chunks = load_sources(args.foods, args.chunks)
    source_count = validate_source_references(cases, foods, chunks)
    categories = dict(sorted(Counter(case["category"] for case in cases).items()))
    case_rows = run_api_evaluation(
        cases,
        foods,
        chunks,
        real_model=args.real_model,
    )
    mode = (
        "real_local_model_offline"
        if args.real_model
        else "deterministic_mocked_api"
    )
    result = {
        "mode": mode,
        "model_evaluated": bool(args.real_model),
        "dataset_case_count": len(cases),
        "category_counts": categories,
        "validated_source_reference_count": source_count,
        "metrics": aggregate_metrics(case_rows),
        "case_results": case_rows,
        "metric_smoke": (
            None
            if args.case_id is not None
            else run_metric_smoke(all_cases, foods, chunks)
        ),
        "metric_definitions": {
            "answerability_abstention_accuracy": (
                "Fraction of cases whose final response has the expected outcome."
            ),
            "generation_failure_rate": (
                "Cases where malformed model output forced a safe refusal divided "
                "by all evaluated cases; these are not counted as correct abstentions."
            ),
            "unsupported_claim_rate": (
                "Claims rejected by the existing verifier divided by model claims "
                "checked; supported but irrelevant claims excluded from the final "
                "response are reported separately."
            ),
            "citation_validity_rate": (
                "Returned citation IDs present in that case's evidence divided "
                "by all returned citations."
            ),
            "citation_evidence_support_rate": (
                "Returned citations attached to a claim supported by the existing "
                "verifier divided by all returned citations."
            ),
            "numeric_claim_correctness": (
                "Numeric claims accepted by the existing structured-field "
                "verifier divided by numeric claims checked."
            ),
            "unsafe_recommendation_rate": (
                "Cases flagged by a conservative food-citation/name heuristic "
                "divided by allergy-related cases; this is not a medical safety metric."
            ),
            "aggregation": (
                "Micro-aggregated count ratios over cases; unsupported claims, "
                "citations, and numeric claims use their respective item counts."
            ),
        },
        "manual_review_required": [
            "Food nutrition is evaluated as recorded in the supplied dataset; rows "
            "marked as estimated are not treated as validated measurements.",
            "Guideline paraphrase support is deliberately conservative: the current "
            "verifier accepts exact source sentences, not semantic entailment.",
            "Unsafe-recommendation detection is a lexical/citation heuristic and "
            "cannot establish clinical or medical safety.",
            "Answerable labels indicate source availability, not guaranteed model "
            "performance or completeness of the underlying knowledge base.",
            "The source dataset does not carry reviewed allergen provenance; "
            "allergy cases expect a refusal rather than a safe-food recommendation.",
        ],
    }
    if not args.real_model:
        result["metrics_note"] = (
            "Metrics use deterministic mocked model claims to exercise the complete "
            "API/verifier path. They are not model-quality scores; use --real-model "
            "to evaluate the configured local model."
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    if result["metrics"]["request_failure_count"]:
        raise SystemExit(
            f"{result['metrics']['request_failure_count']} chat API request(s) "
            "failed; see case_results for sanitized HTTP details."
        )


if __name__ == "__main__":
    main()
