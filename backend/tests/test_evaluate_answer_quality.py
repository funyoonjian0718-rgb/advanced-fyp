import json
from pathlib import Path

import pytest

from app.schemas.chat import ChatCitation, ChatResponse
from app.services.evidence import Evidence
from app.services.generation import ClaimVerification, GeneratedClaim
from app.services.generation import MALFORMED_MODEL_OUTPUT_WARNING
from app.scripts.evaluate_answer_quality import (
    DEFAULT_DATASET,
    DEFAULT_FOODS,
    DEFAULT_CHUNKS,
    aggregate_metrics,
    evaluate_case,
    failed_request_result,
    _has_numeric_facts,
    load_cases,
    load_sources,
    validate_source_references,
)


def _citation(evidence_id: str) -> ChatCitation:
    return ChatCitation(
        evidence_id=evidence_id,
        source_title="Test source",
        source_url=None,
        source_type="guideline",
        authority_score=0.8,
        excerpt="Test evidence.",
    )


def _response(
    answer: str,
    *,
    citations: list[ChatCitation] | None = None,
) -> ChatResponse:
    return ChatResponse(
        session_id=1,
        answer=answer,
        citations=citations or [],
        warnings=[],
        intent="document_evidence",
    )


def _case(
    *,
    outcome: str = "answerable",
    allergy_related: bool = False,
) -> dict[str, object]:
    return {
        "id": "case",
        "category": "guideline_based",
        "question": "Explain the guideline.",
        "expected_outcome": outcome,
        "source_records": [],
        "allergy_related": allergy_related,
    }


def test_dataset_is_grounded_in_available_source_records() -> None:
    cases = load_cases(DEFAULT_DATASET)
    foods, chunks = load_sources(DEFAULT_FOODS, DEFAULT_CHUNKS)

    assert len(cases) == 16
    assert validate_source_references(cases, foods, chunks) > 0
    assert {case["category"] for case in cases} == {
        "exact_nutrition_lookup",
        "food_comparison",
        "guideline_based",
        "numeric_constraints",
        "insufficient_evidence",
        "allergen_safety",
    }


@pytest.mark.parametrize(
    "record",
    [
        {},
        {
            "id": "duplicate",
            "category": "x",
            "question": "q",
            "expected_outcome": "made_up",
            "source_records": [],
        },
        {
            "id": "bad-source",
            "category": "x",
            "question": "q",
            "expected_outcome": "answerable",
            "source_records": [{"kind": "external", "id": "1"}],
        },
    ],
)
def test_loader_rejects_malformed_cases(tmp_path: Path, record) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="invalid fields"):
        load_cases(path)


def test_loader_rejects_duplicate_case_ids(tmp_path: Path) -> None:
    record = {
        "id": "same",
        "category": "unsupported",
        "question": "Moon dust?",
        "expected_outcome": "abstention_expected",
        "source_records": [],
    }
    path = tmp_path / "cases.jsonl"
    path.write_text(
        json.dumps(record) + "\n" + json.dumps(record) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="invalid fields"):
        load_cases(path)


def test_empty_evaluation_returns_null_rates_not_division_errors() -> None:
    metrics = aggregate_metrics([])

    assert metrics["case_count"] == 0
    assert metrics["unsupported_claim_rate"] is None
    assert metrics["citation_validity_rate"] is None
    assert metrics["unsafe_recommendation_count"] == 0


def test_failed_api_case_is_preserved_and_not_counted_as_abstention() -> None:
    row = failed_request_result(
        _case(outcome="abstention_expected"),
        http_status=502,
        detail="The local model failed. Authorization: Bearer private-token",
    )
    metrics = aggregate_metrics([row])

    assert row["actual_outcome"] == "request_failed"
    assert row["outcome_correct"] is False
    assert row["final_answer"] is None
    assert row["claim_details"] == []
    assert row["error_detail"] == (
        "The local model failed. Authorization=[REDACTED]"
    )
    assert metrics["request_failure_count"] == 1
    assert metrics["request_failure_rate"] == 1.0
    assert metrics["answerability_abstention_accuracy"] == 0.0


def test_malformed_model_output_is_not_scored_as_correct_abstention() -> None:
    response = ChatResponse(
        session_id=1,
        answer="I do not have enough supported evidence to answer.",
        citations=[],
        warnings=[MALFORMED_MODEL_OUTPUT_WARNING],
        intent="document_evidence",
    )

    row = evaluate_case(
        _case(outcome="abstention_expected"),
        response,
        [],
    )
    metrics = aggregate_metrics([row])

    assert row["actual_outcome"] == "generation_failed"
    assert row["generation_failed"] is True
    assert row["outcome_correct"] is False
    assert row["warnings"] == [MALFORMED_MODEL_OUTPUT_WARNING]
    assert metrics["generation_failure_count"] == 1
    assert metrics["answerability_abstention_accuracy"] == 0.0


def test_guideline_text_is_not_counted_as_a_numeric_claim() -> None:
    assert not _has_numeric_facts(
        "The guide recommends a variety of fruits and vegetables."
    )
    assert _has_numeric_facts("Tempeh contains 18 g protein per 100 g.")


def test_abstention_is_counted_correct_when_expected() -> None:
    row = evaluate_case(
        _case(outcome="abstention_expected"),
        _response("I do not have enough supported evidence to answer."),
        [],
    )

    assert row["actual_outcome"] == "abstention_expected"
    assert row["outcome_correct"] is True
    assert aggregate_metrics([row])["answerability_abstention_accuracy"] == 1.0


def test_valid_citation_id_is_not_treated_as_claim_support() -> None:
    evidence = Evidence(
        evidence_id="chunk-valid-id",
        text="Source says something unrelated.",
        source_id=1,
        source_title="Test source",
        source_url=None,
        source_type="guideline",
        authority_score=0.8,
        kind="document",
    )
    claim = GeneratedClaim(
        text="The guideline recommends drinking three litres of water.",
        evidence_ids=[evidence.evidence_id],
    )
    row = evaluate_case(
        _case(),
        _response(
            claim.text,
            citations=[_citation(evidence.evidence_id)],
        ),
        [evidence],
        [(claim, ClaimVerification("insufficient"))],
    )

    assert row["valid_citation_count"] == 1
    assert row["supported_citation_count"] == 0
    assert row["actual_outcome"] == "unsupported_answer"
    metrics = aggregate_metrics([row])
    assert metrics["citation_validity_rate"] == 1.0
    assert metrics["citation_evidence_support_rate"] == 0.0


def test_invalid_citation_is_not_valid_or_supported() -> None:
    row = evaluate_case(
        _case(),
        _response(
            "An answer.",
            citations=[_citation("fabricated-id")],
        ),
        [],
    )

    assert row["valid_citation_count"] == 0
    assert row["supported_citation_count"] == 0
    assert row["actual_outcome"] == "unsupported_answer"
    metrics = aggregate_metrics([row])
    assert metrics["citation_validity_rate"] == 0.0
    assert metrics["citation_evidence_support_rate"] == 0.0


def test_unsupported_numeric_claim_is_counted_as_rejected_and_incorrect() -> None:
    row = evaluate_case(
        _case(outcome="abstention_expected"),
        _response("I do not have enough supported evidence."),
        [],
        [
            (
                GeneratedClaim(
                    text="Tempeh contains 200 g protein per 100 g.",
                    evidence_ids=["food-record"],
                ),
                ClaimVerification("contradicted"),
            )
        ],
    )

    metrics = aggregate_metrics([row])
    assert row["unsupported_claim_count"] == 1
    assert row["claim_details"][0]["status"] == "contradicted"
    assert row["claim_details"][0]["included_in_response"] is False
    assert row["numeric_claim_count"] == 1
    assert row["correct_numeric_claim_count"] == 0
    assert metrics["unsupported_claim_rate"] == 1.0
    assert metrics["numeric_claim_correctness"] == 0.0


def test_structured_nutrition_answer_is_verified_against_cited_food() -> None:
    evidence = Evidence(
        evidence_id="food-record:1",
        text="Food: Nasi Lemak\nServing: 1 plate\nCalories: 520 kcal",
        source_id=1,
        source_title="Food dataset",
        source_url=None,
        source_type="user_uploaded_csv",
        authority_score=0.8,
        kind="food",
        metadata={"name": "Nasi Lemak"},
    )
    correct = evaluate_case(
        _case(),
        _response(
            "Nasi Lemak: Calories: 520 kcal for 1 plate.",
            citations=[_citation(evidence.evidence_id)],
        ),
        [evidence],
    )
    incorrect = evaluate_case(
        _case(),
        _response(
            "Nasi Lemak: Calories: 250 kcal for 1 plate.",
            citations=[_citation(evidence.evidence_id)],
        ),
        [evidence],
    )

    assert correct["correct_numeric_claim_count"] == 1
    assert correct["actual_outcome"] == "answerable"
    assert incorrect["correct_numeric_claim_count"] == 0
    assert incorrect["actual_outcome"] == "unsupported_answer"


def test_unverified_food_citation_is_flagged_for_allergy_case() -> None:
    evidence = Evidence(
        evidence_id="food-record:265",
        text="Food: Tempeh\nCalories: 190 kcal",
        source_id=265,
        source_title="Food dataset",
        source_url=None,
        source_type="user_dataset",
        authority_score=0.8,
        kind="food",
        metadata={
            "name": "Tempeh",
            "allergen_data_reviewed": False,
            "allergen_data_verified": False,
        },
    )
    row = evaluate_case(
        _case(outcome="safety_refusal_expected", allergy_related=True),
        _response(
            "Tempeh is safe to eat.",
            citations=[_citation(evidence.evidence_id)],
        ),
        [evidence],
    )

    assert row["unsafe_recommendation"] is True
    assert row["actual_outcome"] == "unsupported_answer"
    metrics = aggregate_metrics([row])
    assert metrics["unsafe_recommendation_count"] == 1
    assert metrics["unsafe_recommendation_rate"] == 1.0


def test_safe_refusal_does_not_trigger_unsafe_recommendation_heuristic() -> None:
    row = evaluate_case(
        _case(outcome="safety_refusal_expected", allergy_related=True),
        _response(
            "I can't verify that the retrieved foods are safe for your allergy."
        ),
        [],
    )

    assert row["actual_outcome"] == "safety_refusal_expected"
    assert row["unsafe_recommendation"] is False
