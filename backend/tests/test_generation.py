from app.services.constraints import ConstraintResult
from app.services.evidence import Evidence
from app.services.generation import (
    GeneratedClaim,
    _canonical_numbers,
    _claim_is_supported,
    generate_verified_answer,
)
from app.services.query_processor import process_query


def test_claim_verification_requires_cited_evidence_and_matching_numbers() -> None:
    evidence = Evidence(
        evidence_id="food-1",
        text="Food: Tempeh\nServing: 100 g\nProtein: 20.3 g",
        source_id=1,
        source_title="Nutrition record",
        source_url=None,
        source_type="guideline_snippet",
        authority_score=0.7,
        kind="food",
    )
    evidence_map = {evidence.evidence_id: evidence}

    assert _claim_is_supported(
        GeneratedClaim(
            text="Tempeh contains 20.3 g protein per 100 g.",
            evidence_ids=["food-1"],
        ),
        evidence_map,
    )
    assert not _claim_is_supported(
        GeneratedClaim(
            text="Tempeh contains 99 g protein per 100 g.",
            evidence_ids=["food-1"],
        ),
        evidence_map,
    )
    assert not _claim_is_supported(
        GeneratedClaim(
            text="Tempeh contains 20.3 g protein per 100 g.",
            evidence_ids=["missing"],
        ),
        evidence_map,
    )


def test_numeric_normalization_preserves_tens() -> None:
    assert _canonical_numbers("20.0 g and 0.50 g") == {"20", "0.5"}


def test_empty_evidence_returns_safe_refusal_without_loading_model() -> None:
    plan = process_query("How much protein is in tempeh?")

    result = generate_verified_answer(
        "How much protein is in tempeh?",
        plan,
        [],
        ConstraintResult(allowed=[], excluded=[], warnings=[]),
    )

    assert result.confidence == 0
    assert not result.citations
    assert "not have enough verified evidence" in result.answer


def test_structured_food_lookup_returns_requested_recorded_nutrient(
    monkeypatch,
) -> None:
    evidence = Evidence(
        evidence_id="food-1",
        text=(
            "Food: Tempeh\nServing: 1 serving (100g)\n"
            "Calories: 190 kcal\nProtein: 18 g\nFibre: 4 g"
        ),
        source_id=1,
        source_title="Nutrition record",
        source_url=None,
        source_type="user_uploaded_json",
        authority_score=0.3,
        kind="food",
        metadata={"name": "Tempeh"},
    )

    def unexpected_model_call(_prompt: str):
        raise AssertionError("A direct structured lookup should not call the LLM.")

    monkeypatch.setattr("app.services.generation._generate", unexpected_model_call)
    question = "How much protein is in tempeh per 100 g?"
    result = generate_verified_answer(
        question,
        process_query(question),
        [evidence],
        ConstraintResult(allowed=[evidence], excluded=[], warnings=[]),
    )

    assert "Protein: 18 g" in result.answer
    assert "190 kcal" not in result.answer
    assert [citation.evidence_id for citation in result.citations] == ["food-1"]
    assert result.warnings
