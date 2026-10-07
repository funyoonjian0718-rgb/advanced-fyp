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


def test_serving_size_lookup_chooses_exact_food_and_skips_model(
    monkeypatch,
) -> None:
    apple = Evidence(
        evidence_id="food-apple",
        text="Food: Apple\nServing: 1 medium apple\nCalories: 95 kcal",
        source_id=1,
        source_title="Nutrition record",
        source_url=None,
        source_type="user_uploaded_json",
        authority_score=0.7,
        kind="food",
        metadata={"name": "Apple"},
    )
    green_banana = Evidence(
        evidence_id="food-green-banana",
        text="Food: Green Banana\nServing: 1 small banana\nCalories: 90 kcal",
        source_id=1,
        source_title="Nutrition record",
        source_url=None,
        source_type="user_uploaded_json",
        authority_score=0.7,
        kind="food",
        metadata={"name": "Green Banana"},
    )
    banana = Evidence(
        evidence_id="food-banana",
        text="Food: Banana\nServing: 1 medium banana (118 g)\nCalories: 105 kcal",
        source_id=1,
        source_title="Nutrition record",
        source_url=None,
        source_type="user_uploaded_json",
        authority_score=0.7,
        kind="food",
        metadata={"name": "Banana"},
    )

    def unexpected_model_call(_prompt: str):
        raise AssertionError("Structured lookup must not call the LLM.")

    monkeypatch.setattr("app.services.generation._generate", unexpected_model_call)
    question = "What is the serving size of banana?"
    result = generate_verified_answer(
        question,
        process_query(question),
        [apple, green_banana, banana],
        ConstraintResult(allowed=[apple, green_banana, banana], excluded=[], warnings=[]),
    )

    assert "Banana: Serving: 1 medium banana (118 g)." in result.answer
    assert [citation.evidence_id for citation in result.citations] == ["food-banana"]


def test_structured_lookup_does_not_fall_back_to_llm_for_unmatched_food(
    monkeypatch,
) -> None:
    evidence = Evidence(
        evidence_id="food-apple",
        text="Food: Apple\nServing: 1 medium apple\nSodium: 2 mg",
        source_id=1,
        source_title="Nutrition record",
        source_url=None,
        source_type="user_uploaded_json",
        authority_score=0.7,
        kind="food",
        metadata={"name": "Apple"},
    )

    def unexpected_model_call(_prompt: str):
        raise AssertionError("Unmatched structured queries must not call the LLM.")

    monkeypatch.setattr("app.services.generation._generate", unexpected_model_call)
    question = "What is the sodium content of fried chicken?"
    result = generate_verified_answer(
        question,
        process_query(question),
        [evidence],
        ConstraintResult(allowed=[evidence], excluded=[], warnings=[]),
    )

    assert "exact nutrition record" in result.answer
    assert result.citations == []


def test_structured_lookup_returns_the_requested_chicken_sodium_value(
    monkeypatch,
) -> None:
    evidence = Evidence(
        evidence_id="food-chicken",
        text=(
            "Food: Fried Chicken\nServing: 1 piece (100 g)\n"
            "Sodium: 620 mg\nProtein: 21 g"
        ),
        source_id=1,
        source_title="Nutrition record",
        source_url=None,
        source_type="user_uploaded_json",
        authority_score=0.7,
        kind="food",
        metadata={"name": "Fried Chicken"},
    )

    def unexpected_model_call(_prompt: str):
        raise AssertionError("Structured lookup must not call the LLM.")

    monkeypatch.setattr("app.services.generation._generate", unexpected_model_call)
    question = "What is the sodium content of fried chicken?"
    result = generate_verified_answer(
        question,
        process_query(question),
        [evidence],
        ConstraintResult(allowed=[evidence], excluded=[], warnings=[]),
    )

    assert "Fried Chicken: Sodium: 620 mg" in result.answer
    assert "Protein" not in result.answer
