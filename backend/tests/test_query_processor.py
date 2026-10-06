from app.services.query_processor import process_query


def test_routes_numeric_nutrition_question_to_structured_lookup() -> None:
    plan = process_query("How much protein is in tempeh per 100 g?")

    assert plan.intent == "structured_lookup"
    assert plan.structured_lookup is True
    assert plan.recommendation_requested is False


def test_personalized_recommendation_extracts_profile_constraints() -> None:
    plan = process_query(
        "What high-protein foods can I eat?",
        dietary_preference="Vegetarian",
        allergies="peanuts",
    )

    assert plan.intent == "personalized_recommendation"
    assert plan.constraints.diets == ("vegetarian",)
    assert plan.constraints.allergens == ("peanut",)
    assert len(plan.queries) > 1


def test_guideline_question_uses_document_route() -> None:
    plan = process_query(
        "What do Malaysian dietary guidelines recommend about fruit intake?"
    )

    assert plan.intent == "document_evidence"
    assert plan.document_lookup is True


def test_negative_allergy_statement_does_not_create_constraint() -> None:
    plan = process_query("I am not allergic to peanuts.")

    assert plan.constraints.allergens == ()


def test_long_question_is_decomposed_into_subquestions() -> None:
    plan = process_query(
        "I am vegetarian and moderately active, and want to lose weight "
        "without too many calories."
    )

    assert plan.subquestions


def test_terminal_command_is_routed_outside_dietary_retrieval() -> None:
    plan = process_query("uvicorn app.main:app --reload --reload-dir app")

    assert plan.intent == "out_of_scope"
    assert plan.structured_lookup is False
    assert plan.document_lookup is False
