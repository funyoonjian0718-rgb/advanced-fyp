from app.services.query_processor import process_query


def test_routes_numeric_nutrition_question_to_structured_lookup() -> None:
    plan = process_query("How much protein is in tempeh per 100 g?")

    assert plan.intent == "structured_lookup"
    assert plan.structured_lookup is True
    assert plan.recommendation_requested is False


def test_kcal_question_routes_to_structured_lookup() -> None:
    plan = process_query("How many kcal are in a banana?")

    assert plan.intent == "structured_lookup"


def test_nutrition_information_request_routes_to_structured_lookup() -> None:
    plan = process_query("Show the nutrition information for plain yogurt.")

    assert plan.intent == "structured_lookup"


def test_serving_size_question_routes_to_structured_lookup() -> None:
    plan = process_query("What is the serving size of banana?")

    assert plan.intent == "structured_lookup"


def test_informational_food_group_question_is_not_a_recommendation() -> None:
    plan = process_query("What are food groups?")

    assert plan.intent == "general_dietary_question"
    assert plan.recommendation_requested is False


def test_guideline_question_about_nutrition_is_document_evidence() -> None:
    plan = process_query("Why do nutrition guidelines discuss food groups?")

    assert plan.intent == "document_evidence"
    assert plan.document_lookup is True


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


def test_suggested_meal_is_detected_as_recommendation() -> None:
    plan = process_query("Suggest a few meals for my dietary preferences.")

    assert plan.intent == "personalized_recommendation"
    assert plan.recommendation_requested is True


def test_meals_to_consider_are_detected_as_recommendation() -> None:
    plan = process_query("What meals should I consider if I am vegan?")

    assert plan.intent == "personalized_recommendation"


def test_guideline_question_uses_document_route() -> None:
    plan = process_query(
        "What do Malaysian dietary guidelines recommend about fruit intake?"
    )

    assert plan.intent == "document_evidence"
    assert plan.document_lookup is True


def test_negative_allergy_statement_does_not_create_constraint() -> None:
    plan = process_query("I am not allergic to peanuts.")

    assert plan.constraints.allergens == ()
    assert plan.constraints.unknown_allergen_constraint is False


def test_allergy_negation_variants_do_not_create_constraints() -> None:
    for message in (
        "I don't have a peanut allergy.",
        "I have no allergy to peanuts.",
        "I am not allergic to peanuts.",
    ):
        plan = process_query(message)
        assert plan.constraints.allergens == ()
        assert plan.constraints.unknown_allergen_constraint is False


def test_unrecognized_allergy_and_dietary_restriction_are_flagged() -> None:
    allergy_plan = process_query(
        "What foods can I eat? I have a mustard allergy."
    )
    diet_plan = process_query(
        "What low-carb foods can I eat?"
    )

    assert allergy_plan.constraints.unknown_allergen_constraint is True
    assert diet_plan.constraints.unknown_dietary_constraint is True


def test_known_allergy_in_question_does_not_become_an_unknown_constraint() -> None:
    plan = process_query("What foods can I eat with peanut allergies?")

    assert plan.constraints.allergens == ("peanut",)
    assert plan.constraints.unknown_allergen_constraint is False


def test_celiac_is_treated_as_a_gluten_constraint() -> None:
    plan = process_query("What can I eat? I have celiac disease.")

    assert plan.constraints.allergens == ("gluten",)
    assert plan.constraints.unknown_allergen_constraint is False


def test_medical_dietary_condition_is_flagged_as_unknown() -> None:
    plan = process_query("What foods can I eat? I have diabetes.")

    assert plan.constraints.unknown_dietary_constraint is True


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
