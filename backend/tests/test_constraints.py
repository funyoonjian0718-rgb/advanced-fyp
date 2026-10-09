from app.services.constraints import enforce_constraints
from app.services.evidence import Evidence
from app.services.query_processor import process_query
from app.services.constraints import extract_numeric_constraints

def _food(
    *,
    allergens: list[str] | None = None,
    dietary_labels: list[str] | None = None,
    name: str = "Lentil patty",
    ingredients: str = "lentils",
) -> Evidence:
    return Evidence(
        evidence_id="food-1",
        text=f"Food: {name}\nProtein: 12 g",
        source_id=1,
        source_title="nutrition dataset",
        source_url=None,
        source_type="user_uploaded_csv",
        authority_score=0.3,
        kind="food",
        metadata={
            "name": name,
            "aliases": "",
            "category": "Plant food",
            "ingredients": ingredients,
            "allergens": allergens,
            "dietary_labels": dietary_labels,
            "allergen_data_reviewed": allergens is not None,
            "dietary_data_reviewed": dietary_labels is not None,
            "allergen_data_verified": allergens is not None,
            "dietary_data_verified": dietary_labels is not None,
        },
    )


def test_unknown_allergen_status_is_not_recommended() -> None:
    plan = process_query(
        "What foods can I eat?",
        allergies="peanuts",
    )

    result = enforce_constraints([_food()], plan)

    assert result.allowed == []
    assert result.excluded
    assert any("allergen" in warning for warning in result.warnings)


def test_verified_conflicting_allergen_is_excluded() -> None:
    plan = process_query("What foods can I eat?", allergies="peanuts")
    evidence = _food(allergens=["peanut"], dietary_labels=["vegetarian"])

    result = enforce_constraints([evidence], plan)

    assert result.allowed == []
    assert result.excluded == [evidence]
    assert result.has_known_conflict


def test_food_with_verified_diet_and_no_allergens_is_allowed() -> None:
    plan = process_query(
        "What foods can I eat?",
        dietary_preference="vegetarian",
        allergies="peanut",
    )
    evidence = _food(allergens=[], dietary_labels=["vegetarian"])

    result = enforce_constraints([evidence], plan)

    assert result.allowed == [evidence]
    assert result.excluded == []


def test_non_dietary_label_does_not_conflict_with_vegetarian() -> None:
    plan = process_query(
        "What foods can I eat?",
        dietary_preference="vegetarian",
    )
    evidence = _food(dietary_labels=["high protein"])

    result = enforce_constraints([evidence], plan)

    assert result.allowed == []
    assert result.excluded == [evidence]
    assert any("unknown status" in warning for warning in result.warnings)


def test_explicit_meat_name_conflicts_with_vegetarian() -> None:
    plan = process_query(
        "What foods can I eat?",
        dietary_preference="vegetarian",
    )
    evidence = _food(
        dietary_labels=["high protein"],
        name="Chicken breast",
        ingredients="chicken",
    )

    result = enforce_constraints([evidence], plan)

    assert result.allowed == []
    assert result.excluded == [evidence]
    assert not any("unknown status" in warning for warning in result.warnings)


def test_verified_vegetarian_label_supports_food() -> None:
    plan = process_query(
        "What foods can I eat?",
        dietary_preference="vegetarian",
    )
    evidence = _food(dietary_labels=["vegetarian"])

    result = enforce_constraints([evidence], plan)

    assert result.allowed == [evidence]
    assert result.excluded == []


def test_missing_dietary_labels_are_unknown_and_not_recommended() -> None:
    plan = process_query(
        "What foods can I eat?",
        dietary_preference="vegetarian",
    )
    evidence = _food(name="Apple", ingredients="")

    result = enforce_constraints([evidence], plan)

    assert result.allowed == []
    assert result.excluded == [evidence]
    assert any("unknown status" in warning for warning in result.warnings)


def test_unknown_allergen_blocks_all_foods() -> None:
    plan = process_query(
        "What foods can I eat? I have a mustard allergy."
    )

    result = enforce_constraints([_food(allergens=[], dietary_labels=[])], plan)

    assert result.allowed == []
    assert result.excluded
    assert any("could not be identified" in warning for warning in result.warnings)


def test_numeric_search_enforces_diet_and_known_allergen_conflicts() -> None:
    plan = process_query(
        "Find foods with less than 300 calories",
        dietary_preference="vegetarian",
        allergies="peanuts",
    )
    safe_food = _food(
        name="Lentil patty",
        ingredients="lentils",
        allergens=[],
        dietary_labels=["vegetarian"],
    )
    conflicting_food = _food(
        name="Chicken",
        ingredients="chicken",
        allergens=["peanut"],
        dietary_labels=["vegetarian"],
    )

    result = enforce_constraints([safe_food, conflicting_food], plan)

    assert result.allowed == [safe_food]
    assert result.excluded == [conflicting_food]


def test_numeric_search_excludes_unknown_allergen_status() -> None:
    plan = process_query(
        "Find foods with less than 300 calories",
        allergies="peanuts",
    )

    result = enforce_constraints([_food(allergens=None)], plan)

    assert result.allowed == []
    assert len(result.excluded) == 1
    assert any("unknown allergen" in warning for warning in result.warnings)
    assert result.allergen_status_unknown


def test_extract_less_than_calories():
    constraints = extract_numeric_constraints(
        "Find foods with less than 300 calories"
    )

    assert len(constraints) == 1
    assert constraints[0].field == "calories"
    assert constraints[0].operator == "<"
    assert constraints[0].value == 300


def test_extract_at_least_protein():
    constraints = extract_numeric_constraints(
        "Find foods with at least 10g protein"
    )

    assert len(constraints) == 1
    assert constraints[0].field == "protein_g"
    assert constraints[0].operator == ">="
    assert constraints[0].value == 10


def test_extract_multiple_constraints():
    constraints = extract_numeric_constraints(
        "Find foods that have less than 300 calories "
        "and at least 10g protein"
    )

    assert len(constraints) == 2

    assert constraints[0].field == "calories"
    assert constraints[0].operator == "<"
    assert constraints[0].value == 300

    assert constraints[1].field == "protein_g"
    assert constraints[1].operator == ">="
    assert constraints[1].value == 10


def test_extract_field_first_constraint():
    constraints = extract_numeric_constraints(
        "Find foods with calories less than 300"
    )

    assert len(constraints) == 1
    assert constraints[0].field == "calories"
    assert constraints[0].operator == "<"
    assert constraints[0].value == 300


def test_extract_more_than_sodium():
    constraints = extract_numeric_constraints(
        "Foods with more than 500 mg sodium"
    )

    assert len(constraints) == 1
    assert constraints[0].field == "sodium_mg"
    assert constraints[0].operator == ">"
    assert constraints[0].value == 500


def test_extract_fibre_alias():
    constraints = extract_numeric_constraints(
        "Foods with at least 5g fibre"
    )

    assert len(constraints) == 1
    assert constraints[0].field == "fiber_g"
    assert constraints[0].operator == ">="
    assert constraints[0].value == 5

def test_extract_calorie_range():
    constraints = extract_numeric_constraints(
        "Find foods between 200 and 400 calories"
    )

    assert len(constraints) == 2

    assert constraints[0].field == "calories"
    assert constraints[0].operator == ">="
    assert constraints[0].value == 200

    assert constraints[1].field == "calories"
    assert constraints[1].operator == "<="
    assert constraints[1].value == 400