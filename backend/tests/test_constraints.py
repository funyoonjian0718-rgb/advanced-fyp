from app.services.constraints import enforce_constraints
from app.services.evidence import Evidence
from app.services.query_processor import process_query


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
