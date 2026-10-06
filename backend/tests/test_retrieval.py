from app.services.evidence import Evidence
from app.services.retrieval import _direct_food_matches


def _food(evidence_id: str, name: str) -> Evidence:
    return Evidence(
        evidence_id=evidence_id,
        text=f"Food: {name}\nProtein: 20 g",
        source_id=1,
        source_title="Test dataset",
        source_url=None,
        source_type="user_uploaded_csv",
        authority_score=0.3,
        kind="food",
        metadata={"name": name, "aliases": "", "category": "Food"},
    )


def test_structured_lookup_prioritizes_exact_food_name() -> None:
    foods = [
        _food("food-1", "Tempe"),
        _food("food-2", "Tempeh Goreng"),
        _food("food-3", "Tempeh"),
    ]

    matches = _direct_food_matches(
        "How much protein is in tempeh per 100 g?",
        foods,
    )

    assert [item.metadata["name"] for item in matches] == ["Tempeh"]
