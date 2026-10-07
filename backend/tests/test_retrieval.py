from app.services.evidence import Evidence
from app.services.query_processor import process_query
from app.services.retrieval import (
    _RetrievalIndex,
    _bm25_rank,
    _direct_food_matches,
    invalidate_retrieval_cache,
    retrieve_evidence,
)
from app.core.database import Base
from app.models.knowledge import DocumentChunk, FoodNutrition, KnowledgeSource
from app.services import retrieval
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool


class _FixedScores:
    def get_scores(self, query_tokens: list[str]) -> list[float]:
        assert query_tokens == ["tempeh"]
        return [0.1, 0.9]


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


def test_bm25_rank_returns_evidence_for_positive_scores() -> None:
    corpus = (
        _food("food-1", "Lentils"),
        _food("food-2", "Tempeh"),
    )
    index = _RetrievalIndex(
        corpus=corpus,
        tokenized_corpus=(("lentils",), ("tempeh",)),
        bm25=_FixedScores(),
    )

    ranked = _bm25_rank("tempeh", index, limit=2)

    assert [(item.evidence_id, score) for item, score in ranked] == [
        ("food-2", 0.9),
        ("food-1", 0.1),
    ]


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


def test_food_match_requires_the_name_phrase_in_the_current_question() -> None:
    foods = [
        _food("food-green", "Green Banana"),
        _food("food-banana", "Banana"),
    ]

    matches = _direct_food_matches("What is the serving size of banana?", foods)

    assert [item.metadata["name"] for item in matches] == ["Banana"]


def test_matching_supports_plural_form_and_aliases() -> None:
    corpus = [
        _food("food-1", "Nasi Lemak"),
        Evidence(
            evidence_id="food-2",
            text="Food: Fried Chicken",
            source_id=1,
            source_title="Test dataset",
            source_url=None,
            source_type="user_uploaded_csv",
            authority_score=0.3,
            kind="food",
            metadata={
                "name": "Fried Chicken",
                "aliases": "ayam goreng",
                "category": "Food",
            },
        ),
    ]

    assert _direct_food_matches("Calories in nasi lemaks?", corpus)[0].evidence_id == "food-1"
    assert _direct_food_matches("Protein in ayam goreng?", corpus)[0].evidence_id == "food-2"
    assert _direct_food_matches("Protein in chicken?", corpus) == []


def test_matching_uses_parenthetical_english_food_name() -> None:
    food = _food("food-ayam-goreng", "Ayam Goreng (Fried Chicken)")

    matches = _direct_food_matches(
        "What is the sodium content of fried chicken?",
        [food],
    )

    assert [item.evidence_id for item in matches] == ["food-ayam-goreng"]


def test_retrieval_caches_corpus_and_bm25_until_invalidated(
    monkeypatch,
) -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(
        engine,
        tables=[
            KnowledgeSource.__table__,
            DocumentChunk.__table__,
            FoodNutrition.__table__,
        ],
    )
    with Session(engine) as db:
        source = KnowledgeSource(
            title="cache test",
            source_type="user_uploaded_json",
            authority_score=0.3,
        )
        db.add(source)
        db.flush()
        db.add(
            FoodNutrition(
                source_id=source.id,
                source_record_id="tempeh",
                name="Tempeh",
                protein_g=18,
            )
        )
        db.commit()

        original_loader = retrieval._load_corpus
        calls = 0

        def counting_loader(session):
            nonlocal calls
            calls += 1
            return original_loader(session)

        monkeypatch.setattr(retrieval, "_load_corpus", counting_loader)
        monkeypatch.setattr(retrieval, "search_dense", lambda *_args: [])
        invalidate_retrieval_cache()
        plan = process_query("How much protein is in tempeh?")

        first = retrieve_evidence(db, plan)
        first[0].text = "request-local mutation"
        second = retrieve_evidence(db, plan)

        assert calls == 1
        assert second[0].text != "request-local mutation"

        invalidate_retrieval_cache()
        retrieve_evidence(db, plan)
        assert calls == 2
    engine.dispose()
