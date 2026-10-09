from dataclasses import replace

import pytest
from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.core.database import Base
from app.models.knowledge import DocumentChunk, FoodNutrition, KnowledgeSource
from app.services.constraints import NumericConstraint
from app.services.evidence import Evidence
from app.services.query_processor import process_query
from app.services.retrieval import (
    _RetrievalIndex,
    _bm25_rank,
    _direct_food_matches,
    _reciprocal_rank_fusion,
    invalidate_retrieval_cache,
    retrieve_evidence,
)
from app.services import retrieval


@pytest.fixture
def nutrition_db(monkeypatch):
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
            title="Numeric filter test data",
            source_type="user_uploaded_csv",
            authority_score=0.7,
        )
        db.add(source)
        db.flush()
        db.add_all(
            [
                FoodNutrition(
                    source_id=source.id,
                    source_record_id="food-a",
                    name="Food A",
                    calories=250,
                    protein_g=15,
                    sodium_mg=400,
                ),
                FoodNutrition(
                    source_id=source.id,
                    source_record_id="food-b",
                    name="Food B",
                    calories=280,
                    protein_g=8,
                    sodium_mg=500,
                ),
                FoodNutrition(
                    source_id=source.id,
                    source_record_id="food-c",
                    name="Food C",
                    calories=310,
                    protein_g=12,
                    sodium_mg=250,
                ),
                FoodNutrition(
                    source_id=source.id,
                    source_record_id="food-missing-protein",
                    name="Food Missing Protein",
                    calories=200,
                    protein_g=None,
                    sodium_mg=300,
                ),
                FoodNutrition(
                    source_id=source.id,
                    source_record_id="chicken",
                    name="Chicken",
                    calories=220,
                    protein_g=20,
                    sodium_mg=450,
                ),
            ]
        )
        db.commit()
        monkeypatch.setattr(retrieval, "search_dense", lambda *_args: [])
        invalidate_retrieval_cache()
        try:
            yield db
        finally:
            invalidate_retrieval_cache()
    engine.dispose()


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


def test_bm25_kind_filter_applies_before_candidate_limit() -> None:
    foods = (_food("food-a", "Tempeh"), _food("food-b", "Tempeh soup"))
    document = Evidence(
        evidence_id="chunk-guideline",
        text="Tempeh dietary guidance.",
        source_id=1,
        source_title="Guidance",
        source_url=None,
        source_type="guideline",
        authority_score=0.8,
        kind="document",
    )

    class Scores:
        def get_scores(self, _query_tokens: list[str]) -> list[float]:
            return [0.9, 0.8, 0.1]

    index = _RetrievalIndex(
        corpus=(*foods, document),
        tokenized_corpus=(("tempeh",), ("tempeh", "soup"), ("tempeh", "guidance")),
        bm25=Scores(),
    )

    ranked = _bm25_rank("tempeh", index, limit=1, kind="document")

    assert [item.evidence_id for item, _ in ranked] == ["chunk-guideline"]


def test_reciprocal_rank_fusion_deduplicates_and_uses_standard_scores() -> None:
    first = _food("food-a", "Food A")
    shared = _food("food-b", "Food B")
    last = _food("food-c", "Food C")

    fused = _reciprocal_rank_fusion(
        [
            [(first, 0.1), (shared, 0.9)],
            [(shared, 0.2), (last, 0.8)],
        ]
    )

    assert [item.evidence_id for item in fused] == [
        "food-b",
        "food-a",
        "food-c",
    ]
    assert fused[0].retrieval_score == 0.9
    assert fused[0].rrf_score == pytest.approx(1 / 62 + 1 / 61)
    assert fused[0] is not shared


def test_reciprocal_rank_fusion_handles_empty_rankings_deterministically() -> None:
    food_b = _food("food-b", "Food B")
    food_a = _food("food-a", "Food A")

    fused = _reciprocal_rank_fusion([[], [(food_b, 0.2), (food_a, 0.1)]])

    assert [item.evidence_id for item in fused] == ["food-b", "food-a"]
    assert _reciprocal_rank_fusion([[], []]) == []


def test_reranker_uses_cross_encoder_scores_and_handles_empty_candidates(
    monkeypatch,
) -> None:
    high_authority = _food("food-a", "Food A")
    high_authority.authority_score = 1.0
    lower_authority = _food("food-b", "Food B")

    class FixedReranker:
        def predict(self, pairs, *, show_progress_bar):
            assert pairs == [
                ("question", high_authority.text[:4000]),
                ("question", lower_authority.text[:4000]),
            ]
            assert show_progress_bar is False
            return [0.1, 0.9]

    monkeypatch.setattr(
        retrieval,
        "_get_reranker",
        lambda: FixedReranker(),
    )

    assert retrieval.rerank_evidence(
        "question",
        [high_authority, lower_authority],
    ) == [lower_authority, high_authority]
    assert retrieval.rerank_evidence("question", []) == []


def test_dense_retrieval_runs_when_the_sql_corpus_is_empty(
    nutrition_db,
    monkeypatch,
) -> None:
    document = Evidence(
        evidence_id="chunk-1",
        text="Dietary guidance explains balanced meals.",
        source_id=1,
        source_title="Guidance",
        source_url=None,
        source_type="guideline",
        authority_score=0.8,
        kind="document",
    )
    monkeypatch.setattr(retrieval, "_load_corpus", lambda _db: [])
    invalidate_retrieval_cache()
    monkeypatch.setattr(
        retrieval,
        "search_dense",
        lambda *_args, **_kwargs: [(document, 0.9)],
    )
    plan = process_query("Explain dietary guidelines")

    results = retrieve_evidence(nutrition_db, plan)

    assert [item.evidence_id for item in results] == ["chunk-1"]


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


def test_comparison_retrieves_each_named_food_before_reranking(
    nutrition_db,
    monkeypatch,
) -> None:
    source = nutrition_db.scalar(select(KnowledgeSource))
    assert source is not None
    foods = [
        FoodNutrition(
            source_id=source.id,
            source_record_id=record_id,
            name=name,
        )
        for record_id, name in (
            ("1", "Nasi Lemak"),
            ("2", "Nasi Goreng Kampung"),
        )
    ]
    nutrition_db.add_all(foods)
    nutrition_db.commit()
    invalidate_retrieval_cache()
    monkeypatch.setattr(retrieval, "_bm25_rank", lambda *_args, **_kwargs: [])

    plan = process_query(
        "Compare the calories in Nasi Lemak and Nasi Goreng Kampung."
    )
    candidates = retrieve_evidence(
        nutrition_db,
        plan,
        candidate_limit=2,
    )

    candidate_keys = {
        f"food:{item.metadata['source_record_id']}"
        for item in candidates
    }
    assert plan.intent == "comparison"
    assert candidate_keys == {"food:1", "food:2"}

    reranker_inputs = []

    class FixedReranker:
        def predict(self, pairs, *, show_progress_bar):
            assert show_progress_bar is False
            reranker_inputs.extend(pairs)
            return [0.9 if "Nasi Lemak" in text else 0.8 for _, text in pairs]

    monkeypatch.setattr(retrieval, "_get_reranker", lambda: FixedReranker())
    reranked = retrieval.rerank_evidence(
        plan.queries[0],
        candidates,
        limit=2,
    )

    assert {text for _, text in reranker_inputs} == {
        item.text for item in candidates
    }
    assert {
        f"food:{item.metadata['source_record_id']}"
        for item in reranked
    } == {"food:1", "food:2"}


def test_comparison_does_not_direct_match_unknown_food_names(
    nutrition_db,
    monkeypatch,
) -> None:
    source = nutrition_db.scalar(select(KnowledgeSource))
    assert source is not None
    nutrition_db.add_all(
        [
            FoodNutrition(
                source_id=source.id,
                source_record_id=record_id,
                name=name,
            )
            for record_id, name in (
                ("1", "Nasi Lemak"),
                ("2", "Nasi Goreng Kampung"),
            )
        ]
    )
    nutrition_db.commit()
    invalidate_retrieval_cache()
    monkeypatch.setattr(retrieval, "_bm25_rank", lambda *_args, **_kwargs: [])

    plan = process_query(
        "Compare calories in Unknown Dish Alpha and Unknown Dish Beta."
    )
    results = retrieve_evidence(
        nutrition_db,
        plan,
        candidate_limit=5,
    )

    assert plan.intent == "comparison"
    assert not {
        item.metadata.get("source_record_id")
        for item in results
    }.intersection({"1", "2"})


@pytest.mark.parametrize(
    ("question", "expected_intent"),
    [
        ("Which foods are relatively high in fiber?", "semantic_search"),
        (
            "What do the dietary guidelines say about balanced meals?",
            "document_evidence",
        ),
    ],
)
def test_semantic_and_guideline_queries_keep_existing_document_retrieval(
    nutrition_db,
    monkeypatch,
    question,
    expected_intent,
) -> None:
    source = nutrition_db.scalar(select(KnowledgeSource))
    assert source is not None
    chunk = DocumentChunk(
        source_id=source.id,
        chunk_index=1,
        content="Dietary guidance discusses balanced meals and fiber.",
        extra_data={},
    )
    nutrition_db.add(chunk)
    nutrition_db.commit()
    document = retrieval.chunk_evidence(chunk)
    direct_match_calls = []
    original_direct_matches = retrieval._direct_food_matches

    def track_direct_matches(query, corpus):
        direct_match_calls.append(query)
        return original_direct_matches(query, corpus)

    monkeypatch.setattr(retrieval, "_direct_food_matches", track_direct_matches)
    monkeypatch.setattr(retrieval, "_bm25_rank", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        retrieval,
        "search_dense",
        lambda *_args, **_kwargs: [(document, 0.9)],
    )

    plan = process_query(question)
    results = retrieve_evidence(nutrition_db, plan)

    assert plan.intent == expected_intent
    assert [item.evidence_id for item in results] == [document.evidence_id]
    assert direct_match_calls == []


def test_numeric_filter_applies_less_than_constraint(nutrition_db) -> None:
    plan = process_query("Find foods with less than 300 calories")

    results = retrieve_evidence(nutrition_db, plan)

    assert {item.metadata["name"] for item in results} == {
        "Food A",
        "Food B",
        "Food Missing Protein",
        "Chicken",
    }
    assert all(item.metadata["source_record_id"] for item in results)


def test_numeric_filter_applies_greater_than_or_equal_constraint(
    nutrition_db,
) -> None:
    plan = process_query("Find foods with at least 10g protein")

    results = retrieve_evidence(nutrition_db, plan)

    assert {item.metadata["name"] for item in results} == {
        "Food A",
        "Food C",
        "Chicken",
    }


def test_multiple_numeric_filters_use_and_logic(nutrition_db) -> None:
    plan = process_query(
        "Find foods with less than 300 calories and at least 10g protein"
    )

    results = retrieve_evidence(nutrition_db, plan)

    assert [item.metadata["name"] for item in results] == ["Food A", "Chicken"]


def test_numeric_filter_excludes_record_missing_required_value(
    nutrition_db,
) -> None:
    plan = process_query("Find foods with at least 10g protein")

    results = retrieve_evidence(nutrition_db, plan)

    assert "Food Missing Protein" not in {
        item.metadata["name"] for item in results
    }


def test_numeric_filter_applies_candidate_limit_in_sql(nutrition_db) -> None:
    statements = []

    def record_statement(_connection, _cursor, statement, _parameters, _context, _many):
        if "FROM food_nutrition" in statement:
            statements.append(statement)

    event.listen(nutrition_db.get_bind(), "before_cursor_execute", record_statement)
    try:
        results = retrieve_evidence(
            nutrition_db,
            process_query("Find foods with less than 400 calories"),
            candidate_limit=2,
        )
    finally:
        event.remove(
            nutrition_db.get_bind(),
            "before_cursor_execute",
            record_statement,
        )

    assert len(results) == 2
    assert len(statements) == 1
    assert "LIMIT" in statements[0].upper()


def test_numeric_limit_scans_past_incompatible_foods(nutrition_db, monkeypatch) -> None:
    source = nutrition_db.scalar(select(KnowledgeSource))
    assert source is not None
    compatible = FoodNutrition(
        source_id=source.id,
        source_record_id="later-vegetarian",
        name="Later vegetarian food",
        calories=100,
        dietary_labels=["vegetarian"],
        allergens=[],
    )
    nutrition_db.add(compatible)
    nutrition_db.commit()

    original_food_evidence = retrieval.food_evidence

    def reviewed_food_evidence(food):
        evidence = original_food_evidence(food)
        if food.source_record_id == "later-vegetarian":
            evidence.metadata["dietary_data_reviewed"] = True
            evidence.metadata["allergen_data_reviewed"] = True
            evidence.metadata["dietary_data_verified"] = True
            evidence.metadata["allergen_data_verified"] = True
        return evidence

    monkeypatch.setattr(retrieval, "food_evidence", reviewed_food_evidence)
    plan = process_query(
        "Find vegetarian foods below 300 calories",
        dietary_preference="vegetarian",
    )

    results = retrieve_evidence(nutrition_db, plan, candidate_limit=1)

    assert [item.metadata["name"] for item in results] == [
        "Later vegetarian food"
    ]


def test_numeric_limit_retains_incompatible_foods_for_safe_refusal(
    nutrition_db,
) -> None:
    plan = process_query(
        "Find foods below 300 calories",
        allergies="peanuts",
    )

    results = retrieve_evidence(nutrition_db, plan, candidate_limit=1)

    assert len(results) == 1
    constraint_result = retrieval.enforce_constraints(results, plan)
    assert constraint_result.allowed == []
    assert constraint_result.excluded == results
    assert constraint_result.allergen_status_unknown


def test_food_evidence_does_not_confuse_populated_with_reviewed_fields(
    nutrition_db,
) -> None:
    food = nutrition_db.scalar(
        select(FoodNutrition).where(
            FoodNutrition.source_record_id == "food-a"
        )
    )
    assert food is not None
    food.allergens = []
    food.dietary_labels = ["vegetarian"]

    evidence = retrieval.food_evidence(food)

    assert evidence.metadata["allergen_data_populated"] is True
    assert evidence.metadata["dietary_data_populated"] is True
    assert evidence.metadata["allergen_data_reviewed"] is False
    assert evidence.metadata["dietary_data_reviewed"] is False
    assert evidence.metadata["allergen_data_verified"] is False
    assert evidence.metadata["dietary_data_verified"] is False


def test_numeric_only_search_does_not_retrieve_documents(
    nutrition_db,
    monkeypatch,
) -> None:
    def unexpected_document_search(*_args):
        raise AssertionError("Numeric-only searches must not retrieve documents.")

    monkeypatch.setattr(retrieval, "search_dense", unexpected_document_search)

    results = retrieve_evidence(
        nutrition_db,
        process_query("Find foods with less than 300 calories"),
    )

    assert results
    assert all(item.kind == "food" for item in results)


def test_numeric_hybrid_search_returns_foods_and_document_evidence(
    nutrition_db,
    monkeypatch,
) -> None:
    source = nutrition_db.scalar(select(KnowledgeSource))
    assert source is not None
    chunk = DocumentChunk(
        source_id=source.id,
        chunk_index=1,
        content="Dietary guidelines recommend choosing a variety of foods.",
        extra_data={},
    )
    nutrition_db.add(chunk)
    nutrition_db.commit()
    document = Evidence(
        evidence_id=f"chunk-{chunk.id}",
        text=chunk.content,
        source_id=source.id,
        source_title=source.title,
        source_url=None,
        source_type=source.source_type,
        authority_score=source.authority_score,
        kind="document",
    )
    monkeypatch.setattr(
        retrieval,
        "search_dense",
        lambda *_args, **kwargs: (
            [(document, 0.9)]
            if kwargs.get("kind") == "document"
            else []
        ),
    )
    invalidate_retrieval_cache()
    plan = process_query(
        "Find foods below 300 calories and explain the dietary guidelines "
        "for choosing them."
    )

    results = retrieve_evidence(nutrition_db, plan, candidate_limit=2)

    assert plan.intent == "hybrid_question"
    assert len(results) <= 2
    assert any(item.kind == "food" for item in results)
    assert any(item.kind == "document" for item in results)
    assert [item.evidence_id for item in results].count(document.evidence_id) == 1
    assert document.evidence_id in {item.evidence_id for item in results}


def test_numeric_filter_supports_equality_and_inclusive_operators(
    nutrition_db,
) -> None:
    equal_plan = process_query("Find foods with exactly 250 calories")
    maximum_plan = process_query("Find foods with at most 250 calories")
    minimum_plan = process_query("Find foods with at least 250 calories")

    equal = retrieve_evidence(nutrition_db, equal_plan)
    maximum = retrieve_evidence(nutrition_db, maximum_plan)
    minimum = retrieve_evidence(nutrition_db, minimum_plan)

    assert [item.metadata["name"] for item in equal] == ["Food A"]
    assert {item.metadata["name"] for item in maximum} == {
        "Food A",
        "Food Missing Protein",
        "Chicken",
    }
    assert {item.metadata["name"] for item in minimum} == {
        "Food A",
        "Food B",
        "Food C",
    }


def test_numeric_range_filters_with_both_inclusive_bounds(nutrition_db) -> None:
    plan = process_query("Find foods between 250 and 280 calories")

    results = retrieve_evidence(nutrition_db, plan)

    assert {item.metadata["name"] for item in results} == {"Food A", "Food B"}


def test_unsupported_numeric_field_fails_explicitly(nutrition_db) -> None:
    plan = process_query("Find foods with less than 300 calories")
    invalid_constraints = replace(
        plan.constraints,
        numeric_constraints=(
            NumericConstraint(field="private_column", operator="<", value=5),
        ),
    )
    invalid_plan = replace(plan, constraints=invalid_constraints)

    with pytest.raises(ValueError, match="Unsupported numeric nutrition field"):
        retrieve_evidence(nutrition_db, invalid_plan)


def test_no_numeric_constraints_preserve_existing_food_lookup(
    nutrition_db,
) -> None:
    plan = process_query("How much protein is in Chicken?")

    results = retrieve_evidence(nutrition_db, plan)

    assert "Chicken" in {item.metadata["name"] for item in results}


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
