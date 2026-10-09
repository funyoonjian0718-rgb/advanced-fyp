from pathlib import Path

import pytest

from app.scripts.evaluate_retrieval import (
    _dense_food_index_available,
    _dense_baseline,
    _mean_metrics,
    _metrics,
    load_cases,
)


def test_retrieval_metrics_match_a_known_ranking() -> None:
    result = _metrics(
        ["food:1", "food:2", "chunk:1"],
        {"food:1", "food:2"},
        top_k=2,
    )

    assert result == {
        "precision_at_k": 1.0,
        "recall_at_k": 1.0,
        "mrr": 1.0,
        "ndcg_at_k": 1.0,
    }


def test_metrics_deduplicate_repeated_retrieved_evidence() -> None:
    result = _metrics(
        ["gold", "gold", "other"],
        {"gold"},
        top_k=2,
    )

    assert result == {
        "precision_at_k": 0.5,
        "recall_at_k": 1.0,
        "mrr": 1.0,
        "ndcg_at_k": 1.0,
    }


def test_empty_ranking_returns_zero_metrics() -> None:
    assert _metrics([], {"gold"}, top_k=3) == {
        "precision_at_k": 0.0,
        "recall_at_k": 0.0,
        "mrr": 0.0,
        "ndcg_at_k": 0.0,
    }


def test_no_relevant_documents_returns_zero_metrics() -> None:
    assert _metrics(["first", "second"], set(), top_k=2) == {
        "precision_at_k": 0.0,
        "recall_at_k": 0.0,
        "mrr": 0.0,
        "ndcg_at_k": 0.0,
    }


def test_single_relevant_document_uses_total_relevant_count_for_recall() -> None:
    result = _metrics(
        ["first", "second", "gold"],
        {"gold"},
        top_k=2,
    )

    assert result == {
        "precision_at_k": 0.0,
        "recall_at_k": 0.0,
        "mrr": 1 / 3,
        "ndcg_at_k": 0.0,
    }


def test_multiple_relevant_documents_are_counted_once_each() -> None:
    result = _metrics(
        ["gold-a", "gold-a", "other", "gold-b"],
        {"gold-a", "gold-b"},
        top_k=3,
    )

    assert result["precision_at_k"] == 2 / 3
    assert result["recall_at_k"] == 1.0
    assert result["mrr"] == 1.0
    assert all(0 <= value <= 1 for value in result.values())


def test_graded_ndcg_uses_sorted_ideal_gains_at_the_same_cutoff() -> None:
    grades = {"high": 3, "medium": 2, "low": 1}

    ideal = _metrics(["high", "medium", "low"], grades, top_k=2)
    reversed_order = _metrics(["low", "medium", "high"], grades, top_k=2)

    assert ideal["ndcg_at_k"] == 1.0
    assert 0 < reversed_order["ndcg_at_k"] < 1
    assert ideal["precision_at_k"] == reversed_order["precision_at_k"] == 1.0
    assert ideal["recall_at_k"] == reversed_order["recall_at_k"] == 2 / 3


def test_macro_metrics_are_unweighted_query_means() -> None:
    rows = [
        _metrics(["gold"], {"gold"}, top_k=1),
        _metrics([], {"gold"}, top_k=1),
    ]

    assert _mean_metrics(rows) == {
        "precision_at_k": 0.5,
        "recall_at_k": 0.5,
        "mrr": 0.5,
        "ndcg_at_k": 0.5,
    }


def test_metrics_reject_invalid_cutoff_or_relevance_grade() -> None:
    with pytest.raises(ValueError, match="top_k must be positive"):
        _metrics([], set(), top_k=0)
    with pytest.raises(ValueError, match="grades must be finite"):
        _metrics(["gold"], {"gold": 4}, top_k=1)


def test_retrieval_evaluation_dataset_has_labeled_question_categories() -> None:
    dataset = (
        Path(__file__).parent
        / "data"
        / "rag_retrieval_eval.jsonl"
    )

    cases = load_cases(dataset)
    assert len(cases) == 16

    categories = {case["category"] for case in cases}
    assert {
        "exact_nutrition_lookup",
        "keyword_sensitive",
        "semantic_dietary",
        "food_comparison",
        "numeric_constraints",
        "dietary_allergen_safety",
        "guideline_based",
        "insufficient_evidence",
    } <= categories


def test_unlabeled_cases_are_empty_gold_sets_not_positive_labels() -> None:
    dataset = (
        Path(__file__).parent
        / "data"
        / "rag_retrieval_eval.jsonl"
    )

    cases = load_cases(dataset)
    skipped = [
        case
        for case in cases
        if not case["relevant_evidence_keys"]
    ]

    assert [
        (case["category"], case["question"])
        for case in skipped
    ] == [
        (
            "dietary_allergen_safety",
            "Which foods can I eat if I have a peanut allergy?",
        ),
        (
            "insufficient_evidence",
            "What is the exact vitamin B12 content of moon dust?",
        ),
    ]


def test_loader_accepts_valid_optional_graded_relevance(tmp_path: Path) -> None:
    dataset = tmp_path / "graded.jsonl"
    dataset.write_text(
        '{"category":"graded","question":"q",'
        '"relevant_evidence_keys":["a","b"],'
        '"relevance_grades":{"a":3,"b":1},'
        '"baseline_comparable":true,"answerable":true}\n',
        encoding="utf-8",
    )

    cases = load_cases(dataset)

    assert cases[0]["relevance_grades"] == {"a": 3, "b": 1}


def test_dense_baseline_applies_structured_evidence_kind(monkeypatch) -> None:
    from app.services.evidence import Evidence
    from app.services.query_processor import process_query

    food = Evidence(
        evidence_id="food-1",
        text="Food: Nasi Lemak",
        source_id=1,
        source_title="Food records",
        source_url=None,
        source_type="nutrition",
        authority_score=0.7,
        kind="food",
    )
    document = Evidence(
        evidence_id="chunk-1",
        text="Guideline",
        source_id=1,
        source_title="Guidelines",
        source_url=None,
        source_type="guideline",
        authority_score=0.7,
        kind="document",
    )
    calls = []

    def fake_search(query, limit, *, kind=None):
        calls.append((query, limit, kind))
        return [(food, 0.9), (document, 0.8)]

    monkeypatch.setattr(
        "app.scripts.evaluate_retrieval.search_dense",
        fake_search,
    )
    plan = process_query("How many calories are in Nasi Lemak?")

    results = _dense_baseline(plan, candidate_limit=12)

    assert calls == [(plan.queries[0], 12, "food")]
    assert [item.evidence_id for item in results] == ["food-1"]


@pytest.mark.parametrize(("food_count", "expected"), [(0, False), (3, True)])
def test_dense_food_index_availability_uses_qdrant_kind_count(
    monkeypatch,
    food_count,
    expected,
) -> None:
    from types import SimpleNamespace

    class FakeClient:
        def collection_exists(self, collection_name):
            assert collection_name == "dietary_evidence"
            return True

        def count(self, *, collection_name, count_filter, exact):
            assert collection_name == "dietary_evidence"
            assert count_filter.must[0].key == "kind"
            assert count_filter.must[0].match.value == "food"
            assert exact is True
            return SimpleNamespace(count=food_count)

    monkeypatch.setattr(
        "app.scripts.evaluate_retrieval.get_vector_client",
        lambda: FakeClient(),
    )

    assert _dense_food_index_available() is expected
