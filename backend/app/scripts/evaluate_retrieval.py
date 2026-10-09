import argparse
import json
import math
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from qdrant_client import models
from sqlalchemy.orm import Session

from app.core.database import SessionLocal
from app.services.evidence import Evidence
from app.services.query_processor import QueryPlan, process_query
from app.services.retrieval import (
    _bm25_rank,
    _load_retrieval_index,
    rerank_evidence,
    retrieve_evidence,
)
from app.services.vector_store import (
    COLLECTION_NAME,
    get_vector_client,
    search_dense,
)


DEFAULT_DATASET = (
    Path(__file__).resolve().parents[2]
    / "tests"
    / "data"
    / "rag_retrieval_eval.jsonl"
)
TOP_K = 8
CANDIDATE_LIMIT = 30
METRIC_NAMES = (
    "precision_at_k",
    "recall_at_k",
    "mrr",
    "ndcg_at_k",
)


def load_cases(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Retrieval evaluation dataset was not found: {path}")

    cases: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue
            try:
                record: Any = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Retrieval evaluation JSONL line {line_number} is invalid."
                ) from error
            if (
                not isinstance(record, dict)
                or not isinstance(record.get("category"), str)
                or not record["category"].strip()
                or not isinstance(record.get("question"), str)
                or not record["question"].strip()
                or not isinstance(record.get("relevant_evidence_keys"), list)
                or not all(
                    isinstance(key, str) and key.strip()
                    for key in record["relevant_evidence_keys"]
                )
                or len(set(record["relevant_evidence_keys"]))
                != len(record["relevant_evidence_keys"])
                or not isinstance(record.get("baseline_comparable"), bool)
                or not isinstance(record.get("answerable"), bool)
            ):
                raise ValueError(
                    f"Retrieval evaluation line {line_number} has invalid fields."
                )
            relevance_grades = record.get("relevance_grades")
            if relevance_grades is not None:
                if (
                    not isinstance(relevance_grades, dict)
                    or set(relevance_grades) != set(record["relevant_evidence_keys"])
                    or any(
                        isinstance(grade, bool)
                        or not isinstance(grade, (int, float))
                        or not math.isfinite(grade)
                        or not 1 <= grade <= 3
                        for grade in relevance_grades.values()
                    )
                ):
                    raise ValueError(
                        f"Retrieval evaluation line {line_number} has invalid "
                        "relevance_grades; provide a grade from 1 to 3 for every "
                        "relevant evidence key."
                    )
            cases.append(record)
    if not cases:
        raise ValueError("Retrieval evaluation dataset is empty.")
    return cases


def _evidence_key(item: Evidence) -> str | None:
    if item.kind == "food":
        record_id = item.metadata.get("source_record_id")
        return f"food:{record_id}" if record_id is not None else None
    if item.kind == "document":
        chunk_id = item.metadata.get("chunk_id")
        return f"chunk:{chunk_id}" if chunk_id is not None else None
    return None


def _metrics(
    ranked_keys: list[str],
    relevant_keys: set[str] | Mapping[str, int | float],
    *,
    top_k: int,
) -> dict[str, float]:
    if top_k < 1:
        raise ValueError("top_k must be positive.")

    grades = (
        dict(relevant_keys)
        if isinstance(relevant_keys, Mapping)
        else {key: 1 for key in relevant_keys}
    )
    if any(
        isinstance(grade, bool)
        or not isinstance(grade, (int, float))
        or not math.isfinite(grade)
        or not 1 <= grade <= 3
        for grade in grades.values()
    ):
        raise ValueError("Relevance grades must be finite values from 1 to 3.")

    # A ranked list is an ordering of unique evidence. Keep its first position.
    unique_ranked_keys = list(dict.fromkeys(ranked_keys))
    top_keys = unique_ranked_keys[:top_k]
    relevant_count = len(grades)
    hits = sum(key in grades for key in top_keys)
    reciprocal_rank = next(
        (
            1 / rank
            for rank, key in enumerate(unique_ranked_keys, start=1)
            if key in grades
        ),
        0.0,
    )
    dcg = sum(
        (2**grades[key] - 1) / math.log2(rank + 1)
        for rank, key in enumerate(top_keys, start=1)
        if key in grades
    )
    ideal_hits = sorted(grades.values(), reverse=True)[:top_k]
    ideal_dcg = sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(ideal_hits, start=1)
    )
    metrics = {
        "precision_at_k": hits / top_k,
        "recall_at_k": hits / relevant_count if relevant_count else 0.0,
        "mrr": reciprocal_rank,
        "ndcg_at_k": dcg / ideal_dcg if ideal_dcg else 0.0,
    }
    if any(
        not math.isfinite(value) or not 0 <= value <= 1
        for value in metrics.values()
    ):
        raise ArithmeticError("A retrieval metric fell outside the range [0, 1].")
    return metrics


def _mean_metrics(rows: list[dict[str, float]]) -> dict[str, float] | None:
    if not rows:
        return None
    return {
        name: sum(row[name] for row in rows) / len(rows)
        for name in METRIC_NAMES
    }


def _dense_food_index_available() -> bool:
    client = get_vector_client()
    if not client.collection_exists(COLLECTION_NAME):
        return False
    return (
        client.count(
            collection_name=COLLECTION_NAME,
            count_filter=models.Filter(
                must=[
                    models.FieldCondition(
                        key="kind",
                        match=models.MatchValue(value="food"),
                    )
                ]
            ),
            exact=True,
        ).count
        > 0
    )


def _bm25_baseline(
    plan: QueryPlan,
    *,
    db: Session,
    candidate_limit: int,
) -> list[Evidence]:
    index = _load_retrieval_index(db.get_bind())
    kind = (
        "food"
        if plan.intent == "structured_lookup"
        else "document"
        if plan.intent == "document_evidence"
        else None
    )
    ranked = [
        item
        for item, _ in _bm25_rank(
            plan.queries[0],
            index,
            candidate_limit,
            kind=kind,
        )
    ]
    return ranked


def _dense_baseline(
    plan: QueryPlan,
    *,
    candidate_limit: int,
) -> list[Evidence]:
    kind = (
        "food"
        if plan.intent == "structured_lookup"
        else "document"
        if plan.intent == "document_evidence"
        else None
    )
    return [
        item
        for item, _ in search_dense(
            plan.queries[0],
            candidate_limit,
            kind=kind,
        )
        if kind is None or item.kind == kind
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Measure fused and reranked retrieval against stable evidence labels. "
            "Requires the configured database, Qdrant collection, and local models."
        )
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--candidate-limit", type=int, default=CANDIDATE_LIMIT)
    args = parser.parse_args()
    if args.top_k < 1 or args.candidate_limit < args.top_k:
        parser.error("top-k must be positive and candidate-limit must be >= top-k")

    cases = load_cases(args.dataset)
    stage_rows: dict[str, list[dict[str, float]]] = defaultdict(list)
    category_rows: dict[str, dict[str, list[dict[str, float]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    baseline_case_count = 0
    dense_case_count = 0
    dense_unavailable_food_case_count = 0
    skipped: list[dict[str, str]] = []
    labeled_case_count = 0

    dense_food_index_available = _dense_food_index_available()
    with SessionLocal() as db:
        for case in cases:
            relevant_keys = set(case["relevant_evidence_keys"])
            if not relevant_keys:
                skipped.append(
                    {
                        "category": case["category"],
                        "question": case["question"],
                        "reason": "no_relevant_evidence_labels",
                    }
                )
                continue
            labeled_case_count += 1

            plan = process_query(case["question"])
            relevance_judgments = case.get("relevance_grades", relevant_keys)
            fused = retrieve_evidence(
                db,
                plan,
                candidate_limit=args.candidate_limit,
            )
            stages = {
                "rrf": fused,
                "reranked": rerank_evidence(
                    case["question"],
                    fused,
                    limit=args.top_k,
                ),
            }
            if case["baseline_comparable"] and not plan.constraints.numeric_constraints:
                stages["bm25"] = _bm25_baseline(
                    plan,
                    db=db,
                    candidate_limit=args.candidate_limit,
                )
                baseline_case_count += 1
                has_food_labels = any(
                    key.startswith("food:")
                    for key in relevant_keys
                )
                if not has_food_labels or dense_food_index_available:
                    stages["dense"] = _dense_baseline(
                        plan,
                        candidate_limit=args.candidate_limit,
                    )
                    dense_case_count += 1
                else:
                    dense_unavailable_food_case_count += 1

            for stage_name, items in stages.items():
                keys = [
                    key
                    for item in items
                    if (key := _evidence_key(item)) is not None
                ]
                row = _metrics(
                    keys,
                    relevance_judgments,
                    top_k=args.top_k,
                )
                stage_rows[stage_name].append(row)
                category_rows[case["category"]][stage_name].append(row)

    result = {
        "top_k": args.top_k,
        "candidate_limit": args.candidate_limit,
        "labeled_case_count": labeled_case_count,
        "bm25_comparison_case_count": baseline_case_count,
        "dense_comparison_case_count": dense_case_count,
        "dense_food_index_available": dense_food_index_available,
        "dense_unavailable_food_case_count": dense_unavailable_food_case_count,
        "macro_metrics": {
            stage: _mean_metrics(rows)
            for stage, rows in sorted(stage_rows.items())
        },
        "metrics_by_category": {
            category: {
                stage: _mean_metrics(rows)
                for stage, rows in sorted(stages.items())
            }
            for category, stages in sorted(category_rows.items())
        },
        "skipped_unlabeled_cases": skipped,
        "note": (
            "Precision@K divides unique relevant results in the first K unique "
            "results by K. Recall@K divides those relevant results by the total "
            "unique relevant labels for the query. MRR uses the rank of the first "
            "relevant result in the full unique ranking. nDCG@K uses gains "
            "2^grade - 1 and an ideal ranking formed by sorting all query grades "
            "descending and truncating at K. Label lists are binary relevance "
            "(grade 1); optional relevance_grades values are graded 1 to 3. "
            "Macro metrics are unweighted means of per-query metrics over queries "
            "where each stage ran; category metrics are unweighted means within "
            "that category. "
            "BM25-only and dense-only baselines are omitted for numeric-filter "
            "cases because neither applies the same SQL constraints. Dense-only "
            "metrics are omitted for cases with food evidence labels when the "
            "Qdrant collection has no food points; those queries cannot evaluate "
            "dense retrieval of structured food records. RRF and reranking are "
            "measured where relevance labels are available. This evaluator "
            "measures retrieval ranking, not answer faithfulness or safety."
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
