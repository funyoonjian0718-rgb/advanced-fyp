import copy
import re
from collections import defaultdict
from dataclasses import dataclass, replace
from functools import lru_cache

from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, joinedload

from app.core.config import settings
from app.models.knowledge import DocumentChunk, FoodNutrition
from app.services.constraints import NumericConstraint, enforce_constraints
from app.services.evidence import Evidence, chunk_evidence, food_evidence
from app.services.query_processor import NUTRIENT_TERMS, QueryPlan
from app.services.vector_store import search_dense


NUMERIC_COLUMNS = {
    "calories": FoodNutrition.calories,
    "protein_g": FoodNutrition.protein_g,
    "carbohydrates_g": FoodNutrition.carbohydrates_g,
    "fat_g": FoodNutrition.fat_g,
    "fiber_g": FoodNutrition.fiber_g,
    "sugar_g": FoodNutrition.sugar_g,
    "sodium_mg": FoodNutrition.sodium_mg,
    "potassium_mg": FoodNutrition.potassium_mg,
    "glycemic_index": FoodNutrition.glycemic_index,
}


def _numeric_filtered_foods(
    db: Session,
    constraints: tuple[NumericConstraint, ...],
    limit: int,
    *,
    plan: QueryPlan | None = None,
) -> list[Evidence]:
    conditions = []
    for constraint in constraints:
        column = NUMERIC_COLUMNS.get(constraint.field)
        if column is None:
            raise ValueError(
                f"Unsupported numeric nutrition field: {constraint.field}"
            )

        if constraint.operator == "<":
            conditions.append(column < constraint.value)
        elif constraint.operator == "<=":
            conditions.append(column <= constraint.value)
        elif constraint.operator == ">":
            conditions.append(column > constraint.value)
        elif constraint.operator == ">=":
            conditions.append(column >= constraint.value)
        elif constraint.operator == "=":
            conditions.append(column == constraint.value)
        else:
            raise ValueError(
                f"Unsupported numeric comparison operator: "
                f"{constraint.operator}"
            )

    statement = (
        select(FoodNutrition)
        .options(joinedload(FoodNutrition.source))
        .where(*conditions)
        .order_by(FoodNutrition.id)
    )
    has_profile_constraints = plan is not None and bool(
        plan.constraints.diets
        or plan.constraints.allergens
        or plan.constraints.unknown_dietary_constraint
        or plan.constraints.unknown_allergen_constraint
    )
    if not has_profile_constraints:
        foods = db.scalars(statement.limit(limit)).all()
        return [food_evidence(food) for food in foods]

    compatible: list[Evidence] = []
    incompatible: list[Evidence] = []
    offset = 0
    while len(compatible) < limit:
        foods = db.scalars(
            statement.limit(limit).offset(offset)
        ).all()
        if not foods:
            break
        offset += len(foods)
        candidates = [food_evidence(food) for food in foods]
        constrained = enforce_constraints(candidates, plan)
        compatible.extend(constrained.allowed)
        if len(incompatible) < limit:
            incompatible.extend(
                constrained.excluded[: limit - len(incompatible)]
            )
    compatible = compatible[:limit]
    return [*compatible, *incompatible[: max(limit - len(compatible), 0)]]


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w]+(?:['’-][\w]+)?", text.casefold())


def _normalize_food_token(token: str) -> str:
    token = token.replace("’", "'")
    if token.endswith("'s"):
        token = token[:-2]
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


_FOOD_QUERY_IGNORED = {
    "a", "about", "and", "are", "do", "does", "for", "give", "have",
    "how", "in", "is", "many", "much", "of", "per", "the", "what",
    "with", "g", "gram", "grams", "kg", "mg", "ml", "100", "one",
    "medium", "large", "small", "serving", "size", "portion", "nutrition",
    "information", "facts", "content",
} | {_normalize_food_token(token) for token in NUTRIENT_TERMS}


def _food_match_score(query: str, item: Evidence) -> tuple[int, int, int] | None:
    if item.kind != "food":
        return None
    query_tokens = [
        _normalize_food_token(token)
        for token in _tokens(query)
        if _normalize_food_token(token) not in _FOOD_QUERY_IGNORED
    ]
    if not query_tokens:
        return None
    name = str(item.metadata.get("name", ""))
    candidates = [
        (3, name),
        *( (2, alias) for alias in re.findall(r"\(([^()]*)\)", name) ),
        *(
            (1, alias)
            for alias in re.split(
                r"[,;|]",
                str(item.metadata.get("aliases", "")),
            )
        ),
    ]
    matches: list[tuple[int, int, int]] = []
    for source_priority, candidate in candidates:
        candidate_tokens = [
            _normalize_food_token(token)
            for token in _tokens(candidate)
        ]
        if not candidate_tokens:
            continue
        if query_tokens == candidate_tokens:
            matches.append((3, len(candidate_tokens), source_priority))
            continue
        if len(candidate_tokens) <= len(query_tokens) and any(
            query_tokens[index : index + len(candidate_tokens)] == candidate_tokens
            for index in range(len(query_tokens) - len(candidate_tokens) + 1)
        ):
            matches.append((2, len(candidate_tokens), source_priority))
    return max(matches, default=None)


def _load_corpus(db: Session) -> list[Evidence]:
    foods = db.scalars(
        select(FoodNutrition)
        .options(joinedload(FoodNutrition.source))
        .order_by(FoodNutrition.id)
    ).all()
    chunks = db.scalars(
        select(DocumentChunk)
        .options(joinedload(DocumentChunk.source))
        .order_by(DocumentChunk.id)
    ).all()
    return [
        *(food_evidence(food) for food in foods),
        *(chunk_evidence(chunk) for chunk in chunks),
    ]


@dataclass(frozen=True)
class _RetrievalIndex:
    corpus: tuple[Evidence, ...]
    tokenized_corpus: tuple[tuple[str, ...], ...]
    bm25: BM25Okapi | None


@lru_cache(maxsize=4)
def _load_retrieval_index(engine: Engine) -> _RetrievalIndex:
    with Session(engine) as db:
        corpus = _load_corpus(db)
    tokenized = tuple(tuple(_tokens(item.text)) for item in corpus)
    bm25 = BM25Okapi([list(tokens) for tokens in tokenized]) if corpus else None
    return _RetrievalIndex(tuple(corpus), tokenized, bm25)


def invalidate_retrieval_cache() -> None:
    _load_retrieval_index.cache_clear()


def _bm25_rank(
    query: str,
    retrieval_index: _RetrievalIndex,
    limit: int,
    *,
    kind: str | None = None,
) -> list[tuple[Evidence, float]]:
    if not retrieval_index.corpus or retrieval_index.bm25 is None:
        return []
    scores = retrieval_index.bm25.get_scores(_tokens(query))
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda corpus_index: scores[corpus_index],
        reverse=True,
    )
    ranked: list[tuple[Evidence, float]] = []
    for corpus_index in ranked_indices:
        item = retrieval_index.corpus[corpus_index]
        if scores[corpus_index] <= 0 or (kind and item.kind != kind):
            continue
        ranked.append((item, float(scores[corpus_index])))
        if len(ranked) == limit:
            break
    return ranked


def _reciprocal_rank_fusion(
    rankings: list[list[tuple[Evidence, float]]],
    *,
    rank_constant: int = 60,
) -> list[Evidence]:
    fused: dict[str, Evidence] = {}
    fused_scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, (item, raw_score) in enumerate(ranking, start=1):
            fused_item = fused.get(item.evidence_id)
            if fused_item is None:
                fused_item = replace(
                    item,
                    metadata=copy.deepcopy(item.metadata),
                )
                fused[item.evidence_id] = fused_item
            fused_scores[item.evidence_id] += 1 / (rank_constant + rank)
            fused_item.retrieval_score = max(fused_item.retrieval_score, raw_score)
            fused_item.rrf_score = fused_scores[item.evidence_id]

    return sorted(
        fused.values(),
        key=lambda item: (-item.rrf_score, item.evidence_id),
    )


def _direct_food_matches(query: str, corpus: list[Evidence]) -> list[Evidence]:
    query_tokens = [
        token
        for token in _tokens(query)
        if _normalize_food_token(token) not in _FOOD_QUERY_IGNORED
    ]
    if not query_tokens:
        return []
    food_query = " ".join(query_tokens)

    matches: list[tuple[int, int, int, float, Evidence]] = []
    for item in corpus:
        score = _food_match_score(food_query, item)
        if score is not None:
            matches.append((*score, item.authority_score, item))

    ranked = sorted(
        matches,
        key=lambda match: (match[0], match[1], match[2]),
        reverse=True,
    )
    return [
        replace(
            item,
            metadata=copy.deepcopy(item.metadata),
        )
        for _, _, _, _, item in ranked
    ]


def retrieve_evidence(
    db: Session,
    plan: QueryPlan,
    *,
    candidate_limit: int = 30,
) -> list[Evidence]:
    if plan.constraints.numeric_constraints:
        foods = _numeric_filtered_foods(
            db,
            plan.constraints.numeric_constraints,
            candidate_limit,
            plan=plan,
        )
        if not plan.document_lookup and plan.intent != "document_evidence":
            return foods
        documents = _retrieve_ranked_evidence(
            db,
            plan,
            candidate_limit=candidate_limit,
            evidence_kind="document",
        )
        combined: list[Evidence] = []
        for index in range(max(len(foods), len(documents))):
            if index < len(foods):
                combined.append(foods[index])
            if index < len(documents):
                combined.append(documents[index])
            if len(combined) >= candidate_limit:
                break
        return combined

    return _retrieve_ranked_evidence(
        db,
        plan,
        candidate_limit=candidate_limit,
    )


def _retrieve_ranked_evidence(
    db: Session,
    plan: QueryPlan,
    *,
    candidate_limit: int,
    evidence_kind: str | None = None,
) -> list[Evidence]:
    index = _load_retrieval_index(db.get_bind())

    bm25_rankings: list[list[tuple[Evidence, float]]] = []
    dense_rankings: list[list[tuple[Evidence, float]]] = []
    query_set = tuple(dict.fromkeys((*plan.queries, *plan.subquestions)))

    for query in query_set:
        bm25_rankings.append(
            _bm25_rank(
                query,
                index,
                candidate_limit,
                kind=evidence_kind,
            )
        )
        if evidence_kind is None:
            dense_rankings.append(search_dense(query, candidate_limit))
        else:
            dense_rankings.append(
                search_dense(
                    query,
                    candidate_limit,
                    kind=evidence_kind,
                )
            )

    ordered = _reciprocal_rank_fusion(
        [*bm25_rankings, *dense_rankings]
    )
    if evidence_kind is not None:
        ordered = [item for item in ordered if item.kind == evidence_kind]
    elif plan.intent == "structured_lookup":
        ordered = [item for item in ordered if item.kind == "food"]
    elif plan.intent == "document_evidence":
        ordered = [item for item in ordered if item.kind == "document"]
    if evidence_kind is None and plan.intent in {
        "structured_lookup",
        "comparison",
    }:
        direct_matches = _direct_food_matches(plan.queries[0], list(index.corpus))
        direct_ids = {item.evidence_id for item in direct_matches}
        ordered = direct_matches + [
            item for item in ordered if item.evidence_id not in direct_ids
        ]
    return ordered[:candidate_limit]


@lru_cache(maxsize=1)
def _get_reranker():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(settings.reranker_model_id, device="cpu")


def rerank_evidence(
    query: str,
    evidence_items: list[Evidence],
    *,
    limit: int = 8,
) -> list[Evidence]:
    if not evidence_items:
        return []
    reranker = _get_reranker()
    scores = reranker.predict(
        [(query, item.text[:4000]) for item in evidence_items],
        show_progress_bar=False,
    )
    for item, score in zip(evidence_items, scores, strict=True):
        item.reranker_score = float(score)
    return sorted(
        evidence_items,
        key=lambda item: (-item.reranker_score, item.evidence_id),
    )[:limit]


def compress_evidence(
    query: str,
    evidence_items: list[Evidence],
    *,
    max_chars_per_item: int = 1400,
) -> list[Evidence]:
    query_terms = set(_tokens(query))
    compressed: list[Evidence] = []
    for item in evidence_items:
        if item.kind == "food":
            compressed.append(item)
            continue

        sentences = [
            sentence.strip()
            for sentence in re.split(r"(?<=[.!?])\s+", item.text)
            if sentence.strip()
        ]
        ranked = sorted(
            enumerate(sentences),
            key=lambda pair: (
                len(query_terms.intersection(_tokens(pair[1])))
                / max(len(set(_tokens(pair[1]))), 1),
                -pair[0],
            ),
            reverse=True,
        )
        selected_indices: list[int] = []
        size = 0
        for index, sentence in ranked:
            if size + len(sentence) > max_chars_per_item:
                continue
            selected_indices.append(index)
            size += len(sentence)
        if selected_indices:
            item.text = " ".join(
                sentences[index] for index in sorted(selected_indices)
            )
            compressed.append(item)
    return compressed
