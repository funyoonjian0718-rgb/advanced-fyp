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
from app.services.evidence import Evidence, chunk_evidence, food_evidence
from app.services.query_processor import NUTRIENT_TERMS, QueryPlan
from app.services.vector_store import search_dense


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
) -> list[tuple[Evidence, float]]:
    if not retrieval_index.corpus or retrieval_index.bm25 is None:
        return []
    scores = retrieval_index.bm25.get_scores(_tokens(query))
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda corpus_index: scores[corpus_index],
        reverse=True,
    )[:limit]
    return [
        (retrieval_index.corpus[corpus_index], float(scores[corpus_index]))
        for corpus_index in ranked_indices
        if scores[corpus_index] > 0
    ]


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
            retrieval_score=1.0,
        )
        for _, _, _, _, item in ranked
    ]


def retrieve_evidence(
    db: Session,
    plan: QueryPlan,
    *,
    candidate_limit: int = 30,
) -> list[Evidence]:
    index = _load_retrieval_index(db.get_bind())
    if not index.corpus:
        return []

    bm25_rankings: list[list[tuple[Evidence, float]]] = []
    dense_rankings: list[list[tuple[Evidence, float]]] = []
    query_set = tuple(dict.fromkeys((*plan.queries, *plan.subquestions)))

    for query in query_set:
        bm25_rankings.append(_bm25_rank(query, index, candidate_limit))
        dense_rankings.append(search_dense(query, candidate_limit))

    fused: dict[str, Evidence] = {}
    fused_scores: dict[str, float] = defaultdict(float)
    for rankings in (bm25_rankings, dense_rankings):
        for ranked in rankings:
            for rank, (item, raw_score) in enumerate(ranked, start=1):
                request_item = fused.get(item.evidence_id)
                if request_item is None:
                    request_item = replace(
                        item,
                        metadata=copy.deepcopy(item.metadata),
                    )
                    fused[item.evidence_id] = request_item
                authority_multiplier = 0.5 + item.authority_score * 0.5
                fused_scores[item.evidence_id] += (
                    authority_multiplier / (60 + rank)
                )
                request_item.retrieval_score = max(
                    request_item.retrieval_score,
                    raw_score,
                )

    ordered = sorted(
        fused.values(),
        key=lambda item: fused_scores[item.evidence_id],
        reverse=True,
    )
    if plan.intent == "structured_lookup":
        ordered = [item for item in ordered if item.kind == "food"]
    elif plan.intent == "document_evidence":
        ordered = [item for item in ordered if item.kind == "document"]
    if plan.intent == "structured_lookup":
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
        key=lambda item: item.reranker_score + 0.2 * item.authority_score,
        reverse=True,
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
