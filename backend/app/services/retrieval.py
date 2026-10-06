import re
from collections import defaultdict
from functools import lru_cache

from rank_bm25 import BM25Okapi
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.core.config import settings
from app.models.knowledge import DocumentChunk, FoodNutrition
from app.services.evidence import Evidence, chunk_evidence, food_evidence
from app.services.query_processor import NUTRIENT_TERMS, QueryPlan
from app.services.vector_store import search_dense


def _tokens(text: str) -> list[str]:
    return re.findall(r"[\w]+(?:['’-][\w]+)?", text.casefold())


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
    return [*(food_evidence(food) for food in foods), *(chunk_evidence(chunk) for chunk in chunks)]


def _bm25_rank(
    query: str,
    corpus: list[Evidence],
    tokenized_corpus: list[list[str]],
    limit: int,
) -> list[tuple[Evidence, float]]:
    if not corpus:
        return []
    scores = BM25Okapi(tokenized_corpus).get_scores(_tokens(query))
    ranked_indices = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )[:limit]
    return [
        (corpus[index], float(scores[index]))
        for index in ranked_indices
        if scores[index] > 0
    ]


def _direct_food_matches(query: str, corpus: list[Evidence]) -> list[Evidence]:
    ignored = NUTRIENT_TERMS | {
        "a",
        "about",
        "and",
        "are",
        "do",
        "does",
        "for",
        "give",
        "have",
        "how",
        "in",
        "is",
        "many",
        "much",
        "of",
        "per",
        "the",
        "what",
        "with",
        "g",
        "gram",
        "grams",
        "kg",
        "mg",
        "ml",
        "100",
    }
    query_tokens = set(_tokens(query)) - ignored
    if not query_tokens:
        return []

    matches: list[tuple[int, int, Evidence]] = []
    for item in corpus:
        if item.kind != "food":
            continue
        name_tokens = set(_tokens(str(item.metadata.get("name", ""))))
        if name_tokens and name_tokens.issubset(query_tokens):
            matches.append((len(name_tokens), len(name_tokens), item))
            continue
        aliases = str(item.metadata.get("aliases", ""))
        alias_match = max(
            (
                len(alias_tokens)
                for alias_tokens in (
                    set(_tokens(alias))
                    for alias in re.split(r"[,;|]", aliases)
                )
                if alias_tokens and alias_tokens.issubset(query_tokens)
            ),
            default=0,
        )
        if alias_match:
            matches.append((alias_match, alias_match - 1, item))

    ranked = sorted(
        matches,
        key=lambda match: (match[0], match[1], match[2].authority_score),
        reverse=True,
    )
    for _, _, item in ranked:
        item.retrieval_score = 1.0
    return [item for _, _, item in ranked]


def retrieve_evidence(
    db: Session,
    plan: QueryPlan,
    *,
    candidate_limit: int = 30,
) -> list[Evidence]:
    corpus = _load_corpus(db)
    if not corpus:
        return []

    tokenized_corpus = [_tokens(item.text) for item in corpus]
    bm25_rankings: list[list[tuple[Evidence, float]]] = []
    dense_rankings: list[list[tuple[Evidence, float]]] = []
    query_set = tuple(dict.fromkeys((*plan.queries, *plan.subquestions)))

    for query in query_set:
        bm25_rankings.append(
            _bm25_rank(query, corpus, tokenized_corpus, candidate_limit)
        )
        dense_rankings.append(search_dense(query, candidate_limit))

    fused: dict[str, Evidence] = {}
    fused_scores: dict[str, float] = defaultdict(float)
    for rankings in (bm25_rankings, dense_rankings):
        for ranked in rankings:
            for rank, (item, raw_score) in enumerate(ranked, start=1):
                fused[item.evidence_id] = item
                authority_multiplier = 0.5 + item.authority_score * 0.5
                fused_scores[item.evidence_id] += (
                    authority_multiplier / (60 + rank)
                )
                item.retrieval_score = max(item.retrieval_score, raw_score)

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
        direct_matches = _direct_food_matches(plan.queries[0], corpus)
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
