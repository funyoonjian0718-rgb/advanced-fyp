import asyncio
import os
from collections.abc import Iterator
from dataclasses import replace

import httpx
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.core.config import settings
from app.core.database import Base, get_db
from app.main import app
from app.models.knowledge import (
    ChatMessage,
    ChatSession,
    FoodNutrition,
    KnowledgeSource,
)
from app.models.user import User
from app.models.user_profile import UserProfile
from app.schemas.chat import ChatResponse
from app.services.auth_service import create_access_token, hash_password
from app.services.chat_service import _safe_refusal
from app.services.constraints import ConstraintResult, NumericConstraint
from app.services.evidence import Evidence, food_evidence
from app.services.generation import (
    GeneratedAnswer,
    GeneratedClaim,
    ModelOutputError,
)
from app.services.query_processor import process_query


@pytest.fixture
def sqlite_engine():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    with Session(engine) as db:
        db.add_all(
            [
                User(
                    id=1,
                    email="chat-one@example.test",
                    password_hash=hash_password("password-one"),
                ),
                User(
                    id=2,
                    email="chat-two@example.test",
                    password_hash=hash_password("password-two"),
                ),
            ]
        )
        db.add(
            ChatSession(
                id=20,
                user_id=2,
                title="Another user's conversation",
            )
        )
        db.commit()

    def override_db() -> Iterator[Session]:
        with Session(engine) as db:
            yield db

    app.dependency_overrides[get_db] = override_db
    yield engine
    app.dependency_overrides.clear()
    engine.dispose()


def _request(method: str, path: str, *, user_id: int | None = None, **kwargs):
    async def send_request():
        headers = {}
        if user_id is not None:
            headers["Authorization"] = f"Bearer {create_access_token(user_id)}"
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport,
            base_url="http://testserver",
        ) as client:
            return await client.request(
                method,
                path,
                headers=headers,
                **kwargs,
            )

    return asyncio.run(send_request())


def _food_evidence(
    *,
    name: str = "Lentil patty",
    ingredients: str = "lentils",
    allergens: list[str] | None = None,
    allergen_data_reviewed: bool = False,
    allergen_data_verified: bool = False,
) -> Evidence:
    allergen_text = (
        ", ".join(allergens)
        if allergens is not None
        else "not provided"
    )
    return Evidence(
        evidence_id="food-lentil-patty",
        text=(
            f"Food: {name}\nCalories: 250 kcal\n"
            f"Ingredients: {ingredients}\nAllergens: {allergen_text}"
        ),
        source_id=1,
        source_title="Nutrition dataset",
        source_url=None,
        source_type="user_uploaded_csv",
        authority_score=0.7,
        kind="food",
        metadata={
            "food_id": 1,
            "source_record_id": "lentil-patty",
            "name": name,
            "aliases": "",
            "ingredients": ingredients,
            "allergens": allergens,
            "dietary_labels": [],
            "allergen_data_populated": allergens is not None,
            "allergen_data_reviewed": allergen_data_reviewed,
            "allergen_data_verified": allergen_data_verified,
            "dietary_data_reviewed": False,
            "dietary_data_verified": False,
        },
    )


@pytest.mark.usefixtures("sqlite_engine")
def test_chat_and_history_require_authentication() -> None:
    response = _request("GET", "/api/chat/history")

    assert response.status_code == 401


@pytest.mark.usefixtures("sqlite_engine")
def test_chat_turn_is_persisted_and_scoped_to_the_jwt_user() -> None:
    created = _request(
        "POST",
        "/api/chat",
        user_id=1,
        json={"message": "How much protein is in tempeh?"},
    )

    assert created.status_code == 200, created.text
    payload = created.json()
    assert "confidence" not in payload
    session_id = payload["session_id"]

    history = _request("GET", "/api/chat/history", user_id=1)
    assert history.status_code == 200
    assert [item["id"] for item in history.json()] == [session_id]

    detail = _request(
        "GET",
        f"/api/chat/history/{session_id}",
        user_id=1,
    )
    assert detail.status_code == 200
    assert [message["role"] for message in detail.json()["messages"]] == [
        "user",
        "assistant",
    ]


@pytest.mark.usefixtures("sqlite_engine")
def test_user_cannot_read_another_users_conversation() -> None:
    response = _request(
        "GET",
        "/api/chat/history/20",
        user_id=1,
    )

    assert response.status_code == 404


@pytest.mark.usefixtures("sqlite_engine")
def test_terminal_command_does_not_trigger_nutrition_retrieval(
    monkeypatch,
) -> None:
    def unexpected_retrieval(*_args, **_kwargs):
        raise AssertionError("Out-of-scope commands must skip retrieval.")

    monkeypatch.setattr(
        "app.services.chat_service.retrieve_evidence",
        unexpected_retrieval,
    )
    response = _request(
        "POST",
        "/api/chat",
        user_id=1,
        json={"message": "uvicorn app.main:app --reload --reload-dir app"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["intent"] == "out_of_scope"
    assert payload["citations"] == []
    assert "confidence" not in payload
    assert "terminal command" in payload["answer"]


def test_new_nutrition_question_is_not_expanded_with_prior_chat_history(
    sqlite_engine,
    monkeypatch,
) -> None:
    with Session(sqlite_engine) as db:
        session = ChatSession(
            id=21,
            user_id=1,
            title="Food questions",
        )
        db.add(session)
        db.flush()
        db.add_all(
            [
                ChatMessage(
                    session_id=session.id,
                    role="user",
                    content="What is the nutrition information for one medium apple?",
                ),
                ChatMessage(
                    session_id=session.id,
                    role="assistant",
                    content="Apple: Protein: 1 g.",
                ),
            ]
        )
        db.commit()

    captured_plans = []

    def capture_plan(_db, plan, **_kwargs):
        captured_plans.append(plan)
        return []

    monkeypatch.setattr(
        "app.services.chat_service.retrieve_evidence",
        capture_plan,
    )
    response = _request(
        "POST",
        "/api/chat",
        user_id=1,
        json={
            "message": "What is the serving size of banana?",
            "session_id": 21,
        },
    )

    assert response.status_code == 200, response.text
    assert len(captured_plans) == 1
    assert captured_plans[0].intent == "structured_lookup"
    assert all("apple" not in query for query in captured_plans[0].queries)
    assert all("banana" in query for query in captured_plans[0].queries)


def test_unsupported_numeric_filter_is_reported_as_a_client_error(
    sqlite_engine,
    monkeypatch,
) -> None:
    valid_plan = process_query("Find foods with less than 300 calories")
    invalid_plan = replace(
        valid_plan,
        constraints=replace(
            valid_plan.constraints,
            numeric_constraints=(
                NumericConstraint(
                    field="private_column",
                    operator="<",
                    value=5,
                ),
            ),
        ),
    )
    monkeypatch.setattr(
        "app.services.chat_service.process_query",
        lambda *_args, **_kwargs: invalid_plan,
    )

    response = _request(
        "POST",
        "/api/chat",
        user_id=1,
        json={"message": "Find foods with less than 300 calories"},
    )

    assert response.status_code == 422
    assert "Unsupported numeric nutrition field" in response.json()["detail"]


def test_empty_numeric_search_reports_no_matching_foods(sqlite_engine) -> None:
    response = _request(
        "POST",
        "/api/chat",
        user_id=1,
        json={"message": "Find foods with less than 300 calories"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert "No foods in the available dataset match" in payload["answer"]
    assert payload["citations"] == []
    assert payload["warnings"] == []


def test_reranker_failure_returns_service_unavailable(
    sqlite_engine,
    monkeypatch,
) -> None:
    evidence = Evidence(
        evidence_id="chunk-guideline",
        text="Dietary guidelines recommend balanced meals.",
        source_id=1,
        source_title="Guideline",
        source_url=None,
        source_type="guideline",
        authority_score=0.8,
        kind="document",
    )
    monkeypatch.setattr(
        "app.services.chat_service.retrieve_evidence",
        lambda *_args, **_kwargs: [evidence],
    )

    def fail_reranking(*_args, **_kwargs):
        raise RuntimeError("reranker unavailable")

    monkeypatch.setattr(
        "app.services.chat_service.rerank_evidence",
        fail_reranking,
    )

    response = _request(
        "POST",
        "/api/chat",
        user_id=1,
        json={"message": "Explain dietary guidelines."},
    )

    assert response.status_code == 503
    assert "reranker is unavailable" in response.json()["detail"]


def test_safety_refusal_distinguishes_unknown_allergy_from_conflict() -> None:
    unknown_allergy = _safe_refusal(
        ConstraintResult(
            allowed=[],
            excluded=[],
            warnings=["Allergen data is missing."],
            allergen_status_unknown=True,
        )
    )
    known_conflict = _safe_refusal(
        ConstraintResult(
            allowed=[],
            excluded=[],
            warnings=["A food conflicts with the user's allergy."],
            has_known_conflict=True,
        )
    )

    assert "allergen information is missing or unclear" in unknown_allergy.answer
    assert "conflict with your dietary or allergen restrictions" in known_conflict.answer
    assert unknown_allergy.answer != known_conflict.answer


@pytest.mark.parametrize(
    ("allergens", "reviewed", "verified"),
    [
        (None, False, False),
        ([], False, False),
        ([], True, False),
        ([], False, True),
    ],
)
def test_chat_refuses_food_without_reviewed_and_verified_allergen_data(
    sqlite_engine,
    monkeypatch,
    allergens,
    reviewed,
    verified,
) -> None:
    evidence = _food_evidence(
        allergens=allergens,
        allergen_data_reviewed=reviewed,
        allergen_data_verified=verified,
    )
    monkeypatch.setattr(
        "app.services.chat_service.retrieve_evidence",
        lambda *_args, **_kwargs: [evidence],
    )

    def unexpected_model_call(*_args, **_kwargs):
        raise AssertionError("Unverified foods must not reach answer generation.")

    monkeypatch.setattr(
        "app.services.chat_service.rerank_evidence",
        unexpected_model_call,
    )
    monkeypatch.setattr(
        "app.services.generation._generate",
        unexpected_model_call,
    )

    response = _request(
        "POST",
        "/api/chat",
        user_id=1,
        json={"message": "Which foods can I eat if I have a peanut allergy?"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert "can't verify" in payload["answer"]
    assert "allergen information is missing or unclear" in payload["answer"]
    assert "Lentil patty" not in payload["answer"]
    assert payload["citations"] == []
    assert any("verified allergen declarations" in item for item in payload["warnings"])


def test_chat_allows_food_only_when_allergen_data_is_reviewed_and_verified(
    sqlite_engine,
    monkeypatch,
) -> None:
    evidence = _food_evidence(
        allergens=[],
        allergen_data_reviewed=True,
        allergen_data_verified=True,
    )
    captured_candidates = []
    monkeypatch.setattr(
        "app.services.chat_service.retrieve_evidence",
        lambda *_args, **_kwargs: [evidence],
    )

    def capture_reranked(_query, items):
        captured_candidates.extend(items)
        return items

    monkeypatch.setattr(
        "app.services.chat_service.rerank_evidence",
        capture_reranked,
    )
    monkeypatch.setattr(
        "app.services.generation._generate",
        lambda _prompt: GeneratedAnswer(
            claims=[
                GeneratedClaim(
                    text="Lentil patty.",
                    evidence_ids=[evidence.evidence_id],
                )
            ]
        ),
    )

    response = _request(
        "POST",
        "/api/chat",
        user_id=1,
        json={"message": "Which foods can I eat if I have a peanut allergy?"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert "Lentil patty." in payload["answer"]
    assert [item.evidence_id for item in captured_candidates] == [
        evidence.evidence_id
    ]
    assert captured_candidates[0].metadata["allergen_data_reviewed"] is True
    assert captured_candidates[0].metadata["allergen_data_verified"] is True
    assert payload["citations"]
    assert "safe" not in payload["answer"].casefold()


def test_chat_refuses_when_all_numeric_food_matches_conflict_with_allergy(
    sqlite_engine,
    monkeypatch,
) -> None:
    with Session(sqlite_engine) as db:
        db.add(UserProfile(user_id=1, allergies="peanuts"))
        db.commit()
    evidence = _food_evidence(
        name="Peanut chicken",
        ingredients="chicken, peanuts",
        allergens=["peanut"],
        allergen_data_reviewed=True,
        allergen_data_verified=True,
    )
    monkeypatch.setattr(
        "app.services.chat_service.retrieve_evidence",
        lambda *_args, **_kwargs: [evidence],
    )

    def unexpected_model_call(*_args, **_kwargs):
        raise AssertionError("Conflicting foods must not reach answer generation.")

    monkeypatch.setattr(
        "app.services.chat_service.rerank_evidence",
        unexpected_model_call,
    )
    monkeypatch.setattr(
        "app.services.generation._generate",
        unexpected_model_call,
    )

    response = _request(
        "POST",
        "/api/chat",
        user_id=1,
        json={"message": "Find foods below 300 calories"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert "conflict with your dietary or allergen restrictions" in payload["answer"]
    assert "Peanut chicken" not in payload["answer"]
    assert payload["citations"] == []


def test_chat_does_not_invent_moon_dust_nutrition_from_irrelevant_evidence(
    sqlite_engine,
    monkeypatch,
) -> None:
    irrelevant = Evidence(
        evidence_id="chunk-nutrition-guide",
        text="Nasi Lemak contains energy and protein per serving.",
        source_id=1,
        source_title="Nutrition guide",
        source_url=None,
        source_type="guideline",
        authority_score=0.8,
        kind="document",
    )
    monkeypatch.setattr(
        "app.services.chat_service.retrieve_evidence",
        lambda *_args, **_kwargs: [irrelevant],
    )
    monkeypatch.setattr(
        "app.services.chat_service.rerank_evidence",
        lambda _query, items: items,
    )
    monkeypatch.setattr(
        "app.services.generation._generate",
        lambda _prompt: GeneratedAnswer(
            claims=[
                GeneratedClaim(
                    text="Moon dust contains 42 micrograms of vitamin B12.",
                    evidence_ids=[irrelevant.evidence_id],
                )
            ]
        ),
    )

    response = _request(
        "POST",
        "/api/chat",
        user_id=1,
        json={"message": "What is the exact vitamin B12 content of moon dust?"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert "not have enough supported evidence" in payload["answer"]
    assert "42" not in payload["answer"]
    assert payload["citations"] == []
    assert any("lacked sufficient supporting evidence" in item for item in payload["warnings"])


def test_chat_returns_insufficient_evidence_when_retrieval_is_empty(
    sqlite_engine,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "app.services.chat_service.retrieve_evidence",
        lambda *_args, **_kwargs: [],
    )

    def unexpected_model_call(*_args, **_kwargs):
        raise AssertionError("Empty retrieval must not reach answer generation.")

    monkeypatch.setattr(
        "app.services.chat_service.rerank_evidence",
        unexpected_model_call,
    )
    monkeypatch.setattr(
        "app.services.generation._generate",
        unexpected_model_call,
    )

    response = _request(
        "POST",
        "/api/chat",
        user_id=1,
        json={"message": "What does the dietary guidance say about fibre?"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert "not have enough verified evidence" in payload["answer"]
    assert payload["citations"] == []


def test_chat_returns_verified_document_claim_and_citation(
    sqlite_engine,
    monkeypatch,
) -> None:
    evidence = Evidence(
        evidence_id="chunk-balanced-meals",
        text="Dietary guidelines recommend balanced meals.",
        source_id=8,
        source_title="Dietary guide",
        source_url="https://example.test/guide",
        source_type="guideline",
        authority_score=0.9,
        kind="document",
    )
    monkeypatch.setattr(
        "app.services.chat_service.retrieve_evidence",
        lambda *_args, **_kwargs: [evidence],
    )
    monkeypatch.setattr(
        "app.services.chat_service.rerank_evidence",
        lambda _query, items: items,
    )
    monkeypatch.setattr(
        "app.services.chat_service.compress_evidence",
        lambda _query, items: items,
    )
    monkeypatch.setattr(
        "app.services.generation._generate",
        lambda _prompt: GeneratedAnswer(
            claims=[
                GeneratedClaim(
                    text="Dietary guidelines recommend balanced meals.",
                    evidence_ids=[evidence.evidence_id],
                )
            ]
        ),
    )

    response = _request(
        "POST",
        "/api/chat",
        user_id=1,
        json={"message": "Explain dietary guidelines about balanced meals."},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert set(payload) == {
        "session_id",
        "answer",
        "citations",
        "warnings",
        "intent",
    }
    parsed = ChatResponse.model_validate(payload)
    assert parsed.intent == "document_evidence"
    assert "The cited source states:" in parsed.answer
    assert [item.evidence_id for item in parsed.citations] == [
        evidence.evidence_id
    ]


def test_chat_answers_nutrition_from_database_food_evidence(
    sqlite_engine,
    monkeypatch,
) -> None:
    with Session(sqlite_engine) as db:
        source = KnowledgeSource(
            id=8,
            title="Nutrition dataset",
            source_type="nutrition_database",
            source_url="https://example.test/nutrition",
            authority_score=0.9,
        )
        db.add(source)
        db.flush()
        food = FoodNutrition(
            id=42,
            source_id=source.id,
            source_record_id="lentil-patty",
            name="Lentil patty",
            serving_size="1 patty",
            calories=250,
            protein_g=12,
        )
        db.add(food)
        db.commit()

    def retrieve_food(db: Session, _plan):
        record = db.scalar(
            select(FoodNutrition).where(FoodNutrition.id == 42)
        )
        assert record is not None
        return [food_evidence(record)]

    monkeypatch.setattr(
        "app.services.chat_service.retrieve_evidence",
        retrieve_food,
    )

    def unexpected_generation(*_args, **_kwargs):
        raise AssertionError("Structured nutrition lookup should not invoke the model.")

    monkeypatch.setattr(
        "app.services.generation._generate",
        unexpected_generation,
    )
    response = _request(
        "POST",
        "/api/chat",
        user_id=1,
        json={"message": "How many calories are in a lentil patty?"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["intent"] == "structured_lookup"
    assert "Lentil patty: Calories: 250 kcal for 1 patty." in payload["answer"]
    assert [item["evidence_id"] for item in payload["citations"]] == ["food-42"]


def test_chat_rejects_unsupported_nutrition_value_for_retrieved_food(
    sqlite_engine,
    monkeypatch,
) -> None:
    evidence = Evidence(
        evidence_id="food-tempeh",
        text="Food: Tempeh\nServing: 100 g\nProtein: 20 g",
        source_id=8,
        source_title="Nutrition dataset",
        source_url=None,
        source_type="nutrition_database",
        authority_score=0.9,
        kind="food",
        metadata={"name": "Tempeh"},
    )
    monkeypatch.setattr(
        "app.services.chat_service.retrieve_evidence",
        lambda *_args, **_kwargs: [evidence],
    )
    monkeypatch.setattr(
        "app.services.chat_service.rerank_evidence",
        lambda _query, items: items,
    )
    monkeypatch.setattr(
        "app.services.chat_service.compress_evidence",
        lambda _query, items: items,
    )
    monkeypatch.setattr(
        "app.services.generation._generate",
        lambda _prompt: GeneratedAnswer(
            claims=[
                GeneratedClaim(
                    text="Tempeh contains 200 g protein per 100 g.",
                    evidence_ids=[evidence.evidence_id],
                )
            ]
        ),
    )

    response = _request(
        "POST",
        "/api/chat",
        user_id=1,
        json={"message": "Explain why tempeh is nutritious."},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert "not have enough supported evidence" in payload["answer"]
    assert "200 g" not in payload["answer"]
    assert payload["citations"] == []
    assert any(
        "contradicted the retrieved evidence" in warning
        for warning in payload["warnings"]
    )


def test_chat_unsupported_moon_dust_question_does_not_call_generator_when_empty(
    sqlite_engine,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "app.services.chat_service.retrieve_evidence",
        lambda *_args, **_kwargs: [],
    )

    def unexpected_generation(*_args, **_kwargs):
        raise AssertionError("Unsupported empty-evidence questions must abstain.")

    monkeypatch.setattr(
        "app.services.chat_service.rerank_evidence",
        unexpected_generation,
    )
    monkeypatch.setattr(
        "app.services.generation._generate",
        unexpected_generation,
    )

    response = _request(
        "POST",
        "/api/chat",
        user_id=1,
        json={"message": "What is the exact vitamin B12 content of moon dust?"},
    )

    assert response.status_code == 200, response.text
    payload = response.json()
    assert "not have enough verified evidence" in payload["answer"]
    assert "vitamin B12" not in payload["answer"]
    assert payload["citations"] == []


def test_verifier_failure_returns_error_instead_of_an_unverified_answer(
    sqlite_engine,
    monkeypatch,
) -> None:
    evidence = Evidence(
        evidence_id="chunk-guideline",
        text="Dietary guidelines recommend balanced meals.",
        source_id=1,
        source_title="Guideline",
        source_url=None,
        source_type="guideline",
        authority_score=0.8,
        kind="document",
    )
    monkeypatch.setattr(
        "app.services.chat_service.retrieve_evidence",
        lambda *_args, **_kwargs: [evidence],
    )
    monkeypatch.setattr(
        "app.services.chat_service.rerank_evidence",
        lambda _query, items: items,
    )
    monkeypatch.setattr(
        "app.services.chat_service.compress_evidence",
        lambda _query, items: items,
    )
    monkeypatch.setattr(
        "app.services.generation._generate",
        lambda _prompt: GeneratedAnswer(
            claims=[
                GeneratedClaim(
                    text="Dietary guidelines recommend balanced meals.",
                    evidence_ids=["chunk-guideline"],
                )
            ]
        ),
    )

    def fail_verification(*_args, **_kwargs):
        raise RuntimeError("verification failed")

    monkeypatch.setattr(
        "app.services.generation._claim_verification",
        fail_verification,
    )
    response = _request(
        "POST",
        "/api/chat",
        user_id=1,
        json={"message": "Explain dietary guidelines."},
    )

    assert response.status_code == 503
    assert "Evidence verification failed" in response.json()["detail"]


def test_malformed_model_output_returns_safe_abstention(
    sqlite_engine,
    monkeypatch,
) -> None:
    evidence = Evidence(
        evidence_id="chunk-guideline",
        text="Dietary guidelines recommend balanced meals.",
        source_id=1,
        source_title="Guideline",
        source_url=None,
        source_type="guideline",
        authority_score=0.8,
        kind="document",
    )
    monkeypatch.setattr(
        "app.services.chat_service.retrieve_evidence",
        lambda *_args, **_kwargs: [evidence],
    )
    monkeypatch.setattr(
        "app.services.chat_service.rerank_evidence",
        lambda _query, items: items,
    )
    monkeypatch.setattr(
        "app.services.chat_service.compress_evidence",
        lambda _query, items: items,
    )

    def malformed_model_output(_prompt):
        raise ModelOutputError(
            "The local language model returned a response that did not match "
            "the required claims schema (claims.0.evidence_ids: missing)."
        )

    monkeypatch.setattr(
        "app.services.generation._generate",
        malformed_model_output,
    )

    response = _request(
        "POST",
        "/api/chat",
        user_id=1,
        json={"message": "Explain dietary guidelines."},
    )

    assert response.status_code == 200
    payload = response.json()
    assert "not have enough supported evidence" in payload["answer"]
    assert payload["citations"] == []
    assert any(
        "response was malformed" in warning
        for warning in payload["warnings"]
    )


@pytest.mark.skipif(
    os.getenv("ADVANCED_FYP_RUN_LOCAL_MODEL_INTEGRATION") != "1",
    reason="Set ADVANCED_FYP_RUN_LOCAL_MODEL_INTEGRATION=1 to load the real model.",
)
def test_chat_api_with_offline_local_model_and_claim_verifier(
    sqlite_engine,
    monkeypatch,
) -> None:
    from huggingface_hub import try_to_load_from_cache

    model_id = settings.llm_model_id
    cache_dir = str(settings.model_cache_path)
    required_files = ("config.json", "tokenizer.json", "tokenizer_config.json")
    if any(
        try_to_load_from_cache(
            model_id,
            filename,
            cache_dir=cache_dir,
            revision="main",
        )
        is None
        for filename in required_files
    ):
        pytest.skip(f"Configured model {model_id!r} is not fully cached.")
    has_weights = any(
        try_to_load_from_cache(
            model_id,
            filename,
            cache_dir=cache_dir,
            revision="main",
        )
        is not None
        for filename in (
            "model.safetensors",
            "model.safetensors.index.json",
            "pytorch_model.bin",
            "pytorch_model.bin.index.json",
        )
    )
    if not has_weights:
        pytest.skip(f"Weights for configured model {model_id!r} are not cached.")

    evidence = Evidence(
        evidence_id="chunk-balanced-meals",
        text="Dietary guidelines recommend balanced meals.",
        source_id=8,
        source_title="Dietary guide",
        source_url="https://example.test/guide",
        source_type="guideline",
        authority_score=0.9,
        kind="document",
    )
    monkeypatch.setattr(settings, "llm_offline", True)
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(
        "app.services.chat_service.retrieve_evidence",
        lambda *_args, **_kwargs: [evidence],
    )
    monkeypatch.setattr(
        "app.services.chat_service.rerank_evidence",
        lambda _query, items: items,
    )
    monkeypatch.setattr(
        "app.services.chat_service.compress_evidence",
        lambda _query, items: items,
    )

    from app.services.generation import _load_model
    from app.services.generation import _claim_verification as verify_claim

    verification_calls = []

    def track_verification(*args, **kwargs):
        verification_calls.append(args[0])
        return verify_claim(*args, **kwargs)

    monkeypatch.setattr(
        "app.services.generation._claim_verification",
        track_verification,
    )
    _load_model.cache_clear()
    try:
        response = _request(
            "POST",
            "/api/chat",
            user_id=1,
            json={
                "message": "Explain dietary guidelines about balanced meals."
            },
        )
        assert response.status_code == 200, response.text
        payload = response.json()
        assert set(payload) == {
            "session_id",
            "answer",
            "citations",
            "warnings",
            "intent",
        }
        parsed = ChatResponse.model_validate(payload)
        assert parsed.intent == "document_evidence"
        assert verification_calls
        citation_ids = [item.evidence_id for item in parsed.citations]
        if citation_ids:
            assert citation_ids == [evidence.evidence_id]
            assert "Dietary guidelines recommend balanced meals." in parsed.answer
        else:
            assert "not have enough supported evidence" in parsed.answer
            assert parsed.answer != ""
    finally:
        _load_model.cache_clear()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except (ImportError, RuntimeError):
            pass
