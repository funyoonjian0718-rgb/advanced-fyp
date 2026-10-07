import asyncio
from collections.abc import Iterator

import httpx
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.core.database import Base, get_db
from app.main import app
from app.models.knowledge import ChatMessage, ChatSession
from app.models.user import User
from app.services.auth_service import create_access_token, hash_password


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
