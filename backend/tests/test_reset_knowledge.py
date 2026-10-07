from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.core.database import Base
from app.models.knowledge import (
    ChatMessage,
    ChatSession,
    DocumentChunk,
    FoodNutrition,
    KnowledgeSource,
)
from app.models.user import User
from app.models.user_profile import UserProfile
from app.scripts.reset_knowledge import reset_knowledge_tables


def test_knowledge_reset_preserves_users_profiles_and_chat_history() -> None:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    try:
        with Session(engine) as db:
            user = User(email="person@example.com", password_hash="hashed")
            db.add(user)
            db.flush()
            db.add(UserProfile(user_id=user.id, full_name="Person"))
            chat = ChatSession(user_id=user.id, title="Saved conversation")
            db.add(chat)
            db.flush()
            db.add(
                ChatMessage(
                    session_id=chat.id,
                    role="user",
                    content="Hello",
                )
            )
            source = KnowledgeSource(
                title="Nutrition source",
                source_type="user_uploaded_csv",
                authority_score=0.3,
            )
            db.add(source)
            db.flush()
            db.add(
                FoodNutrition(
                    source_id=source.id,
                    source_record_id="food-1",
                    name="Tempeh",
                )
            )
            db.add(
                DocumentChunk(
                    source_id=source.id,
                    chunk_index=0,
                    content="Tempeh nutrition evidence.",
                )
            )
            db.commit()

            reset_knowledge_tables(db)

            assert db.scalars(select(KnowledgeSource)).all() == []
            assert db.scalars(select(FoodNutrition)).all() == []
            assert db.scalars(select(DocumentChunk)).all() == []
            assert len(db.scalars(select(User)).all()) == 1
            assert len(db.scalars(select(UserProfile)).all()) == 1
            assert len(db.scalars(select(ChatSession)).all()) == 1
            assert len(db.scalars(select(ChatMessage)).all()) == 1
    finally:
        engine.dispose()
