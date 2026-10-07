import argparse

from sqlalchemy import delete
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.database import SessionLocal
from app.models.knowledge import DocumentChunk, FoodNutrition, KnowledgeSource
from app.services.retrieval import invalidate_retrieval_cache
from app.services.vector_store import close_vector_client, reset_collection


def reset_knowledge_tables(db: Session) -> None:
    db.execute(delete(DocumentChunk))
    db.execute(delete(FoodNutrition))
    db.execute(delete(KnowledgeSource))
    db.commit()
    invalidate_retrieval_cache()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Development-only reset of knowledge tables and the "
            "dietary_evidence Qdrant collection. User accounts and chat history "
            "are preserved."
        )
    )
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Confirm deletion of all knowledge sources, foods, chunks, and vectors.",
    )
    args = parser.parse_args()
    if not args.confirm:
        parser.error(
            "This reset is destructive. Rerun with --confirm after verifying "
            "the configured development database."
        )

    database = make_url(settings.database_url)
    print(
        "Resetting knowledge data for "
        f"database={database.database!r}, host={database.host!r}; "
        f"Qdrant path={settings.qdrant_path}."
    )
    try:
        reset_collection()
        with SessionLocal() as db:
            reset_knowledge_tables(db)
    finally:
        close_vector_client()

    print(
        "Knowledge tables and the dietary_evidence collection were reset. "
        "Users, profiles, sessions, and messages were preserved."
    )


if __name__ == "__main__":
    main()
