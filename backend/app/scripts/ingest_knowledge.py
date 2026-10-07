import argparse
from pathlib import Path

from app.core.config import settings
from app.core.database import SessionLocal
from app.services.ingestion import (
    ingest_document,
    ingest_food_dataset,
    ingest_rag_chunks,
)
from app.services.vector_store import close_vector_client


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load structured food data, RAG chunks, and reference documents."
    )
    parser.add_argument(
        "--foods",
        type=Path,
        default=settings.food_dataset_path,
        help="Optional structured food dataset (.csv or .json).",
    )
    parser.add_argument(
        "--rag-chunks",
        type=Path,
        default=settings.rag_chunks_path,
        help="RAG chunk dataset (.jsonl); imported into document_chunks and Qdrant.",
    )
    parser.add_argument(
        "--skip-rag-chunks",
        action="store_true",
        help="Skip the default RAG chunk dataset.",
    )
    parser.add_argument(
        "--document",
        type=Path,
        action="append",
        default=[],
        help="A .pdf, .txt, or .md dietary reference; may be repeated.",
    )
    parser.add_argument(
        "--document-source-type",
        default="user_uploaded_document",
        help="Provenance label for the provided documents.",
    )
    parser.add_argument("--document-source-url")
    parser.add_argument("--document-authority", type=float)
    args = parser.parse_args()

    try:
        with SessionLocal() as db:
            if args.foods is not None:
                count = ingest_food_dataset(db, args.foods)
                print(f"Imported {count} structured food records from {args.foods}.")
            if not args.skip_rag_chunks:
                count = ingest_rag_chunks(db, args.rag_chunks)
                print(f"Indexed {count} RAG chunks from {args.rag_chunks}.")
            for document in args.document:
                count = ingest_document(
                    db,
                    document,
                    source_type=args.document_source_type,
                    source_url=args.document_source_url,
                    authority_score=args.document_authority,
                )
                print(f"Indexed {count} chunks from {document}.")
    finally:
        close_vector_client()


if __name__ == "__main__":
    main()
