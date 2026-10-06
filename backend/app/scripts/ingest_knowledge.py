import argparse
from pathlib import Path

from app.core.database import SessionLocal
from app.services.ingestion import (
    DEFAULT_NUTRITION_DATASET,
    ingest_document,
    ingest_nutrition_dataset,
)
from app.services.vector_store import close_vector_client


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load nutrition records and trusted dietary reference documents."
    )
    parser.add_argument(
        "--nutrition",
        type=Path,
        default=DEFAULT_NUTRITION_DATASET,
        help="Path to a .csv or .json food nutrition dataset.",
    )
    parser.add_argument(
        "--skip-nutrition",
        action="store_true",
        help="Skip the default nutrition dataset.",
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
            if not args.skip_nutrition:
                count = ingest_nutrition_dataset(db, args.nutrition)
                print(f"Imported {count} nutrition records from {args.nutrition}.")
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
