# Dietary assistant backend

## Runtime and dependencies

The Python API, retrieval models, local vector store, and fine-tuning tools run
from `backend\.venv`. No Ollama service is used. The language model is loaded
directly with Hugging Face Transformers; the base model and optional LoRA
adapter are cached locally.

From PowerShell at the repository root:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
& .\backend\.venv\Scripts\Activate.ps1
python -m pip install -r .\backend\requirements-ai.txt
```

Model weights are downloaded on first use and cached under
`backend\.model-cache`. The default Qwen 0.5B model is selected for modest local
GPU memory. `LLM_MODEL_ID`, `EMBEDDING_MODEL_ID`, and `RERANKER_MODEL_ID` can be
changed in the root `.env`. `LLM_ADAPTER_PATH` defaults to
`backend\models\dietary-assistant`.

## Database, ingestion, and API

With PostgreSQL configured through the root `.env`, from `backend`:

```powershell
alembic upgrade head
python -m app.scripts.ingest_knowledge
python -m uvicorn app.main:app --reload --reload-dir .\app
```

Run Uvicorn from the `backend` directory with the backend virtual environment
active. Limiting reload watching to `.\app` prevents writes to runtime data such
as `.\data\qdrant` from restarting the server while handling a chat request.
If Uvicorn is already running, stop it with `Ctrl+C` and restart it with the
command above.

The ingestion command imports the root
`database\malaysian_food_rag_dataset_EXPANDED_full.json` by default. It also
accepts curated PDF, text, or Markdown references:

```powershell
python -m app.scripts.ingest_knowledge `
  --skip-nutrition `
  --document ..\knowledge-base\malaysian-guidelines.pdf `
  --document-source-type government_guideline `
  --document-authority 0.95 `
  --document-source-url https://example.gov.my/guideline
```

The command stores structured foods and document chunks in PostgreSQL and
embeddings in the local Qdrant store under `backend\data\qdrant`. Re-running
ingestion replaces records for the same source. The API's Swagger UI is at
`http://localhost:8000/docs`.

The assistant uses structured food records, hybrid BM25/dense retrieval, RRF,
a cross-encoder reranker, deterministic dietary filtering, evidence
compression, local JSON generation, citation checks, and confidence scoring.
Authenticated conversation history is stored per user. The long-term user
profile and short-term conversation summary are kept separate.

## Dataset safety and provenance

Imported `source_type` values affect the stored authority score. Generated
estimates and generated variants receive low authority. User-uploaded records
are not treated as verified simply because they have a source URL. Review
dataset facts against authoritative sources before relying on them.

The supplied nutrition dataset has no verified allergen or dietary-compatibility
columns. For a personalized recommendation with allergy or diet constraints,
foods without explicit verified declarations are excluded. If no compatible
record remains, the API refuses to recommend a specific food instead of
assuming that missing data means safe.

## Fine-tuning the local model

The project uses a small open-weight base model with a locally saved LoRA
adapter; it does not train a foundation model from scratch. Build a starter JSONL file from the ingested database, or use a UTF-8 curated
JSONL file with at least four examples. Generated examples retain source IDs and
the existing records' authority limitations; they are not validation of the
source facts:

```powershell
python -m app.ml.build_training_data
python -m app.ml.train_adapter --data .\data\training\dietary_assistant.jsonl
```

Training downloads the configured base model into the local cache and writes
the adapter to `backend\models\dietary-assistant`. The adapter is loaded
automatically when present. Grounding, citations, hard constraints, and
verification remain application responsibilities; fine-tuning is not a
replacement for retrieval or verified training examples.

## Tests

From `backend` with its virtual environment active:

```powershell
pytest
```
