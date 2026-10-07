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

Model weights are downloaded on first use and cached in the repository-root
`model-cache` directory, outside the backend source tree watched by Uvicorn.
Override it with `MODEL_CACHE_PATH` if desired. The default Qwen 0.5B model is
selected for modest local GPU memory. `LLM_MODEL_ID`,
`EMBEDDING_MODEL_ID`, and `RERANKER_MODEL_ID` can be changed in the root `.env`.
`LLM_ADAPTER_PATH` defaults to
`backend\models\dietary-assistant`.
An existing adapter without a `curated_training_manifest.json` is deliberately
not loaded; retrain from a human-reviewed dataset to activate a LoRA adapter.

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

By default, ingestion reads RAG chunks from
`knowledge-base\processed\v2\malaysian_food_rag_v2_multichunk.jsonl`. Set
`RAG_CHUNKS_PATH` in the root `.env` or pass `--rag-chunks` to select another
multichunk JSONL file. Structured food imports are separate and optional:

```powershell
python -m app.scripts.ingest_knowledge `
  --foods ..\knowledge-base\processed\v2\malaysian_food_rag_v2_foods.csv
```

The expanded JSON and CSV are preserved under
`knowledge-base\raw\original`; the current multichunk JSONL is under
`knowledge-base\processed\v2`. The v2 foods CSV, human-inspection multichunk
CSV, and standalone SQLite file were not present when this layout was set up,
so they were not generated or added. If supplied later, the SQLite file belongs
under `knowledge-base\database` and remains a standalone reference, never an
ingestion source.

Food CSV or JSON records populate PostgreSQL `food_nutrition` only. Multichunk
JSONL records populate PostgreSQL `document_chunks` and the local Qdrant index
only; they are not parsed into structured food records. The human-inspection
CSV and standalone SQLite reference database are not ingestion inputs. Importing
the default v2 chunk file also clears Qdrant entries from the previous expanded
dataset, but preserves PostgreSQL food rows; the RAG import does not mutate the
structured food dataset.

It also accepts curated PDF, text, or Markdown references:

```powershell
python -m app.scripts.ingest_knowledge `
  --skip-rag-chunks `
  --document ..\knowledge-base\malaysian-guidelines.pdf `
  --document-source-type government_guideline `
  --document-authority 0.95 `
  --document-source-url https://example.gov.my/guideline
```

The command stores structured foods and document chunks in PostgreSQL and
embeddings in the local Qdrant store under `backend\data\qdrant`. Re-running
ingestion replaces records for the same source. The API's Swagger UI is at
`http://localhost:8000/docs`.

### Rebuilding development knowledge data

Stop Uvicorn first so it releases the local Qdrant store. The explicit reset
below deletes only `document_chunks`, `food_nutrition`, and
`knowledge_sources` from the configured database, and deletes only the
`dietary_evidence` Qdrant collection. It does not touch users, profiles, chat
sessions, or messages:

```powershell
python -m app.scripts.reset_knowledge --confirm
```

Then import the structured food CSV when one is available, followed by the
default RAG chunks:

```powershell
python -m app.scripts.ingest_knowledge `
  --foods ..\knowledge-base\processed\v2\malaysian_food_rag_v2_foods.csv
```

If there is no processed food CSV yet, run
`python -m app.scripts.ingest_knowledge` without `--foods`; it will import the
default multichunk JSONL into PostgreSQL and create a fresh Qdrant collection
while embedding the chunks. The SQLite reference database is never loaded by
this application.

The assistant uses structured food records, hybrid BM25/dense retrieval, RRF,
a cross-encoder reranker, deterministic dietary filtering, evidence
compression, local JSON generation, and citation checks. It does not expose an
uncalibrated numeric confidence score.
Authenticated conversation history is stored per user. The long-term user
profile and short-term conversation summary are kept separate. Each turn's
retrieval and factual answer use the current question only; restate the food
when asking a follow-up so facts from an earlier food cannot leak into the
answer. Structured food lookups use an exact food-name match and return
recorded values directly without calling the language model.

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
adapter; it does not train a foundation model from scratch. Fine-tuning requires
a human-reviewed UTF-8 JSONL file with at least four examples. Do not use
automatically generated database rows as fine-tuning answers. Each record must
set `"reviewed": true` after checking its prompt and answer. Supported formats:
`{"reviewed":true,"prompt":"...","response":"..."}`,
`{"reviewed":true,"instruction":"...","input":"...","output":"..."}`,
`{"reviewed":true,"question":"...","answer":"..."}`, or a v2
`{"reviewed":true,"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}`
conversation record.

```powershell
python -m app.ml.train_adapter --data .\data\training\curated.jsonl
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

Measure rule-based query intent classification against the manually labeled
100-question baseline:

```powershell
python -m app.scripts.evaluate_query_intents
```

The report includes overall and per-intent accuracy, a confusion summary, and
the misclassified questions. Update the evaluation labels manually as the
baseline improves; the benchmark does not replace the rule-based processor.
