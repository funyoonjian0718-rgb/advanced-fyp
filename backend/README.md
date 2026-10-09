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
Set `LLM_OFFLINE=true` to require cached model files during normal loading.
The opt-in raw-generation smoke test always enables offline mode and does not
run claim verification:

```powershell
python -m app.scripts.smoke_test_local_model
```

Run it from `backend` with the backend virtual environment active. It loads the
configured model and reports generation status without printing generated
content.

## Answer-quality evaluation

The source-grounded answer-quality cases are separate from retrieval-ranking
labels in `tests\data\answer_quality_eval.jsonl`. Run the offline dataset and
metric smoke checks from `backend`:

```powershell
.\.venv\Scripts\python.exe -m app.scripts.evaluate_answer_quality
```

This validates every food/chunk reference against the processed knowledge-base
files and runs all cases through the chat API with deterministic mocked model
claims. It reports fixture-path metrics to verify the API and verifier wiring;
these are not model-quality scores. To explicitly evaluate final verified chat
API responses with the configured cached model, use:

```powershell
$env:LLM_OFFLINE = "true"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
.\.venv\Scripts\python.exe -m app.scripts.evaluate_answer_quality --real-model
```

Use `--case-id <dataset-id>` to isolate a case. Non-200 API responses remain in
`case_results` with `actual_outcome: "request_failed"` and sanitized HTTP
details; they count as incorrect outcomes and the evaluator exits nonzero.

The real-model mode uses a temporary SQLite database and controlled evidence
loaded from the processed food CSV and multichunk JSONL. It runs the chat API
and existing claim verifier, does not access PostgreSQL or Qdrant, and refuses
to fetch uncached model files. Citation support, numeric correctness, and
unsupported-claim counts are automated through the existing verifier.
Guideline paraphrase entailment and allergy safety still require human review;
the safety flag is a conservative heuristic, not a medical safety assessment.

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

RRF uses the reciprocal-rank contribution `1 / (60 + rank)` and deduplicates
evidence by stable evidence ID. `retrieval_score` stores the highest raw BM25
or dense-retrieval score; `rrf_score` stores the fused reciprocal-rank score.
The cross-encoder determines the final candidate order; if reranking or
verification fails, the API returns an error instead of falling back to an
unverified draft. Nutrition claim checks bind reported values to a named food
and its own nutrient fields. Guideline claims are retained only when they
exactly match a sentence in a cited document and are attributed to that source.
Citations are drawn only from evidence that supports retained claims. The
current response schema has a flat citation list, so it cannot expose
claim-to-citation associations directly.

## Dataset safety and provenance

Imported `source_type` values affect the stored authority score. Generated
estimates and generated variants receive low authority. User-uploaded records
are not treated as verified simply because they have a source URL. Review
dataset facts against authoritative sources before relying on them.

The nutrition database stores allergen and dietary-label values but has no
review-status or provenance fields for those values. Populated fields are
therefore not treated as reviewed: `*_data_populated` indicates presence while
`*_data_verified` remains false for database-derived food evidence. A reviewed
attestation must be explicit in the evidence metadata; old vector payloads that
only contain the former populated-as-verified flags are treated as unverified.
For a personalized recommendation with allergy or diet constraints, records
without verified declarations are excluded. If no compatible record remains,
the API refuses to recommend a specific food instead of assuming that missing
or unreviewed data means safe.

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

Compare BM25-only, dense-only, RRF-fused, and reranked retrieval against the
stable evidence labels in the Malaysian question set:

```powershell
python -m app.scripts.evaluate_retrieval --top-k 8 --candidate-limit 30
```

This retrieval evaluation requires the configured PostgreSQL database, imported
food and document evidence, the local Qdrant collection, and the configured
embedding and reranker models. BM25-only and dense-only comparisons omit
numeric-filter cases because neither applies the same SQL constraints. Their
single-stage baselines use the primary query; RRF uses the full query set and
subquestions. Safety and insufficient-evidence cases without positive
retrieval labels are listed as skipped. The benchmark reports retrieval
metrics only; it does not measure generation faithfulness or citation accuracy.

The evaluator deduplicates retrieved evidence keys before ranking metrics.
`relevant_evidence_keys` gives binary relevance; optional `relevance_grades`
maps each relevant key to a grade from 1 to 3 and uses gain `2^grade - 1` for
nDCG. Precision@K uses K as its denominator, Recall@K uses all unique relevant
labels for the query, MRR uses the first relevant rank in the full ranking, and
nDCG@K normalizes against the sorted relevance grades truncated at K. Macro
metrics are unweighted averages of per-query scores; category scores are
unweighted averages of their per-query scores. Food labels must use the
`source_record_id` identity emitted by the evaluator. Manually verify existing
`food:1` and `food:2` labels against the imported food records before
interpreting comparison-case scores.
