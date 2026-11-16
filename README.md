# Enterprise RAG on Google Cloud

Internal documents are a poor fit for a chat model that answers from memory. The useful system retrieves a few passages, answers only from those passages, and says when it does not know. Ingestion has to keep up with new documents without sitting on the query path, and the query path has to have a budget: timeouts, a concurrency cap, and a bill that does not grow without a ceiling.

This repository is a small v1 of that system, shaped for Cloud Run and Vertex AI. It runs locally with a deterministic fake provider, so tests and CI do not need GCP credentials. It has not been deployed from this checkout, and it does not claim production traffic or measured latency.

## Architecture

One image, two Cloud Run services. The API answers questions and publishes documents. The worker turns documents into chunks and vectors. The vector store is an interface with three implementations.

```mermaid
flowchart TB
  client[Client]
  subgraph run [Cloud Run]
    api[API service]
    worker[Worker service]
  end
  cache[Process-local response cache]
  topic[Pub/Sub documents topic]
  dlq[Dead-letter topic]
  embed[Vertex AI embeddings]
  gemini[Vertex AI Gemini]
  subgraph stores [Vector store]
    memory[In-process]
    pg[AlloyDB pgvector]
    ann[Vertex AI Vector Search plus a chunk catalog]
  end

  client --> api
  api --> cache
  api --> embed
  api --> gemini
  api --> stores
  api --> topic
  topic --> worker
  topic --> dlq
  worker --> embed
  worker --> stores
```

A query checks an exact cache, embeds the question, and searches the store. If the top score is below the rewrite threshold, the generator rewrites the query once and the store is searched again. The loop then either abstains or asks Gemini to answer from the passages it was given. There is no open-ended tool router. `agent_max_steps` is 2.

```mermaid
sequenceDiagram
  participant Client
  participant API
  participant Cache
  participant Embeddings
  participant Store
  participant Gemini
  Client->>API: POST /v1/query
  API->>Cache: exact key
  alt cache miss
    API->>Embeddings: embed query
    API->>Store: nearest chunks
    opt top score below rewrite threshold
      API->>Gemini: rewrite once
      API->>Embeddings: embed rewrite
      API->>Store: nearest chunks
    end
    alt top score below abstain threshold
      API-->>Client: abstain
    else
      API->>Gemini: answer from passages
      API->>Cache: store
      API-->>Client: answer and citations
    end
  else cache hit
    API-->>Client: cached answer
  end
```

Ingestion accepts a batch on `POST /v1/documents` and returns 202. With `INGEST_TRANSPORT=inline` the API embeds before it responds, which is the local and test mode. With `pubsub`, the API publishes one message per document, ordered by document id, and the worker does the embedding. A second delivery of the same bytes does not call the embedding model again: the document hash stored with the chunks is the idempotency key. Chunk ids include a content hash, and a changed document replaces its previous chunks.

The Terraform subscription is a push subscription, so the worker scales to zero with the API. Push delivers one document per request; embeddings are still batched across that document's chunks. `python -m rag_service.worker --pull` is the alternative when a batch should span documents. That process has to stay up. `python -m rag_service.worker --input docs.jsonl` is the one-shot shape for a Cloud Run job. A memory store is discarded when that process exits.

## Design decisions

**Cloud Run, one image, two services.** The API and the worker scale on different signals. Query concurrency and ingest volume should not share a single instance pool. The trade-off is cold start, a request timeout, and no durable local disk. Min instances default to 0 so an idle deploy does not bill CPU. Set the API minimum to 1 when cold start matters. Startup CPU boost is on for the API.

**Provider and store interfaces.** Tests use a hashing embedder and an extractive answerer. Vertex AI (`gemini-embedding-001` at 768 dimensions, `gemini-2.5-flash`) is the production pair and is imported only when selected. On Cloud Run the process refuses to become ready if the fake provider or the memory store is selected, unless `ALLOW_FAKE_PROVIDER` or `ALLOW_MEMORY_STORE` is set on purpose. That keeps a missing env var from silently serving hash embeddings.

**pgvector first, Vector Search when the corpus earns it.** AlloyDB (or any Postgres with pgvector) stores the passage and the vector in one transactional replace, and it is consistent as soon as the commit lands. Vertex AI Vector Search returns ids and distances, so passage text has to live in a catalog. This repo uses a Postgres catalog when `DATABASE_URL` is set and an in-process catalog otherwise. The in-process catalog does not survive a new instance. Streaming index updates are briefly stale; a neighbor id missing from the catalog is skipped. Terraform does not create AlloyDB or a Vector Search index. Both are billed while idle, and a portfolio apply should not start that bill by accident.

**The agent is a bounded loop.** A weak first search may spend one rewrite. A score under the abstain threshold returns a fixed refusal and does not call the generator. That is the hallucination control and a cost control. The rewrite threshold defaults to the abstain threshold, so the extra call happens only when the first pass would have refused. Citations are the passages that were put in the prompt, which is an audit trail the model's bracket formatting cannot drop.

**Cache is local and TTL-bounded.** The exact key is the normalized query, top-k, model names, and the local index generation. Another instance can still serve an older answer until `CACHE_TTL_S` (default 300 seconds). A semantic cache is implemented as a linear scan over a capped list and is off. A wrong semantic hit is worse than a miss, and the scan is only reasonable for a small hot set. A shared cache such as Memorystore is the step that makes the hit rate follow the fleet instead of the instance.

**Per-instance backpressure, platform budget.** The token bucket and the in-flight gate live in the process. They shed load on one instance; they are not a global quota. Cloud Run concurrency on the API is 32 and `MAX_INFLIGHT` defaults to 24, so health checks still fit while model calls are capped. Max instances in Terraform are the spend ceiling. Edge rate limiting belongs on Cloud Armor or an API gateway.

**Retries.** Provider calls use a timeout and a small number of attempts with jitter. Retries cover timeouts and transient statuses (unavailable, resource exhausted, deadline, internal, aborted). A bad argument is not retried. `asyncio.wait_for` does not cancel work already running in a thread, so a timed-out Vertex call can still finish in the background. The in-flight cap bounds how many of those can pile up.

**Auth.** The Terraform API does not grant `allUsers`. Callers need `run.invoker`, which is the right place for IAP or an internal load balancer later. An optional `X-API-Key` is defense in depth for a drill that does open the service. The worker accepts Pub/Sub push only with an OIDC token whose audience is `PUBSUB_PUSH_AUDIENCE`, and checks the email when `PUBSUB_PUSH_SERVICE_ACCOUNT` is set. Invalid push payloads fail the request so Pub/Sub can dead-letter them after five attempts. Logs record document ids and sizes, not document text, and spans record the query length, not the query.

## Scaling notes

Queries are I/O bound on Vertex. 32 concurrent requests at 1 vCPU is a starting cap, not a measured optimum. If p95 grows, lower concurrency before adding CPU. The worker concurrency is 4 because each request holds an embedding batch.

Pub/Sub absorbs ingest spikes; the API returns 202 without embedding. The dead-letter topic keeps messages that still fail after five deliveries, and the subscription pauses further ordered messages for a document id while an earlier one is outstanding. That is the trade-off for per-document ordering.

The practical quota is Vertex, not Cloud Run. `api_max_instances` times the in-flight cap should stay under the project's embedding and Gemini quotas. Cache hits are the cheap way to cut repeat generation cost. Abstention is the cheap way to avoid a generation call the corpus cannot support.

## Cost

The expensive lines are Gemini output tokens, embedding characters, and any database that bills while idle. Cloud Run bills vCPU and memory per request-second when CPU is idle between requests, which is how both services are configured. Pub/Sub is a small per-GiB line next to those.

A cost per 1,000 queries needs measured token counts. The API logs `embedding_usage`, `generation_usage`, and `query_answered` (with `cached`, `rewritten`, and call counts). The worksheet and the empty results table are in [docs/load-testing.md](docs/load-testing.md). This README does not invent prices or latencies.

`text-embedding-005` is the smaller English embedding model if `gemini-embedding-001` is more than the corpus needs. Switching models is an env change, followed by a reindex, because the vectors will not match.

## Layout

```text
src/rag_service/     API, worker, providers, stores, ingest, agent
tests/               unit tests, no network and no GCP
eval/                synthetic corpus, questions, and the harness
load/locustfile.py   load script
docs/load-testing.md what to measure, with empty result cells
terraform/           Cloud Run, Pub/Sub, Artifact Registry, Secret Manager, IAM
```

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
uvicorn rag_service.main:app --port 8080
```

The defaults are the fake provider, the in-process store, and inline ingest.

```bash
curl -s localhost:8080/healthz
curl -s localhost:8080/v1/documents -H 'content-type: application/json' -d '{
  "documents": [{
    "document_id": "expenses",
    "title": "Expense policy",
    "text": "The mealcap is seventyfive dollars per person."
  }]
}'
curl -s localhost:8080/v1/query -H 'content-type: application/json' \
  -d '{"query": "What is the mealcap?"}'
```

`processed_inline` is true in this mode, so the query can run immediately. A second identical query returns `"cached": true`.

Point the same process at Vertex by exporting `EMBEDDING_PROVIDER=vertex`, `GENERATION_PROVIDER=vertex`, `GCP_PROJECT`, and `GCP_LOCATION`, with Application Default Credentials. That is not required for tests.

```bash
ruff check .
ruff format --check .
mypy
pytest -q
```

The image serves the same process. It installs the GCP libraries so Cloud Run can call Vertex. A local container still boots in fake mode:

```bash
docker build -t rag-service .
docker run --rm -p 8080:8080 rag-service
```

## Evaluation

`eval/corpus.jsonl` and `eval/questions.jsonl` are a synthetic company handbook written for this harness. There is no proprietary data in them.

```bash
python eval/run_eval.py
python eval/run_eval.py --strict
```

The report prints retrieval hit rate (expected document present in the top-k), decision accuracy (abstain matches the label), and mean lexical groundedness. Groundedness is the share of content words in the answer that also appear in the cited excerpts, after bracketed chunk ids are removed. It checks that the answer stays on the supplied text. It does not judge usefulness, and it is not an LLM judge. The fake answerer copies context sentences, so a high groundedness score on this path means the wiring is intact. A Vertex run is the one that can fail it. `--strict` exits 1 below a perfect hit rate, a perfect decision score, or 0.8 groundedness. The default run always exits 0 so a measurement against Vertex is still a measurement.

The unit test `tests/test_eval.py` runs this sample through the fake provider and expects it to clear those bars. That assertion is a regression check of the harness, not a quality claim about Gemini.

## Load testing

See [docs/load-testing.md](docs/load-testing.md). The Locust file is `load/locustfile.py`. Results are intentionally blank.

## Deploy

The stack in `terraform/` creates:

- Artifact Registry repository
- API and worker Cloud Run services, separate service accounts
- Pub/Sub topic, push subscription, and dead-letter topic
- Secret Manager secret for a database URL
- least privilege bindings for Pub/Sub, the secret, logging, metrics, and traces

It does not create AlloyDB, a Vector Search index, or a VPC connector. `roles/aiplatform.user` is granted on the project because that is the narrowest predefined role covering Gemini, embeddings, and index updates together. A custom role is a follow-up once a real project shows the exact methods in the audit log.

Build and push before apply. Cloud Run will fail the revision if the image is missing.

```bash
gcloud auth configure-docker REGION-docker.pkg.dev
docker build -t REGION-docker.pkg.dev/PROJECT/rag/rag-service:latest .
docker push REGION-docker.pkg.dev/PROJECT/rag/rag-service:latest

cd terraform
terraform init
terraform plan -var project_id=PROJECT
terraform apply -var project_id=PROJECT
```

The default `vector_store` is `memory`, and the services set `ALLOW_MEMORY_STORE=true` so they become ready. Instances do not share data in that mode. It is there so a first apply does not create a billable database. Switch to `pgvector` or `vertex` before loading a corpus that has to survive a new instance.

For `pgvector`, add a secret version first. The URL should point at AlloyDB through the Auth Proxy or the managed connector; the application uses `asyncpg` and creates the `vector` extension and tables on startup. A locked-down database should run those statements from a migration role instead and keep the runtime role on DML.

```bash
printf '%s' 'postgres://USER:PASSWORD@HOST:5432/rag' | \
  gcloud secrets versions add rag-database-url --data-file=-
```

Then apply with `-var vector_store=pgvector`. For Vector Search, pass the existing index, endpoint, and deployed index id, and set `vertex_catalog_database=true` if chunk text should live in Postgres. The index itself should use cosine distance, which is the default `VECTOR_SEARCH_DISTANCE`.

State is local. Move it to a GCS backend before anyone else applies this. `deletion_protection` defaults to false so a trial stack can be destroyed.

`terraform fmt`, `terraform init -backend=false`, and `terraform validate` run in CI. CI does not apply.

To call the API after apply, the caller needs `roles/run.invoker` on `rag-api`. The worker is not in that grant; only the push service account can invoke it.

## Configuration

Settings are environment variables matching the field names in `src/rag_service/config.py`. The ones that change the shape of a deployment:

| Variable | Default | Role |
| --- | --- | --- |
| `ROLE` | `all` | `api`, `worker`, or `all` |
| `EMBEDDING_PROVIDER` / `GENERATION_PROVIDER` | `fake` | `vertex` on Cloud Run |
| `VECTOR_STORE` | `memory` | `pgvector` or `vertex` for shared state |
| `INGEST_TRANSPORT` | `inline` | `pubsub` on the API |
| `GCP_PROJECT` / `GCP_LOCATION` | empty / `us-central1` | Vertex and Vector Search |
| `PUBSUB_TOPIC` | empty | required for `pubsub` transport |
| `PUBSUB_PUSH_AUDIENCE` | empty | required on the Cloud Run worker |
| `DATABASE_URL` | empty | pgvector, or the Vector Search catalog |
| `ABSTAIN_THRESHOLD` / `REWRITE_THRESHOLD` | `0.2` / `0.2` | refusal and the single rewrite |
| `RATE_LIMIT_RPS` / `RATE_LIMIT_BURST` / `MAX_INFLIGHT` | `5` / `10` / `24` | per-instance shed |
| `CACHE_TTL_S` / `SEMANTIC_CACHE_ENABLED` | `300` / `false` | staleness bound; semantic cache off |
| `API_KEYS` | empty | optional comma-separated keys |
| `OTEL_EXPORTER_OTLP_ENDPOINT` | empty | traces stay in-process until this is set |
| `LOG_FORMAT` | `console` | `json` in the Terraform services |

## Roadmap

- Record a real load test in `docs/load-testing.md`, then resize concurrency, min instances, and the rate limit from those numbers.
- Move the response cache to Memorystore and key it by a corpus epoch stored next to the chunks, so every instance invalidates together.
- Put Cloud Armor or an API gateway in front of the global limit, and IAP in front of user identity.
- Replace the runtime DDL with a migration and a DML-only database role. Use the AlloyDB connector with IAM auth instead of a URL in Secret Manager.
- Add a custom Vertex IAM role from the methods the audit log actually shows.
- Add an LLM-judged groundedness score next to the lexical one, on a sample large enough to mean something.
- Reindex with a heading-aware chunker once the lexical hit rate on real documents says the windows are the problem.

## License

MIT. See [LICENSE](LICENSE).
