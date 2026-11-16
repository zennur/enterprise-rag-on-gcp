# Load testing

The Locust file lives at `load/locustfile.py`. It does not record results. Fill the table below after a run. Numbers from the fake provider describe the service loop, not Vertex AI.

## What to measure

Run two passes and keep them separate.

1. **Platform pass.** Local process or Cloud Run with `ALLOW_FAKE_PROVIDER=true` and the in-process store. This shows framework overhead, concurrency, and the rate limiter. It is not a model benchmark.
2. **Vertex pass.** A deployed API with `EMBEDDING_PROVIDER=vertex` and `GENERATION_PROVIDER=vertex`, against a small corpus, with `api_max_instances` and `RATE_LIMIT_RPS` set so the test cannot outrun the quota. This is the pass that informs cost.

For each pass record:

- p50, p95, and p99 latency of `POST /v1/query`
- successful requests per second
- error rate, split by HTTP status (`429` means the limiter or the in-flight gate, `503` means a provider timeout or a failed dependency)
- on the Vertex pass, token and character usage from the `embedding_usage` and `generation_usage` log events

Locust prints latency percentiles and request counts. It does not know Cloud Run or Vertex prices.

## How to run locally

```bash
pip install -e ".[dev,load]"
uvicorn rag_service.main:app --port 8080
```

In another shell, with the corpus already ingested (`python eval/run_eval.py` does not leave a server up; post the sample documents, or point Locust at a process that has ingested them):

```bash
locust -f load/locustfile.py --headless \
  -u 20 -r 5 -t 60s \
  --host http://127.0.0.1:8080
```

`-u` is users, `-r` is the spawn rate, `-t` is the duration. Start lower than 20 users. The default `RATE_LIMIT_RPS` is 5 per process with a burst of 10, so a larger Locust run measures the limiter on purpose. Raise `RATE_LIMIT_RPS` and `MAX_INFLIGHT` when the goal is the model path instead of the shed behavior.

Against Cloud Run, send an identity token and keep the user count inside the instance cap:

```bash
TOKEN="$(gcloud auth print-identity-token)"
locust -f load/locustfile.py --headless \
  -u 10 -r 2 -t 60s \
  --host "$API_URI" \
  --header "Authorization: Bearer $TOKEN"
```

Locust's `--header` flag applies to every request in current Locust releases. If the installed Locust does not accept it, set the header in a small subclass of `RagUser` locally. Do not commit tokens.

## Cost per 1,000 queries

Do not copy prices into this file. Read them from the current Cloud Run, Vertex AI, Pub/Sub, and (if used) AlloyDB or Vector Search price lists on the day of the run.

```text
per_query =
    cloud_run_vcpu_seconds * vcpu_price
  + cloud_run_gib_seconds * gib_seconds_price
  + cloud_run_request_price
  + embedding_characters * embedding_price
  + prompt_tokens * gemini_input_price
  + output_tokens * gemini_output_price
  + vector_query_price

cost_per_1k = per_query * 1000
```

Take token counts from logs, not from a guess. Cache hits should show `cached=true` on `query_answered` and should not add a generation call. A rewrite adds one generation call and one extra embedding call; `rewritten=true` marks those queries.

Vector Search is billed largely by node-hours even when idle. AlloyDB is billed by instance shape. The in-process store adds no database line. That difference is why the default Terraform stack does not create either database.

## Results

Fill this in after measuring. Leave the placeholders until then.

| | Local fake provider | Cloud Run + Vertex |
| --- | --- | --- |
| Date | _to be measured_ | _to be measured_ |
| Git SHA | _to be measured_ | _to be measured_ |
| Traffic shape (users, duration) | _to be measured_ | _to be measured_ |
| Instance size and concurrency | _to be measured_ | _to be measured_ |
| p50 latency | _to be measured_ | _to be measured_ |
| p95 latency | _to be measured_ | _to be measured_ |
| p99 latency | _to be measured_ | _to be measured_ |
| Successful queries / second | _to be measured_ | _to be measured_ |
| Error rate | _to be measured_ | _to be measured_ |
| Embedding characters / query | n/a | _to be measured_ |
| Gemini input tokens / query | n/a | _to be measured_ |
| Gemini output tokens / query | n/a | _to be measured_ |
| Cost per 1,000 queries | n/a | _to be measured_ |

The local column is not a prediction of the Vertex column.
