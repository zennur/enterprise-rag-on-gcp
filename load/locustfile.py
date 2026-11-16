"""Load for the query path.

Install the extra first: pip install -e ".[load]"

The default mix is queries plus health checks. Ingest is a separate task with
low weight and a fixed document id so repeats stay idempotent. See
docs/load-testing.md before treating any of these numbers as capacity.
"""

from __future__ import annotations

import random

from locust import HttpUser, between, task

QUERIES = [
    "Who gets paged for a sev1 incident?",
    "How many days are in the vacationaccrual?",
    "How long is application logretention?",
    "What is the mealcap per person?",
    "How quickly does securityreview finish?",
    "What sourdough hydration does the bakery use?",
]


class RagUser(HttpUser):
    wait_time = between(0.05, 0.2)

    @task(10)
    def query(self) -> None:
        self.client.post(
            "/v1/query",
            json={"query": random.choice(QUERIES), "top_k": 4},
            name="/v1/query",
        )

    @task(1)
    def health(self) -> None:
        self.client.get("/healthz", name="/healthz")

    @task(1)
    def ingest(self) -> None:
        self.client.post(
            "/v1/documents",
            json={
                "documents": [
                    {
                        "document_id": "load-smoke",
                        "title": "Load smoke",
                        "text": "This load-test document says the canary token is loadsmoke.",
                    }
                ]
            },
            name="/v1/documents",
        )
