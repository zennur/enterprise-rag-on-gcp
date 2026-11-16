"""Token and vector helpers for the local embedding provider and the eval harness."""

from __future__ import annotations

import hashlib
import math
import re

_TOKEN = re.compile(r"[a-z0-9]+")

# Function words are dropped so the local hash is driven by content tokens.
# This is not a claim about how Vertex embeddings behave.
STOPWORDS = frozenset(
    {
        "a",
        "about",
        "after",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "before",
        "by",
        "can",
        "did",
        "do",
        "does",
        "during",
        "for",
        "from",
        "how",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "many",
        "much",
        "of",
        "on",
        "or",
        "per",
        "should",
        "than",
        "that",
        "the",
        "their",
        "then",
        "they",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "who",
        "why",
        "will",
        "with",
        "within",
    }
)


def content_tokens(text: str) -> list[str]:
    return [
        token for token in _TOKEN.findall(text.lower()) if token not in STOPWORDS and len(token) > 1
    ]


def features(text: str) -> list[str]:
    words = content_tokens(text)
    bigrams = [f"{left}_{right}" for left, right in zip(words, words[1:], strict=False)]
    return words + bigrams


def hashing_embed(text: str, dimensions: int) -> list[float]:
    """Signed feature hashing. Stable across processes, unlike Python's hash()."""
    if dimensions <= 0:
        raise ValueError("dimensions must be positive")
    vector = [0.0] * dimensions
    for feature in features(text):
        digest = hashlib.sha256(feature.encode()).digest()
        bucket = int.from_bytes(digest[:4], "little") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    return l2_normalize(vector)


def l2_normalize(vector: list[float]) -> list[float]:
    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0.0:
        return vector
    return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError("vectors have different dimensions")
    dot = 0.0
    left_norm = 0.0
    right_norm = 0.0
    for x_value, y_value in zip(left, right, strict=True):
        dot += x_value * y_value
        left_norm += x_value * x_value
        right_norm += y_value * y_value
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / math.sqrt(left_norm * right_norm)


def distance_to_score(distance: float, mode: str) -> float:
    """Convert a Vector Search distance into a higher-is-better score.

    Cosine indexes return a distance (0 is identical). Dot-product indexes on
    normalized vectors return a similarity. The deployed index has to match
    ``mode``.
    """
    if mode == "dot":
        return distance
    if mode == "cosine":
        return 1.0 - distance
    raise ValueError(f"unsupported distance mode: {mode}")
