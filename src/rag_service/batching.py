"""Small batch helper used by embedding calls."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

T = TypeVar("T")


def batched(items: Sequence[T], size: int) -> list[list[T]]:
    if size < 1:
        raise ValueError("batch size must be positive")
    return [list(items[index : index + size]) for index in range(0, len(items), size)]
