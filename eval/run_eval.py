"""CLI wrapper so `python eval/run_eval.py` works from a checkout."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rag_service.evalrunner import main  # noqa: E402

if __name__ == "__main__":
    corpus = ROOT / "eval" / "corpus.jsonl"
    questions = ROOT / "eval" / "questions.jsonl"
    args = sys.argv[1:]
    if "--corpus" not in args:
        args = ["--corpus", str(corpus), "--questions", str(questions), *args]
    raise SystemExit(main(args))
