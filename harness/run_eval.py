"""Entry point: load questions + pipeline, score each, render report.

Usage:
    uv run python harness/run_eval.py \\
        --questions data/golden.yml \\
        --pipeline examples.baseline_rag
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml

from .metrics import Chunk, answer_relevance, faithfulness, retrieval_recall
from .report import RunReport, load_latest_prior, print_report, save_run


def _load_pipeline(module_path: str):
    """Import a pipeline module that exposes `retrieve()` and `generate()`."""
    module = importlib.import_module(module_path)
    if not hasattr(module, "retrieve") or not hasattr(module, "generate"):
        raise SystemExit(f"{module_path} must expose retrieve(question) and generate(question, chunks)")
    return module


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True, help="Path to a YAML labelled question set")
    parser.add_argument("--pipeline", required=True, help="Python module path, e.g. examples.baseline_rag")
    args = parser.parse_args()

    questions = yaml.safe_load(Path(args.questions).read_text())
    pipeline = _load_pipeline(args.pipeline)

    latencies: list[float] = []
    faith_scores: list[float] = []
    rel_scores: list[float] = []
    recall_scores: list[float] = []

    for q in questions:
        question = q["question"]
        gold_ids = q.get("gold_chunk_ids", [])
        start = time.time()
        chunks_raw = pipeline.retrieve(question)
        answer = pipeline.generate(question, chunks_raw)
        latencies.append(time.time() - start)

        chunks = [Chunk(id=c["id"], text=c["text"]) for c in chunks_raw]
        faith_scores.append(faithfulness(question, answer, chunks, gold_ids))
        rel_scores.append(answer_relevance(question, answer, chunks, gold_ids))
        recall_scores.append(retrieval_recall(question, answer, chunks, gold_ids))

    latencies.sort()
    p50 = latencies[len(latencies) // 2]
    p95 = latencies[int(len(latencies) * 0.95)] if len(latencies) >= 20 else latencies[-1]

    report = RunReport(
        timestamp=datetime.utcnow().strftime("%Y%m%d-%H%M%S"),
        pipeline=args.pipeline,
        n_questions=len(questions),
        faithfulness=sum(faith_scores) / len(faith_scores),
        answer_relevance=sum(rel_scores) / len(rel_scores),
        retrieval_recall=sum(recall_scores) / len(recall_scores),
        latency_p50=p50,
        latency_p95=p95,
        total_cost_usd=0.0,  # TODO: token accounting if you want it
    )
    prior = load_latest_prior(report.timestamp)
    ok = print_report(report, prior)
    save_run(report)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
