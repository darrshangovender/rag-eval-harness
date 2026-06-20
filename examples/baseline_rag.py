"""Toy RAG pipeline for the harness's smoke test.

Real pipelines plug in by implementing the same two functions.
This one uses a hard-coded corpus and BM25-like keyword match — just enough
to demonstrate the contract with the harness without any external services.
"""

from __future__ import annotations

# A tiny corpus that the smoke-test questions in data/golden.yml ask about.
CORPUS = [
    {
        "id": "doc-1",
        "text": (
            "Agulhas Code is an engineering studio based in Durban, South Africa. "
            "It was founded in 2024 and specialises in AI-powered web platforms for SMEs."
        ),
    },
    {
        "id": "doc-2",
        "text": (
            "InsightEngine is a natural-language SQL analytics layer over multi-table "
            "warehouses. It uses an LLM query-planner with sqlglot AST guardrails "
            "against destructive operations."
        ),
    },
    {
        "id": "doc-3",
        "text": (
            "Postgres with pgvector is the recommended storage backend for embeddings "
            "in small to mid-sized RAG systems. It avoids the operational cost of a "
            "separate vector database while supporting transactional writes."
        ),
    },
]


def retrieve(question: str) -> list[dict]:
    """Return the two most keyword-overlapping chunks. Deliberately simple."""
    q_words = {w.lower().strip(".,?!") for w in question.split() if len(w) > 3}
    scored = []
    for chunk in CORPUS:
        chunk_words = {w.lower().strip(".,?!") for w in chunk["text"].split()}
        overlap = len(q_words & chunk_words)
        scored.append((overlap, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:2]]


def generate(question: str, chunks: list[dict]) -> str:
    """A zero-LLM baseline: concatenate the retrieved chunks as the 'answer'.

    Real pipelines would call an LLM here. This stub lets the harness run
    end-to-end without API credits during initial setup / CI smoke tests.
    """
    if not chunks:
        return "I don't have enough context to answer that."
    summaries = " ".join(c["text"] for c in chunks)
    return f"Based on the available context: {summaries}"
