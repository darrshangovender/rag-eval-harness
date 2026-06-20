"""Three metrics: faithfulness, answer relevance, retrieval recall@k.

Each metric is one function taking (question, answer, retrieved_chunks,
gold_chunk_ids) and returning a 0..1 score. The harness composes them.

Design choice: prompts live next to the metric they serve. This makes it
trivial to A/B a metric prompt — change one file, re-run the harness, see
the diff. Keeping prompts split across a /prompts/ folder makes the
indirection cost outweigh the organisation benefit at this scale.
"""

from __future__ import annotations

from dataclasses import dataclass

from .judge import judge


@dataclass
class Chunk:
    id: str
    text: str


# ---------- Faithfulness ----------

_FAITHFULNESS_PROMPT = """\
You are evaluating whether an ANSWER is faithful to the supplied CONTEXT.

An answer is faithful if and only if every factual claim it makes is supported
by something in the context. Stylistic / connective text is fine. Numbers,
names, dates, and specific facts must be grounded.

CONTEXT:
{context}

ANSWER:
{answer}

Respond with compact JSON only:
{{"score": <float 0..1>, "reasoning": "<one short sentence>"}}

Where score is the fraction of factual claims that are supported.
1.0 = every claim grounded.  0.0 = nothing grounded / hallucinated.
"""


def faithfulness(question: str, answer: str, chunks: list[Chunk], gold_ids: list[str]) -> float:
    """Score how grounded the answer is in retrieved chunks."""
    context = "\n\n---\n\n".join(c.text for c in chunks) or "(no chunks retrieved)"
    prompt = _FAITHFULNESS_PROMPT.format(context=context, answer=answer)
    return judge(prompt).score


# ---------- Answer relevance ----------

_RELEVANCE_PROMPT = """\
You are evaluating whether an ANSWER addresses a QUESTION.

A relevant answer directly addresses what was asked. Evasive,
off-topic, or "I don't know" responses score low.
Refusing to answer because the context lacks the info IS acceptable
behaviour and should score ~0.7 (it's relevant and honest).

QUESTION: {question}
ANSWER: {answer}

Respond with compact JSON only:
{{"score": <float 0..1>, "reasoning": "<one short sentence>"}}
"""


def answer_relevance(question: str, answer: str, chunks: list[Chunk], gold_ids: list[str]) -> float:
    prompt = _RELEVANCE_PROMPT.format(question=question, answer=answer)
    return judge(prompt).score


# ---------- Retrieval recall@k ----------

def retrieval_recall(question: str, answer: str, chunks: list[Chunk], gold_ids: list[str]) -> float:
    """Fraction of gold chunks that appear in the retrieved set.

    No LLM call — pure exact match. Cheap and deterministic.
    """
    if not gold_ids:
        return 1.0  # nothing to find, trivially satisfied
    retrieved_ids = {c.id for c in chunks}
    found = sum(1 for gid in gold_ids if gid in retrieved_ids)
    return found / len(gold_ids)
