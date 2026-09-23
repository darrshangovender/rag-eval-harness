"""Reference-pipeline tests.

`examples/baseline_rag.py` is the contract every real pipeline is written
against, and `data/golden.yml` is the labelled set shipped with it. The last two
tests here check that the two files still agree with each other — that is the
thing most likely to rot when someone edits the corpus.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from examples.baseline_rag import CORPUS, generate, retrieve
from harness.metrics import Chunk, retrieval_recall

GOLDEN_PATH = Path(__file__).resolve().parents[1] / "data" / "golden.yml"


@pytest.fixture(scope="module")
def golden() -> list[dict]:
    return yaml.safe_load(GOLDEN_PATH.read_text(encoding="utf-8"))


# --------------------------------------------------------------------------- #
# retrieve
# --------------------------------------------------------------------------- #


def test_retrieve_returns_the_top_two_chunks():
    assert len(retrieve("Where is Agulhas Code based?")) == 2


def test_retrieve_returns_chunks_in_the_documented_dict_shape():
    """`run_eval` does `Chunk(id=c["id"], text=c["text"])`, so these keys are load-bearing."""
    for chunk in retrieve("anything"):
        assert set(chunk) >= {"id", "text"}


def test_retrieve_ranks_the_keyword_overlapping_chunk_first():
    assert retrieve("Where is Agulhas Code based?")[0]["id"] == "doc-1"


def test_retrieve_matches_on_pgvector_terminology():
    assert retrieve("Why would you use Postgres with pgvector for RAG?")[0]["id"] == "doc-3"


def test_short_words_are_dropped_before_matching():
    """The `len(w) > 3` filter is a crude stopword proxy; `the`/`was`/`is` must not
    decide the ranking."""
    assert retrieve("was the is")[0]["id"] == CORPUS[0]["id"], "expected the stable input order"


def test_length_filter_runs_before_punctuation_is_stripped():
    """`RAG?` is 4 characters so it survives the filter and *then* becomes `rag` —
    a 3-character token that would have been dropped if the order were reversed."""
    assert retrieve("pgvector RAG?")[0]["id"] == "doc-3"


def test_trailing_punctuation_does_not_block_a_match():
    assert retrieve("What does InsightEngine do?")[0]["id"] == "doc-2"


def test_matching_is_case_insensitive():
    assert retrieve("INSIGHTENGINE")[0]["id"] == retrieve("insightengine")[0]["id"] == "doc-2"


def test_a_question_with_no_overlap_still_returns_two_chunks():
    """Zero-overlap must degrade to an arbitrary-but-stable pair rather than an
    empty list — the harness would otherwise score an empty context."""
    results = retrieve("zzz qqq xxx")
    assert len(results) == 2
    assert results == retrieve("zzz qqq xxx"), "retrieval must be deterministic"


def test_empty_question_does_not_crash():
    assert len(retrieve("")) == 2


# --------------------------------------------------------------------------- #
# generate
# --------------------------------------------------------------------------- #


def test_generate_concatenates_every_retrieved_chunk():
    chunks = [{"id": "a", "text": "First."}, {"id": "b", "text": "Second."}]
    answer = generate("q", chunks)
    assert answer.startswith("Based on the available context:")
    assert "First." in answer and "Second." in answer


def test_generate_refuses_when_nothing_was_retrieved():
    assert generate("q", []) == "I don't have enough context to answer that."


# --------------------------------------------------------------------------- #
# The shipped golden set must stay consistent with the shipped corpus
# --------------------------------------------------------------------------- #


def test_every_gold_chunk_id_exists_in_the_corpus(golden):
    corpus_ids = {c["id"] for c in CORPUS}
    for question in golden:
        assert set(question["gold_chunk_ids"]) <= corpus_ids, question["question"]


def test_golden_entries_have_the_fields_run_eval_reads(golden):
    assert golden, "golden.yml must not be empty"
    for question in golden:
        assert isinstance(question["question"], str) and question["question"]
        assert isinstance(question.get("gold_chunk_ids", []), list)


def test_the_baseline_pipeline_scores_perfect_recall_on_its_own_golden_set(golden):
    """The shipped demo is what a new user runs first. If it cannot retrieve its own
    labelled answers, the harness looks broken when it is the fixture that rotted."""
    for question in golden:
        chunks = [Chunk(id=c["id"], text=c["text"]) for c in retrieve(question["question"])]
        recall = retrieval_recall(
            question["question"], "", chunks, question.get("gold_chunk_ids", [])
        )
        assert recall == 1.0, f"{question['question']!r} lost its gold chunk"
