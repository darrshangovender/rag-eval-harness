"""Metric tests.

`retrieval_recall` is pure arithmetic and gets exhaustive treatment. The two
judge-backed metrics are tested for the part the harness owns — prompt assembly
and score pass-through — with the model itself stubbed out.
"""
from __future__ import annotations

import pytest

from harness.metrics import Chunk, answer_relevance, faithfulness, retrieval_recall

# --------------------------------------------------------------------------- #
# retrieval_recall — no LLM, pure set arithmetic
# --------------------------------------------------------------------------- #


def test_no_gold_ids_is_trivially_satisfied():
    """Guards the `found / len(gold_ids)` division. An unlabelled question should
    not drag the recall average down."""
    assert retrieval_recall("q", "a", [Chunk("doc-1", "text")], []) == 1.0


def test_no_gold_ids_and_no_chunks_is_still_one():
    assert retrieval_recall("q", "a", [], []) == 1.0


def test_all_gold_chunks_retrieved_scores_one():
    chunks = [Chunk("doc-1", "a"), Chunk("doc-2", "b")]
    assert retrieval_recall("q", "a", chunks, ["doc-1", "doc-2"]) == 1.0


def test_partial_retrieval_scores_the_fraction_found():
    chunks = [Chunk("doc-1", "a"), Chunk("doc-9", "b")]
    assert retrieval_recall("q", "a", chunks, ["doc-1", "doc-2", "doc-3"]) == pytest.approx(1 / 3)


def test_nothing_relevant_retrieved_scores_zero():
    assert retrieval_recall("q", "a", [Chunk("doc-9", "x")], ["doc-1"]) == 0.0


def test_empty_retrieval_against_labelled_question_scores_zero():
    assert retrieval_recall("q", "a", [], ["doc-1"]) == 0.0


def test_extra_retrieved_chunks_do_not_penalise_recall():
    """This is recall, not precision — over-retrieval is not punished here."""
    chunks = [Chunk(f"doc-{i}", "x") for i in range(20)]
    assert retrieval_recall("q", "a", chunks, ["doc-3"]) == 1.0


def test_duplicate_gold_ids_are_counted_once_each():
    """Pinning the behaviour: `gold_ids` is treated as a list, not a set, so a
    duplicated label is counted twice in both numerator and denominator."""
    chunks = [Chunk("doc-1", "x")]
    assert retrieval_recall("q", "a", chunks, ["doc-1", "doc-1"]) == 1.0
    assert retrieval_recall("q", "a", chunks, ["doc-1", "doc-2"]) == pytest.approx(0.5)


def test_duplicate_retrieved_chunks_do_not_inflate_recall():
    chunks = [Chunk("doc-1", "x"), Chunk("doc-1", "x")]
    assert retrieval_recall("q", "a", chunks, ["doc-1", "doc-2"]) == pytest.approx(0.5)


def test_chunk_ids_match_exactly_and_are_case_sensitive():
    assert retrieval_recall("q", "a", [Chunk("DOC-1", "x")], ["doc-1"]) == 0.0


def test_unicode_chunk_ids_match():
    assert retrieval_recall("q", "a", [Chunk("café-☕", "x")], ["café-☕"]) == 1.0


# --------------------------------------------------------------------------- #
# faithfulness
# --------------------------------------------------------------------------- #


def test_faithfulness_returns_the_judge_score(mock_judge, chunks):
    mock_judge(0.42)
    assert faithfulness("q", "answer", chunks, []) == pytest.approx(0.42)


def test_faithfulness_joins_every_chunk_into_the_context(mock_judge, chunks):
    mock = mock_judge(1.0)
    faithfulness("Where are they?", "In Durban.", chunks, [])
    prompt = mock.last_prompt
    assert "Agulhas Code is based in Durban." in prompt
    assert "InsightEngine is a natural-language SQL layer." in prompt
    assert "\n\n---\n\n" in prompt, "chunk separator missing; the judge cannot tell them apart"
    assert "In Durban." in prompt


def test_faithfulness_with_no_chunks_tells_the_judge_so(mock_judge):
    """Without the placeholder the judge sees an empty CONTEXT block and tends to
    score generously instead of marking the answer ungrounded."""
    mock = mock_judge(0.0)
    faithfulness("q", "Some claim.", [], [])
    assert "(no chunks retrieved)" in mock.last_prompt


def test_faithfulness_treats_blank_chunk_text_as_no_context(mock_judge):
    """Edge case worth knowing: chunks whose text is empty join to `""`, which is
    falsy, so the placeholder kicks in even though chunks *were* retrieved."""
    mock = mock_judge(0.0)
    faithfulness("q", "a", [Chunk("doc-1", "")], [])
    assert "(no chunks retrieved)" in mock.last_prompt


def test_faithfulness_prompt_renders_the_json_example_literally(mock_judge, chunks):
    """The doubled braces in the template must survive `.format`, otherwise the
    judge is shown a broken schema."""
    mock = mock_judge(1.0)
    faithfulness("q", "a", chunks, [])
    assert '{"score": <float 0..1>, "reasoning": "<one short sentence>"}' in mock.last_prompt


def test_braces_in_the_answer_do_not_break_prompt_formatting(mock_judge, chunks):
    """Answers about JSON payloads are common in RAG over API docs."""
    mock = mock_judge(1.0)
    answer = 'The API returns {"status": "ok", "id": 1}.'
    faithfulness("q", answer, chunks, [])
    assert answer in mock.last_prompt


def test_gold_ids_are_ignored_by_faithfulness(mock_judge, chunks):
    """All metrics share one signature; faithfulness has no use for the labels."""
    mock = mock_judge(1.0)
    faithfulness("q", "a", chunks, ["doc-1"])
    assert "doc-1" not in mock.last_prompt


# --------------------------------------------------------------------------- #
# answer_relevance
# --------------------------------------------------------------------------- #


def test_answer_relevance_returns_the_judge_score(mock_judge, chunks):
    mock_judge(0.9)
    assert answer_relevance("q", "a", chunks, []) == pytest.approx(0.9)


def test_answer_relevance_prompt_carries_question_and_answer(mock_judge, chunks):
    mock = mock_judge(1.0)
    answer_relevance("Where is it?", "Durban.", chunks, [])
    assert "QUESTION: Where is it?" in mock.last_prompt
    assert "ANSWER: Durban." in mock.last_prompt


def test_answer_relevance_does_not_show_the_judge_the_context(mock_judge, chunks):
    """Relevance is about the question/answer pair only — leaking the context would
    let the judge conflate relevance with faithfulness."""
    mock = mock_judge(1.0)
    answer_relevance("q", "a", chunks, [])
    assert "Agulhas Code is based in Durban." not in mock.last_prompt


def test_answer_relevance_instructs_the_judge_to_reward_honest_refusals(mock_judge, chunks):
    mock = mock_judge(1.0)
    answer_relevance("q", "I don't know.", chunks, [])
    assert "0.7" in mock.last_prompt


def test_unicode_question_and_answer_reach_the_judge_intact(mock_judge, chunks):
    mock = mock_judge(1.0)
    answer_relevance("Où est le café ?", "À Durban ☕", chunks, [])
    assert "Où est le café ?" in mock.last_prompt
    assert "À Durban ☕" in mock.last_prompt
