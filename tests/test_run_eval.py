"""Runner tests: pipeline loading, aggregation, percentiles, and the exit code.

`main()` is driven end-to-end here with a fake pipeline module and stubbed
metrics, so the orchestration is covered without a single API call.
"""
from __future__ import annotations

import sys
import types

import pytest
import yaml

from harness import run_eval
from harness.run_eval import _load_pipeline, main


@pytest.fixture
def fake_pipeline(monkeypatch):
    """Register an importable module implementing the retrieve/generate contract."""

    def _install(name: str = "fake_pipeline", *, retrieve=None, generate=None):
        module = types.ModuleType(name)
        if retrieve is not None:
            module.retrieve = retrieve
        if generate is not None:
            module.generate = generate
        monkeypatch.setitem(sys.modules, name, module)
        return module

    return _install


@pytest.fixture
def questions_file(tmp_path):
    def _write(entries: list[dict]) -> str:
        path = tmp_path / "questions.yml"
        path.write_text(yaml.safe_dump(entries), encoding="utf-8")
        return str(path)

    return _write


@pytest.fixture
def stub_metrics(monkeypatch):
    """Replace the three metrics with constants so the runner's own maths is visible."""

    def _install(faith: float = 1.0, relevance: float = 1.0, recall: float = 1.0):
        monkeypatch.setattr(run_eval, "faithfulness", lambda *a: faith)
        monkeypatch.setattr(run_eval, "answer_relevance", lambda *a: relevance)
        monkeypatch.setattr(run_eval, "retrieval_recall", lambda *a: recall)

    return _install


def _default_pipeline(fake_pipeline):
    return fake_pipeline(
        retrieve=lambda q: [{"id": "doc-1", "text": "context"}],
        generate=lambda q, c: "an answer",
    )


def _run(monkeypatch, questions_path: str, module: str = "fake_pipeline") -> int:
    monkeypatch.setattr(
        sys, "argv", ["run_eval", "--questions", questions_path, "--pipeline", module]
    )
    return main()


# --------------------------------------------------------------------------- #
# _load_pipeline
# --------------------------------------------------------------------------- #


def test_loads_a_module_implementing_the_contract(fake_pipeline):
    _default_pipeline(fake_pipeline)
    assert _load_pipeline("fake_pipeline").retrieve("q") == [{"id": "doc-1", "text": "context"}]


@pytest.mark.parametrize("missing", ["retrieve", "generate"])
def test_a_pipeline_missing_half_the_contract_is_rejected(fake_pipeline, missing):
    kwargs = {"retrieve": lambda q: [], "generate": lambda q, c: ""}
    kwargs.pop(missing)
    fake_pipeline("partial_pipeline", **kwargs)
    with pytest.raises(SystemExit, match="must expose retrieve"):
        _load_pipeline("partial_pipeline")


def test_the_shipped_example_pipeline_satisfies_the_contract():
    module = _load_pipeline("examples.baseline_rag")
    assert callable(module.retrieve) and callable(module.generate)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #


def test_a_passing_run_exits_zero_and_saves_a_report(
    monkeypatch, fake_pipeline, questions_file, stub_metrics, runs_dir
):
    _default_pipeline(fake_pipeline)
    stub_metrics()
    path = questions_file([{"question": "q1", "gold_chunk_ids": ["doc-1"]}])

    assert _run(monkeypatch, path) == 0
    saved = list(runs_dir.glob("*.json"))
    assert len(saved) == 1


def test_a_failing_run_exits_nonzero(
    monkeypatch, fake_pipeline, questions_file, stub_metrics, runs_dir
):
    """The exit code is what turns a metric drop into a red CI check."""
    _default_pipeline(fake_pipeline)
    stub_metrics(faith=0.1)
    path = questions_file([{"question": "q1", "gold_chunk_ids": ["doc-1"]}])
    assert _run(monkeypatch, path) == 1


def test_scores_are_averaged_across_questions(
    monkeypatch, fake_pipeline, questions_file, runs_dir
):
    _default_pipeline(fake_pipeline)
    recalls = iter([1.0, 0.0, 1.0, 1.0])
    monkeypatch.setattr(run_eval, "faithfulness", lambda *a: 1.0)
    monkeypatch.setattr(run_eval, "answer_relevance", lambda *a: 1.0)
    monkeypatch.setattr(run_eval, "retrieval_recall", lambda *a: next(recalls))

    path = questions_file([{"question": f"q{i}", "gold_chunk_ids": ["doc-1"]} for i in range(4)])
    _run(monkeypatch, path)

    report = run_eval.load_latest_prior("never-matches")
    assert report.retrieval_recall == pytest.approx(0.75)
    assert report.n_questions == 4


def test_questions_without_gold_ids_default_to_an_empty_label_set(
    monkeypatch, fake_pipeline, questions_file, runs_dir
):
    """`gold_chunk_ids` is optional in the YAML; a question lacking it must not
    KeyError the whole run."""
    _default_pipeline(fake_pipeline)
    captured: list[list[str]] = []
    monkeypatch.setattr(run_eval, "faithfulness", lambda *a: 1.0)
    monkeypatch.setattr(run_eval, "answer_relevance", lambda *a: 1.0)
    monkeypatch.setattr(
        run_eval, "retrieval_recall", lambda q, a, c, gold: captured.append(gold) or 1.0
    )

    _run(monkeypatch, questions_file([{"question": "unlabelled"}]))
    assert captured == [[]]


def test_the_pipeline_receives_the_question_and_its_own_chunks(
    monkeypatch, fake_pipeline, questions_file, stub_metrics, runs_dir
):
    seen: dict = {}

    def retrieve(q):
        seen["question"] = q
        return [{"id": "doc-1", "text": "context"}]

    def generate(q, chunks):
        seen["chunks"] = chunks
        return "an answer"

    fake_pipeline(retrieve=retrieve, generate=generate)
    stub_metrics()
    _run(monkeypatch, questions_file([{"question": "Where is it?"}]))

    assert seen["question"] == "Where is it?"
    assert seen["chunks"] == [{"id": "doc-1", "text": "context"}]


def test_report_records_the_pipeline_module_that_produced_it(
    monkeypatch, fake_pipeline, questions_file, stub_metrics, runs_dir
):
    _default_pipeline(fake_pipeline)
    stub_metrics()
    _run(monkeypatch, questions_file([{"question": "q"}]))
    assert run_eval.load_latest_prior("never").pipeline == "fake_pipeline"


@pytest.mark.parametrize("n_questions", [1, 5, 19, 20, 40])
def test_percentile_selection_across_sample_sizes(
    monkeypatch, fake_pipeline, questions_file, stub_metrics, runs_dir, n_questions
):
    """Below 20 samples a real p95 is meaningless, so the runner reports the max
    instead. This pins the switchover so the number is never silently reinterpreted."""
    _default_pipeline(fake_pipeline)
    stub_metrics()
    _run(monkeypatch, questions_file([{"question": f"q{i}"} for i in range(n_questions)]))

    report = run_eval.load_latest_prior("never")
    assert report.latency_p50 <= report.latency_p95
    assert report.latency_p95 >= 0.0
    assert report.n_questions == n_questions


def test_an_empty_question_set_fails_loudly_rather_than_reporting_zeroes(
    monkeypatch, fake_pipeline, questions_file, stub_metrics, runs_dir
):
    """Known rough edge: there is no guard for an empty YAML, so the percentile
    lookup raises IndexError. Pinned so it stays a crash rather than silently
    becoming a 0.0-across-the-board 'passing' run."""
    _default_pipeline(fake_pipeline)
    stub_metrics()
    with pytest.raises(IndexError):
        _run(monkeypatch, questions_file([]))


def test_the_shipped_golden_set_runs_end_to_end_offline(monkeypatch, stub_metrics, runs_dir):
    """Integration: the real golden.yml through the real baseline pipeline, with only
    the two LLM-backed metrics stubbed. Retrieval recall is left real."""
    monkeypatch.setattr(run_eval, "faithfulness", lambda *a: 1.0)
    monkeypatch.setattr(run_eval, "answer_relevance", lambda *a: 1.0)

    assert _run(monkeypatch, "data/golden.yml", module="examples.baseline_rag") == 0
    report = run_eval.load_latest_prior("never")
    assert report.retrieval_recall == 1.0
    assert report.n_questions == 4
