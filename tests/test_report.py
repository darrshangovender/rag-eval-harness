"""Report tests: the pass/fail logic that gates CI.

`print_report` returns the boolean that becomes the process exit code, so its
two independent failure conditions — missing an absolute target, and regressing
against the previous run — are tested separately and at their boundaries.
"""
from __future__ import annotations

import json

import pytest

from harness.report import (
    REGRESSION_THRESHOLD,
    TARGETS,
    RunReport,
    load_latest_prior,
    print_report,
    save_run,
)


def make_report(
    *,
    timestamp: str = "20240101-120000",
    faithfulness: float = 0.95,
    answer_relevance: float = 0.90,
    retrieval_recall: float = 0.85,
) -> RunReport:
    """A run that clears every target, so tests can lower one field at a time."""
    return RunReport(
        timestamp=timestamp,
        pipeline="examples.baseline_rag",
        n_questions=4,
        faithfulness=faithfulness,
        answer_relevance=answer_relevance,
        retrieval_recall=retrieval_recall,
        latency_p50=0.1,
        latency_p95=0.2,
        total_cost_usd=0.0,
    )


# --------------------------------------------------------------------------- #
# Target checks
# --------------------------------------------------------------------------- #


def test_a_clean_run_with_no_history_passes():
    assert print_report(make_report(), None) is True


@pytest.mark.parametrize("metric", ["faithfulness", "answer_relevance", "retrieval_recall"])
def test_any_metric_below_its_target_fails_the_run(metric):
    assert print_report(make_report(**{metric: TARGETS[metric] - 0.01}), None) is False


@pytest.mark.parametrize("metric", ["faithfulness", "answer_relevance", "retrieval_recall"])
def test_a_metric_exactly_on_target_passes(metric):
    """The comparison is `>=`; landing exactly on the target must not fail CI."""
    assert print_report(make_report(**{metric: TARGETS[metric]}), None) is True


def test_all_three_metrics_are_evaluated_not_just_the_first():
    assert print_report(make_report(retrieval_recall=0.0), None) is False


# --------------------------------------------------------------------------- #
# Regression checks
# --------------------------------------------------------------------------- #


def test_regression_beyond_the_threshold_fails_even_when_the_target_is_met():
    """The whole point of the harness: 0.93 still clears the 0.90 faithfulness bar,
    but dropping 0.07 in one PR is the signal engineers actually need."""
    prior = make_report(faithfulness=1.0)
    current = make_report(faithfulness=1.0 - REGRESSION_THRESHOLD - 0.02)
    assert print_report(current, prior) is False


def test_a_drop_just_under_the_threshold_is_tolerated():
    """Normal run-to-run judge jitter must not fail the build.

    Deliberately tested a hair inside the boundary rather than exactly on it: the
    comparison is `delta >= -REGRESSION_THRESHOLD` on raw floats, and a nominal
    0.05 drop can land at -0.050000000000000044 depending on the operands. Any
    test asserting the exact boundary would be pinning float representation, not
    behaviour.
    """
    prior = make_report(faithfulness=1.0)
    current = make_report(faithfulness=1.0 - REGRESSION_THRESHOLD + 0.01)
    assert print_report(current, prior) is True


def test_a_drop_just_over_the_threshold_fails():
    prior = make_report(faithfulness=1.0)
    current = make_report(faithfulness=1.0 - REGRESSION_THRESHOLD - 0.01)
    assert print_report(current, prior) is False


def test_improving_against_the_prior_run_passes():
    assert print_report(make_report(faithfulness=0.99), make_report(faithfulness=0.91)) is True


def test_a_run_can_fail_on_target_and_regression_at_once():
    prior = make_report(retrieval_recall=0.95)
    assert print_report(make_report(retrieval_recall=0.50), prior) is False


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def test_as_dict_returns_a_copy_that_cannot_mutate_the_report():
    report = make_report()
    d = report.as_dict()
    d["faithfulness"] = 0.0
    assert report.faithfulness == pytest.approx(0.95)


def test_save_run_writes_a_json_file_named_for_the_timestamp(runs_dir):
    path = save_run(make_report(timestamp="20240501-101010"))
    assert path == runs_dir / "20240501-101010.json"
    assert json.loads(path.read_text())["pipeline"] == "examples.baseline_rag"


def test_saved_run_round_trips_back_into_a_runreport(runs_dir):
    original = make_report(timestamp="20240501-101010")
    save_run(original)
    assert load_latest_prior("20990101-000000") == original


def test_save_run_creates_the_report_directory(runs_dir):
    assert not runs_dir.exists()
    save_run(make_report())
    assert runs_dir.is_dir()


def test_no_prior_runs_returns_none(runs_dir):
    assert load_latest_prior("20240101-120000") is None


def test_the_current_run_is_excluded_from_its_own_comparison(runs_dir):
    """Otherwise every run would be compared against itself and never regress."""
    save_run(make_report(timestamp="20240101-120000"))
    assert load_latest_prior("20240101-120000") is None


def test_the_most_recent_prior_run_wins(runs_dir):
    """Timestamps are `YYYYmmdd-HHMMSS`, so lexicographic order is chronological."""
    save_run(make_report(timestamp="20240101-090000", faithfulness=0.91))
    save_run(make_report(timestamp="20240301-090000", faithfulness=0.99))
    save_run(make_report(timestamp="20240201-090000", faithfulness=0.95))
    prior = load_latest_prior("20240401-090000")
    assert prior.timestamp == "20240301-090000"
    assert prior.faithfulness == pytest.approx(0.99)


def test_non_json_files_in_the_runs_directory_are_ignored(runs_dir):
    runs_dir.mkdir(parents=True)
    (runs_dir / "notes.txt").write_text("scratch")
    save_run(make_report(timestamp="20240101-090000"))
    assert load_latest_prior("20240401-090000").timestamp == "20240101-090000"
