"""Shared fixtures.

The harness exists to call LLM judges, so the single most important property of
this suite is that it never does. `MockJudge` replaces `harness.judge.judge` at
the seam every metric goes through, and the provider-specific client tests inject
fake SDK clients rather than letting `anthropic` / `openai` open a socket.
"""
from __future__ import annotations

import pytest

from harness.judge import JudgeResult
from harness.metrics import Chunk


class MockJudge:
    """Deterministic stand-in for `judge()`.

    Returns `scores` in order (repeating the last one once exhausted) and records
    every prompt it was given, so tests can assert on prompt construction without
    a network call.
    """

    def __init__(self, *scores: float, reasoning: str = "stubbed"):
        self.scores = list(scores) or [1.0]
        self.reasoning = reasoning
        self.prompts: list[str] = []

    def __call__(self, prompt: str, *, system: str = "") -> JudgeResult:
        self.prompts.append(prompt)
        score = self.scores[min(len(self.prompts) - 1, len(self.scores) - 1)]
        return JudgeResult(score=score, reasoning=self.reasoning, raw="{}")

    @property
    def last_prompt(self) -> str:
        return self.prompts[-1]


@pytest.fixture
def mock_judge(monkeypatch):
    """Install a MockJudge into `harness.metrics`.

    `metrics.py` does `from .judge import judge`, so the name lives in the metrics
    module — patching `harness.judge.judge` would not be seen by the metrics.
    """

    def _install(*scores: float) -> MockJudge:
        mock = MockJudge(*scores)
        monkeypatch.setattr("harness.metrics.judge", mock)
        return mock

    return _install


@pytest.fixture(autouse=True)
def no_ambient_api_keys(monkeypatch):
    """A developer's exported keys must never turn a unit test into a billed call."""
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("JUDGE_MODEL", raising=False)


@pytest.fixture
def chunks() -> list[Chunk]:
    return [
        Chunk(id="doc-1", text="Agulhas Code is based in Durban."),
        Chunk(id="doc-2", text="InsightEngine is a natural-language SQL layer."),
    ]


@pytest.fixture
def runs_dir(tmp_path, monkeypatch):
    """Redirect the module-level `runs/` directory into tmp_path.

    `REPORT_DIR` is a module global dereferenced inside `save_run` /
    `load_latest_prior`, so patching the attribute is enough — no chdir needed.
    """
    target = tmp_path / "runs"
    monkeypatch.setattr("harness.report.REPORT_DIR", target)
    return target
