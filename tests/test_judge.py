"""Judge tests: response parsing, score clamping, and provider selection.

`_parse` is the only place raw model output becomes a number, so a bug here
silently corrupts every metric in the report. No test in this file makes a
network call — the SDK clients are replaced with fakes.
"""
from __future__ import annotations

import pytest
from tenacity import RetryError, stop_after_attempt, wait_none

from harness import judge as judge_module
from harness.judge import JudgeResult, _parse, judge


@pytest.fixture
def judge_once():
    """`judge` is wrapped in a 3-attempt exponential-backoff retry, which would add
    seconds of real sleeping to a unit test. Rebuild it with the retry disabled."""
    return judge.retry_with(stop=stop_after_attempt(1), wait=wait_none(), reraise=True)


# --------------------------------------------------------------------------- #
# _parse
# --------------------------------------------------------------------------- #


def test_parses_strict_json():
    result = _parse('{"score": 0.75, "reasoning": "mostly grounded"}')
    assert result.score == pytest.approx(0.75)
    assert result.reasoning == "mostly grounded"


def test_extracts_json_from_a_chatty_model():
    """Judges routinely wrap JSON in prose or a markdown fence despite instructions."""
    raw = 'Sure! Here is my assessment:\n```json\n{"score": 0.5, "reasoning": "partial"}\n```\nHope that helps.'
    result = _parse(raw)
    assert result.score == pytest.approx(0.5)
    assert result.reasoning == "partial"


@pytest.mark.parametrize(
    ("raw_score", "expected"),
    [
        ("1.5", 1.0),  # clamped down
        ("-0.2", 0.0),  # clamped up
        ("100", 1.0),  # judge answered on a 0-100 scale
        ("0", 0.0),
        ("1", 1.0),
        ('"0.8"', 0.8),  # stringified number still coerces
    ],
)
def test_score_is_clamped_into_the_unit_interval(raw_score, expected):
    """A judge that ignores the 0..1 instruction must not skew the run average."""
    assert _parse(f'{{"score": {raw_score}}}').score == pytest.approx(expected)


def test_missing_score_defaults_to_zero():
    """Fail closed: an unparseable verdict counts against the pipeline, not for it."""
    assert _parse('{"reasoning": "forgot the score"}').score == 0.0


def test_missing_reasoning_becomes_an_empty_string():
    assert _parse('{"score": 0.9}').reasoning == ""


def test_non_string_reasoning_is_coerced_and_stripped():
    assert _parse('{"score": 0.9, "reasoning": "  padded  "}').reasoning == "padded"
    assert _parse('{"score": 0.9, "reasoning": 42}').reasoning == "42"


def test_raw_response_is_retained_for_debugging():
    raw = '{"score": 0.4, "reasoning": "x"}'
    assert _parse(raw).raw == raw


def test_response_with_no_json_object_raises():
    with pytest.raises(ValueError, match="not JSON-parseable"):
        _parse("I refuse to answer.")


def test_empty_response_raises():
    """An OpenAI refusal arrives as `content=None`, which the caller turns into ''."""
    with pytest.raises(ValueError, match="not JSON-parseable"):
        _parse("")


@pytest.mark.parametrize(
    "raw",
    [
        "{score: not valid json at all",  # opening brace, never closed
        "{score: not valid json at all}",  # balanced but not JSON
    ],
)
def test_malformed_json_raises_rather_than_scoring_zero(raw):
    """`json.JSONDecodeError` subclasses `ValueError`, so both the unbalanced and the
    balanced-but-invalid cases surface the same way to the caller."""
    with pytest.raises(ValueError):
        _parse(raw)


# --------------------------------------------------------------------------- #
# Provider selection
# --------------------------------------------------------------------------- #


def test_no_api_key_is_a_clear_configuration_error(judge_once):
    with pytest.raises(RuntimeError, match="No judge API key set"):
        judge_once("prompt")


def test_configuration_errors_are_retried_before_surfacing(monkeypatch):
    """Pinning current behaviour: the `@retry` decorator wraps the whole function,
    so a missing key is retried 3 times and surfaces as `RetryError` rather than the
    underlying `RuntimeError`. Worth knowing when reading a CI traceback."""
    fast = judge.retry_with(stop=stop_after_attempt(3), wait=wait_none())
    with pytest.raises(RetryError) as excinfo:
        fast("prompt")
    assert isinstance(excinfo.value.last_attempt.exception(), RuntimeError)


def test_anthropic_is_preferred_when_both_keys_are_present(monkeypatch, judge_once):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai-test")
    called: list[str] = []
    monkeypatch.setattr(
        judge_module,
        "_judge_anthropic",
        lambda p, *, system: called.append("anthropic") or JudgeResult(1.0, "", ""),
    )
    monkeypatch.setattr(
        judge_module,
        "_judge_openai",
        lambda p, *, system: called.append("openai") or JudgeResult(1.0, "", ""),
    )
    judge_once("prompt")
    assert called == ["anthropic"]


def test_openai_is_used_when_it_is_the_only_key(monkeypatch, judge_once):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-oai-test")
    called: list[str] = []
    monkeypatch.setattr(
        judge_module,
        "_judge_openai",
        lambda p, *, system: called.append("openai") or JudgeResult(1.0, "", ""),
    )
    judge_once("prompt")
    assert called == ["openai"]


# --------------------------------------------------------------------------- #
# Client response unwrapping — fake SDKs, no sockets
# --------------------------------------------------------------------------- #


def test_anthropic_response_is_unwrapped_from_the_content_block(monkeypatch):
    import anthropic

    captured: dict = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            block = type("Block", (), {"text": '{"score": 0.6, "reasoning": "ok"}'})()
            return type("Resp", (), {"content": [block]})()

    monkeypatch.setattr(
        anthropic, "Anthropic", lambda *a, **kw: type("C", (), {"messages": FakeMessages()})()
    )
    monkeypatch.setenv("JUDGE_MODEL", "claude-test-model")

    result = judge_module._judge_anthropic("the prompt", system="")
    assert result.score == pytest.approx(0.6)
    assert captured["model"] == "claude-test-model"
    assert captured["messages"] == [{"role": "user", "content": "the prompt"}]
    assert "compact JSON" in captured["system"], "default judge system prompt was dropped"


def test_explicit_system_prompt_overrides_the_default(monkeypatch):
    import anthropic

    captured: dict = {}

    class FakeMessages:
        def create(self, **kwargs):
            captured.update(kwargs)
            block = type("Block", (), {"text": '{"score": 1.0}'})()
            return type("Resp", (), {"content": [block]})()

    monkeypatch.setattr(
        anthropic, "Anthropic", lambda *a, **kw: type("C", (), {"messages": FakeMessages()})()
    )
    judge_module._judge_anthropic("p", system="BE TERSE")
    assert captured["system"] == "BE TERSE"


def test_openai_response_is_unwrapped_and_requests_json_mode(monkeypatch):
    import openai

    captured: dict = {}

    class FakeCompletions:
        def create(self, **kwargs):
            captured.update(kwargs)
            message = type("M", (), {"content": '{"score": 0.3, "reasoning": "meh"}'})()
            return type("Resp", (), {"choices": [type("C", (), {"message": message})()]})()

    monkeypatch.setattr(
        openai,
        "OpenAI",
        lambda *a, **kw: type(
            "C", (), {"chat": type("Chat", (), {"completions": FakeCompletions()})()}
        )(),
    )

    result = judge_module._judge_openai("the prompt", system="")
    assert result.score == pytest.approx(0.3)
    assert captured["response_format"] == {"type": "json_object"}


def test_openai_refusal_with_null_content_raises_rather_than_scoring_zero(monkeypatch):
    """`content or ""` turns a refusal into an empty string; that must surface as a
    parse error, not quietly become a 0.0 that drags the run average down."""
    import openai

    class FakeCompletions:
        def create(self, **kwargs):
            message = type("M", (), {"content": None})()
            return type("Resp", (), {"choices": [type("C", (), {"message": message})()]})()

    monkeypatch.setattr(
        openai,
        "OpenAI",
        lambda *a, **kw: type(
            "C", (), {"chat": type("Chat", (), {"completions": FakeCompletions()})()}
        )(),
    )
    with pytest.raises(ValueError, match="not JSON-parseable"):
        judge_module._judge_openai("p", system="")
