"""LLM-as-judge wrappers. Single entry-point `judge(prompt) -> JudgeResult`.

We deliberately keep this thin: prompts live in metrics.py next to the metric
they serve. This file only wires the chosen model and handles retries.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass
class JudgeResult:
    score: float          # 0..1
    reasoning: str        # short justification from the judge
    raw: str              # raw response for debugging


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10))
def judge(prompt: str, *, system: str = "") -> JudgeResult:
    """Send a prompt to the configured judge model and parse a 0..1 score.

    Judge prompts must ask the model to respond as compact JSON:
        {"score": 0.0-1.0, "reasoning": "..."}

    The harness uses Anthropic by default; falls back to OpenAI if only the
    OPENAI key is set. This avoids forcing engineers to maintain two providers.
    """
    if os.getenv("ANTHROPIC_API_KEY"):
        return _judge_anthropic(prompt, system=system)
    if os.getenv("OPENAI_API_KEY"):
        return _judge_openai(prompt, system=system)
    raise RuntimeError("No judge API key set. Set ANTHROPIC_API_KEY or OPENAI_API_KEY in .env")


def _judge_anthropic(prompt: str, *, system: str) -> JudgeResult:
    from anthropic import Anthropic

    client = Anthropic()
    model = os.getenv("JUDGE_MODEL", "claude-sonnet-4-5")
    resp = client.messages.create(
        model=model,
        max_tokens=512,
        system=system or "You are a strict evaluation judge. Always respond with compact JSON.",
        messages=[{"role": "user", "content": prompt}],
    )
    raw = resp.content[0].text
    return _parse(raw)


def _judge_openai(prompt: str, *, system: str) -> JudgeResult:
    from openai import OpenAI

    client = OpenAI()
    model = os.getenv("JUDGE_MODEL", "gpt-4o-mini")
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system or "You are a strict evaluation judge. Always respond with compact JSON."},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
    )
    raw = resp.choices[0].message.content or ""
    return _parse(raw)


def _parse(raw: str) -> JudgeResult:
    """Be tolerant: try strict JSON, fall back to extracting the first {...} block."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        start = raw.find("{")
        end = raw.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"Judge response was not JSON-parseable: {raw[:200]}")
        data = json.loads(raw[start : end + 1])
    score = float(data.get("score", 0.0))
    score = max(0.0, min(1.0, score))   # clamp defensively
    return JudgeResult(score=score, reasoning=str(data.get("reasoning", "")).strip(), raw=raw)
