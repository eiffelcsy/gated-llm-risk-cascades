"""Minimal LLM plumbing shared by both tier-2 judges.

Any OpenAI-compatible endpoint works (vLLM or Ollama serving Qwen locally,
or a hosted API), so the open-weight model in the config is swappable.
"""
from __future__ import annotations

import json
import os
import re


def make_client(base_url: str | None = None):
    from openai import OpenAI

    return OpenAI(base_url=base_url or os.getenv("LLM_BASE_URL", "http://localhost:8000/v1"),
                  api_key=os.getenv("LLM_API_KEY", "EMPTY"))


def parse_json(text: str | None) -> dict:
    """Last {...} block in a model reply, or {} if none parses."""
    for block in reversed(re.findall(r"\{.*?\}", text or "", flags=re.S)):
        try:
            return json.loads(block)
        except json.JSONDecodeError:
            continue
    return {}


def majority(verdicts: list[int]) -> tuple[int, float]:
    """Majority verdict over k samples and the fraction that agree with it (self-consistency)."""
    if not verdicts:
        return 0, 0.0
    v = int(sum(verdicts) * 2 > len(verdicts))
    return v, sum(x == v for x in verdicts) / len(verdicts)
