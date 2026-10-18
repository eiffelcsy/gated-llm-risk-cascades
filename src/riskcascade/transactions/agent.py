"""Tier 2 for transactions: a plain tool-calling investigation loop.

Four tools, each a pure function over data before the case's timestamp:
  entity_history(entity, value)   past transactions sharing this card/email/device
  linked_accounts(case)           accounts sharing any entity with this one
  explain(case)                   top SHAP features from tier 1
  similar_cases(case, k)          k nearest past labelled cases

Budget: at most N tool calls per case. Output: verdict + short rationale.

Evidence features for the override gate:
  - number of tool calls used
  - number of linked accounts with past fraud labels
  - self-consistency over k sampled verdicts
  - whether the rationale cites a tool result (cheap regex check)

Known risk (arXiv 2607.19266): the agent writes convincing rationales for
wrong overrides. Log every override with its rationale and read a sample by hand.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np


class InvestigationAgent:
    def __init__(self, llm: str = "Qwen/Qwen2.5-7B-Instruct", max_tool_calls: int = 6, k_samples: int = 3):
        self.llm_name = llm
        self.max_tool_calls = max_tool_calls
        self.k = k_samples

    def judge(self, cases: Sequence, tier1_scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return verdicts (n,) and evidence features (n, 4)."""
        raise NotImplementedError
