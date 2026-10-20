"""Tier 2 for transactions: four investigation tools and an LLM investigator.

Tools only see the past. A case at time t sees transactions before t, and
fraud labels only for transactions older than t - label_delay (chargebacks
arrive late):
  entity_history(entity)   past transactions sharing this card/address/email/device
  linked_accounts()        this account's labelled history, and linked accounts (shared card or
                           device) with known fraud
  explain()                top tier-1 SHAP contributions
  similar_cases(k)         fraud rate among the k nearest labelled cases

A "dossier" is every tool's output for one case. It is the shared input of
all tier-2 arms (LLM here, encoder and evidence-GBDT in classifiers.py), so
they differ only in how they decide.

The LLM runs in one of two modes:
  agent    chooses which tools to call (up to max_tool_calls), multi-turn
  dossier  reads the full dossier in one turn (the mode GRPO post-training uses)
It may abstain (-1) when evidence is inconclusive, so tier 1 stands.

Known risk (arXiv 2607.19266): the agent writes convincing rationales for
wrong overrides. Every verdict is logged with its rationale for manual review.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from ..core.llm import majority, make_client, parse_json
from .features import ENTITY_COLS
from .scorer import LgbmScorer

DAY = 86400
LINK_COLS = ["card1", "DeviceInfo"]  # addresses (regions) and email domains are too broad to link accounts
TOOL_NAMES = ("entity_history", "linked_accounts", "explain", "similar_cases")
CASE_FIELDS = {"TransactionAmt": "amount", "ProductCD": "product", "P_emaildomain": "email",
               "DeviceInfo": "device", "n_tx_3600s": "tx_last_hour", "n_tx_86400s": "tx_last_day",
               "secs_since_prev": "secs_since_prev", "account_n_prev": "account_prev_tx"}


class CaseTools:
    """Time-bounded lookups over the full transaction history (sorted by time).

    Every tool takes `row`, a one-row DataFrame for the case under investigation.
    """

    def __init__(self, history: pd.DataFrame, scorer: LgbmScorer, labelled: pd.DataFrame,
                 label_delay_days: float = 14, max_link: int = 200):
        self.scorer, self.delay, self.max_link = scorer, label_delay_days * DAY, max_link
        self.t = history["TransactionDT"].to_numpy()
        self.y = history["isFraud"].to_numpy()
        self.acct = history["account"].to_numpy()
        self.index = {c: history.groupby(c, observed=True).indices for c in ENTITY_COLS + ["account"]}
        # kNN over labelled rows; the caller passes only rows whose labels are visible to every case
        self.knn_cols = [c for c in scorer.cols if pd.api.types.is_numeric_dtype(labelled[c])]
        self.scaler = StandardScaler().fit(self._knn_x(labelled))
        self.knn = NearestNeighbors().fit(self.scaler.transform(self._knn_x(labelled)))
        self.knn_y = labelled["isFraud"].to_numpy()

    def _knn_x(self, df: pd.DataFrame) -> np.ndarray:
        return df[self.knn_cols].fillna(-1).to_numpy(float)

    def _past(self, col: str, value, t: float) -> np.ndarray:
        """Row positions sharing `value` in `col`, strictly before time t."""
        pos = self.index[col].get(value, np.array([], dtype=int))
        return pos[: np.searchsorted(self.t[pos], t, side="left")]

    def _known_fraud(self, pos: np.ndarray, t: float) -> int:
        """Fraud labels among `pos` that have arrived by time t."""
        return int(((self.t[pos] < t - self.delay) & (self.y[pos] == 1)).sum())

    def entity_history(self, row: pd.DataFrame, entity: str) -> dict:
        if entity not in ENTITY_COLS:
            return {"error": f"entity must be one of {ENTITY_COLS}"}
        c = row.iloc[0]
        pos = self._past(entity, c[entity], c["TransactionDT"])
        return {"entity": entity, "value": str(c[entity]), "past_transactions": len(pos),
                "distinct_accounts": len(set(self.acct[pos])),
                "known_fraud_transactions": self._known_fraud(pos, c["TransactionDT"])}

    def linked_accounts(self, row: pd.DataFrame) -> dict:
        c = row.iloc[0]
        t, linked, skipped = c["TransactionDT"], set(), []
        for col in LINK_COLS:
            accounts = set(self.acct[self._past(col, c[col], t)])
            if len(accounts) > self.max_link:  # e.g. DeviceInfo="Windows": shared by everyone, not a link
                skipped.append(col)
            else:
                linked |= accounts
        linked.discard(c["account"])
        fraud = sum(self._known_fraud(self._past("account", a, t), t) > 0 for a in linked)
        own = self._past("account", c["account"], t)
        own_fraud = self._known_fraud(own, t)
        return {"account_labelled_transactions": int((self.t[own] < t - self.delay).sum()),
                "account_known_fraud": own_fraud, "linked_accounts": len(linked),
                "linked_accounts_with_known_fraud": fraud + int(own_fraud > 0),
                "ignored_common_entities": skipped}

    def explain(self, row: pd.DataFrame) -> dict:
        return {"top_tier1_contributions": self.scorer.explain(row)}

    def similar_cases(self, row: pd.DataFrame, k: int = 10) -> dict:
        k = int(min(max(k, 1), 50))
        _, nn = self.knn.kneighbors(self.scaler.transform(self._knn_x(row)), n_neighbors=k)
        return {"k": k, "fraud_rate": round(float(self.knn_y[nn[0]].mean()), 3)}

    def call(self, name: str, row: pd.DataFrame, **args) -> dict:
        if name not in TOOL_NAMES:
            return {"error": f"unknown tool {name}"}
        try:
            return getattr(self, name)(row, **args)
        except TypeError as e:  # bad arguments from the model
            return {"error": str(e)}


# ---- dossier: the shared tier-2 input -------------------------------------------------

def dossier(tools: CaseTools, row: pd.DataFrame) -> list[tuple[str, dict]]:
    """Every tool, called once, as a (tool name, result) log."""
    log = [("entity_history", tools.entity_history(row, e)) for e in ENTITY_COLS]
    return log + [("linked_accounts", tools.linked_accounts(row)), ("explain", tools.explain(row)),
                  ("similar_cases", tools.similar_cases(row))]


def case_text(row: pd.DataFrame, score: float) -> str:
    c = row.iloc[0]
    fields = " ".join(f"{name}={_fmt(c[col])}" for col, name in CASE_FIELDS.items() if col in row)
    return f"fast_model_score={score:.3f}\ntransaction: {fields}"


def dossier_text(row: pd.DataFrame, score: float, log: list[tuple[str, dict]]) -> str:
    """Compact text form (~150 tokens), so encoders with 512-token context fit it."""
    lines = [case_text(row, score)]
    for name, r in log:
        if name == "explain":
            body = ", ".join(f"{f}={v:+.2f}" for f, v in r["top_tier1_contributions"])
        else:
            body = " ".join(f"{k}={_fmt(v)}" for k, v in r.items() if k not in ("entity", "value"))
            body = f"{r['entity']}: {body}" if "entity" in r else body
        lines.append(f"{name} {body}")
    return "\n".join(lines)


def evidence(log: list[tuple[str, dict]], confidence: float) -> list[float]:
    """Gate features from what the tools returned (0 if a tool was not called):
    [tool calls, linked accounts with known fraud, similar-case fraud rate,
     clean labelled history of this account, tier-2 confidence]."""
    links = [r for n, r in log if n == "linked_accounts" and "error" not in r]
    linked = max((r["linked_accounts_with_known_fraud"] for r in links), default=0)
    clean = max((r["account_labelled_transactions"] for r in links if r["account_known_fraud"] == 0), default=0)
    similar = max((r.get("fraud_rate", 0.0) for n, r in log if n == "similar_cases"), default=0.0)
    return [len(log), linked, similar, clean, confidence]


def _fmt(v) -> str:
    if isinstance(v, (list, tuple)):
        return ",".join(map(str, v)) or "none"
    if isinstance(v, (float, np.floating)):
        return "nan" if np.isnan(v) else f"{v:.4g}"
    return str(v)


# ---- LLM investigator ---------------------------------------------------------------

RULES = """Answer with JSON only: {"verdict": 1 if fraud, 0 if legitimate, -1 if inconclusive, "rationale": "<two sentences citing evidence>"}
Wrongly allowing fraud costs 10x more than wrongly blocking a customer.
Only give 0 or 1 when the evidence supports it; finding no fraud link is not proof of legitimacy.
Generic devices like "Windows" are shared by many honest users and are not evidence on their own."""

AGENT_SYSTEM = ("You are a payment-fraud investigator. A fast model scored this transaction as uncertain.\n"
                "Use the tools to gather evidence (at most {budget} calls), then answer.\n" + RULES)
DOSSIER_SYSTEM = ("You are a payment-fraud investigator. A fast model scored this transaction as uncertain.\n"
                  "You are given every investigation tool's output. Reason briefly, then answer.\n" + RULES)


def dossier_messages(text: str) -> list[dict]:
    """Prompt used by dossier mode at inference and by GRPO post-training, so they match exactly."""
    return [{"role": "system", "content": DOSSIER_SYSTEM}, {"role": "user", "content": text}]


def parse_verdict(text: str | None) -> tuple[int, str]:
    out = parse_json(text)
    v = out.get("verdict")
    return (v if v in (0, 1) else -1), str(out.get("rationale", ""))  # unparseable = abstain


TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "entity_history", "description": "Past transactions sharing one entity with this case.",
        "parameters": {"type": "object", "properties": {"entity": {"type": "string", "enum": ENTITY_COLS}},
                       "required": ["entity"]}}},
    {"type": "function", "function": {
        "name": "linked_accounts",
        "description": "This account's labelled history, and accounts sharing a card or device with it that had known fraud.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "explain", "description": "Top features behind the fast model's score.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "similar_cases", "description": "Fraud rate among the k most similar past labelled cases.",
        "parameters": {"type": "object", "properties": {"k": {"type": "integer"}}}}},
]


class InvestigationAgent:
    """LLM investigator over an OpenAI-compatible endpoint (vLLM / Ollama serving Qwen, or a hosted API).

    k sampled runs per case give a majority verdict and a self-consistency signal.
    `llm` can name a GRPO post-trained checkpoint served the same way.
    """

    def __init__(self, tools: CaseTools, llm: str = "Qwen/Qwen2.5-7B-Instruct", base_url: str | None = None,
                 mode: str = "agent", max_tool_calls: int = 6, k_samples: int = 3, temperature: float = 0.7):
        if mode not in ("agent", "dossier"):
            raise ValueError("mode must be 'agent' or 'dossier'")
        self.tools, self.model, self.mode = tools, llm, mode
        self.client = make_client(base_url)
        self.budget, self.k, self.temperature = max_tool_calls, k_samples, temperature
        self.log: list[dict] = []

    def _chat(self, msgs: list, **kwargs):
        return self.client.chat.completions.create(
            model=self.model, messages=msgs, temperature=self.temperature, **kwargs).choices[0].message

    def _run_dossier(self, row: pd.DataFrame, s: float, log: list) -> tuple[int, str, list]:
        v, why = parse_verdict(self._chat(dossier_messages(dossier_text(row, s, log))).content)
        return v, why, log

    def _run_agent(self, row: pd.DataFrame, s: float) -> tuple[int, str, list]:
        msgs = [{"role": "system", "content": AGENT_SYSTEM.format(budget=self.budget)},
                {"role": "user", "content": case_text(row, s)}]
        log: list[tuple[str, dict]] = []
        while True:
            tools = {"tools": TOOL_SPECS} if len(log) < self.budget else {}  # budget spent -> must answer
            msg = self._chat(msgs, **tools)
            if not msg.tool_calls:
                return (*parse_verdict(msg.content), log)
            msgs.append(msg.model_dump(exclude_none=True))
            for call in msg.tool_calls:
                result = self.tools.call(call.function.name, row, **parse_json(call.function.arguments))
                log.append((call.function.name, result))
                msgs.append({"role": "tool", "tool_call_id": call.id, "content": json.dumps(result, default=str)})

    def judge(self, cases: pd.DataFrame, tier1_scores: np.ndarray):
        verdicts, ev = [], []
        for i, s in enumerate(tier1_scores):
            row = cases.iloc[[i]]
            if self.mode == "dossier":
                log = dossier(self.tools, row)  # same evidence for every sample
                runs = [self._run_dossier(row, s, log) for _ in range(self.k)]
            else:
                runs = [self._run_agent(row, s) for _ in range(self.k)]
            votes = [r[0] for r in runs if r[0] >= 0]
            v, agreement = majority(votes) if votes else (-1, 0.0)  # all runs abstained: tier 1 stands
            agreement *= len(votes) / self.k  # abstaining runs count against consistency
            verdicts.append(v)
            ev.append(np.mean([evidence(r[2], agreement) for r in runs], axis=0))
            self.log.append({"TransactionID": int(row["TransactionID"].iloc[0]), "tier1": float(s), "verdict": v,
                             "agreement": agreement, "rationales": [r[1] for r in runs]})
        return np.array(verdicts), np.array(ev)
