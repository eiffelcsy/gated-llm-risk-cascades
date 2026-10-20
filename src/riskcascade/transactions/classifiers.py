"""Supervised tier-2 arms: same dossier as the LLM, a trained model decides.

  EncoderInvestigator   fine-tuned encoder (ModernBERT by default) over the dossier text
  EvidenceGBDT          LightGBM over the dossier's numbers

The encoder asks "does reading the evidence as text beat pattern-matching on
it?"; the GBDT keeps that honest, since the tool outputs are numbers already.
Both are trained only on train-period cases (see pipeline.prepare), and give
their argmax verdict; the override gate does the cost-aware trust decision.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .agent import CaseTools, dossier, dossier_text, evidence
from .features import ENTITY_COLS


class _DossierClassifier:
    """Calls every tool, then a trained model maps the dossier to P(fraud)."""

    def __init__(self, tools: CaseTools):
        self.tools = tools
        self.log: list[dict] = []

    def _dossiers(self, rows: pd.DataFrame) -> list:
        return [dossier(self.tools, rows.iloc[[i]]) for i in range(len(rows))]

    def fit(self, rows: pd.DataFrame, scores: np.ndarray, y) -> "_DossierClassifier":
        self._fit(rows, np.asarray(scores), self._dossiers(rows), np.asarray(y))
        return self

    def judge(self, rows: pd.DataFrame, tier1_scores: np.ndarray):
        logs = self._dossiers(rows)
        p = self._proba(rows, np.asarray(tier1_scores), logs)
        v, conf = (p >= 0.5).astype(int), np.maximum(p, 1 - p)
        for i, s in enumerate(tier1_scores):
            self.log.append({"TransactionID": int(rows["TransactionID"].iloc[i]), "tier1": float(s),
                             "verdict": int(v[i]), "p_fraud": float(p[i])})
        return v, np.array([evidence(log, c) for log, c in zip(logs, conf)])

    def _fit(self, rows, scores, logs, y): raise NotImplementedError
    def _proba(self, rows, scores, logs) -> np.ndarray: raise NotImplementedError


def dossier_numbers(score: float, log: list[tuple[str, dict]]) -> list[float]:
    """Fixed-order numeric view of a dossier (the explain tool's top-k list is left out)."""
    r = dict((res["entity"], res) for name, res in log if name == "entity_history")
    link = next(res for name, res in log if name == "linked_accounts")
    sim = next(res for name, res in log if name == "similar_cases")
    ents = [r[e][k] for e in ENTITY_COLS for k in ("past_transactions", "distinct_accounts", "known_fraud_transactions")]
    return [score, *ents, link["account_labelled_transactions"], link["account_known_fraud"],
            link["linked_accounts"], link["linked_accounts_with_known_fraud"], sim["fraud_rate"]]


class EvidenceGBDT(_DossierClassifier):
    def __init__(self, tools: CaseTools, **params):
        super().__init__(tools)
        self.params = {"n_estimators": 200, "num_leaves": 15, "learning_rate": 0.05, "verbose": -1, **params}

    def _x(self, scores, logs) -> np.ndarray:
        return np.array([dossier_numbers(s, log) for s, log in zip(scores, logs)])

    def _fit(self, rows, scores, logs, y):
        import lightgbm as lgb

        self.model = lgb.LGBMClassifier(**self.params).fit(self._x(scores, logs), y)

    def _proba(self, rows, scores, logs):
        return self.model.predict_proba(self._x(scores, logs))[:, 1]


class EncoderInvestigator(_DossierClassifier):
    """Full fine-tune of a pretrained encoder with a 2-class head on the dossier text."""

    def __init__(self, tools: CaseTools, model: str = "answerdotai/ModernBERT-base", epochs: int = 2,
                 lr: float = 3e-5, batch_size: int = 16, max_len: int = 512, device: str | None = None):
        import torch

        super().__init__(tools)
        self.name, self.epochs, self.lr, self.bs, self.max_len = model, epochs, lr, batch_size, max_len
        self.device = device or ("cuda" if torch.cuda.is_available()
                                 else "mps" if torch.backends.mps.is_available() else "cpu")

    def _texts(self, rows, scores, logs) -> list[str]:
        return [dossier_text(rows.iloc[[i]], s, log) for i, (s, log) in enumerate(zip(scores, logs))]

    def _encode(self, texts: list[str]):
        batch = self.tok(texts, truncation=True, max_length=self.max_len, padding=True, return_tensors="pt")
        return {k: v.to(self.device) for k, v in batch.items()}

    def _fit(self, rows, scores, logs, y):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        torch.manual_seed(0)
        texts = self._texts(rows, scores, logs)
        self.tok = AutoTokenizer.from_pretrained(self.name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.name, num_labels=2).to(self.device)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=0.01)
        labels = torch.as_tensor(y, dtype=torch.long)
        self.model.train()
        for _ in range(self.epochs):
            for idx in torch.randperm(len(texts)).split(self.bs):
                out = self.model(**self._encode([texts[i] for i in idx]), labels=labels[idx].to(self.device))
                out.loss.backward()
                opt.step(); opt.zero_grad()
        self.model.eval()

    def _proba(self, rows, scores, logs):
        import torch

        texts, out = self._texts(rows, scores, logs), []
        with torch.no_grad():
            for i in range(0, len(texts), 64):
                logits = self.model(**self._encode(texts[i:i + 64])).logits
                out.append(torch.softmax(logits.float(), -1)[:, 1].cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)
