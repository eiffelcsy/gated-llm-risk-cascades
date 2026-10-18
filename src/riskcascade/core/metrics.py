"""Metrics reported identically for both tracks."""
from __future__ import annotations

import numpy as np


def cost_weighted_loss(y: np.ndarray, pred: np.ndarray, c_fn: float = 10.0, c_fp: float = 1.0) -> float:
    """Average cost per item. c_fn = missed risk, c_fp = friction on an honest user."""
    y, pred = np.asarray(y), np.asarray(pred)
    fn = ((y == 1) & (pred == 0)).sum()
    fp = ((y == 0) & (pred == 1)).sum()
    return float((c_fn * fn + c_fp * fp) / max(len(y), 1))


def recall_at_fpr(y: np.ndarray, scores: np.ndarray, max_fpr: float = 0.01) -> float:
    """Recall at the strictest threshold whose FPR on negatives is <= max_fpr."""
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    neg = np.sort(scores[y == 0])[::-1]
    if len(neg) == 0 or (y == 1).sum() == 0:
        return float("nan")
    k = int(np.floor(max_fpr * len(neg)))
    thr = neg[k] if k < len(neg) else -np.inf
    return float((scores[y == 1] > thr).mean())


def decision_rates(y: np.ndarray, pred: np.ndarray) -> dict:
    y, pred = np.asarray(y), np.asarray(pred)
    tp = ((y == 1) & (pred == 1)).sum(); fp = ((y == 0) & (pred == 1)).sum()
    fn = ((y == 1) & (pred == 0)).sum(); tn = ((y == 0) & (pred == 0)).sum()
    return {
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "fpr": float(fp / max(fp + tn, 1)),
    }


def override_quality(y: np.ndarray, base: np.ndarray, final: np.ndarray) -> dict:
    """Of the decisions tier 2 changed, how many it fixed vs. broke."""
    y, base, final = map(np.asarray, (y, base, final))
    changed = base != final
    return {
        "n_overrides": int(changed.sum()),
        "fixed": int((changed & (final == y)).sum()),
        "broke": int((changed & (base == y)).sum()),
    }


def p99_ms(latencies_ms: np.ndarray) -> float:
    return float(np.percentile(np.asarray(latencies_ms, dtype=float), 99))
