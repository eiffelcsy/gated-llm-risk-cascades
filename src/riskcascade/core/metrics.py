"""Metrics reported identically for both tracks."""
from __future__ import annotations

import time
from typing import Any

import numpy as np


def cost_weighted_loss(y, pred, c_fn: float = 10.0, c_fp: float = 1.0) -> float:
    """Average cost per item. c_fn = missed risk, c_fp = friction on an honest user."""
    y, pred = np.asarray(y), np.asarray(pred)
    fn = ((y == 1) & (pred == 0)).sum()
    fp = ((y == 0) & (pred == 1)).sum()
    return float((c_fn * fn + c_fp * fp) / max(len(y), 1))


def best_threshold(y, scores, c_fn: float, c_fp: float) -> float:
    """Score threshold (block if s >= thr) that minimises cost-weighted loss."""
    y, s = np.asarray(y), np.asarray(scores, dtype=float)
    order = np.argsort(-s)
    s_sorted, y_sorted = s[order], y[order]
    # blocking the top k items: cost = c_fp * negatives in top k + c_fn * positives below k
    fp = np.concatenate([[0], np.cumsum(y_sorted == 0)])
    fn = (y == 1).sum() - np.concatenate([[0], np.cumsum(y_sorted == 1)])
    cost = c_fp * fp + c_fn * fn
    # only cut between distinct scores, so the threshold is realisable
    valid = np.r_[True, s_sorted[1:] != s_sorted[:-1], True]
    k = int(np.flatnonzero(valid)[np.argmin(cost[valid])])
    return float(s_sorted[k - 1]) if k > 0 else float(np.nextafter(s_sorted[0], np.inf))


def recall_at_fpr(y, scores, max_fpr: float = 0.01) -> float:
    """Recall at the strictest threshold whose FPR on negatives is <= max_fpr."""
    y, scores = np.asarray(y), np.asarray(scores, dtype=float)
    neg = np.sort(scores[y == 0])[::-1]
    if len(neg) == 0 or (y == 1).sum() == 0:
        return float("nan")
    k = int(np.floor(max_fpr * len(neg)))
    thr = neg[k] if k < len(neg) else -np.inf
    return float((scores[y == 1] > thr).mean())


def decision_rates(y, pred) -> dict:
    y, pred = np.asarray(y), np.asarray(pred)
    tp = ((y == 1) & (pred == 1)).sum(); fp = ((y == 0) & (pred == 1)).sum()
    fn = ((y == 1) & (pred == 0)).sum(); tn = ((y == 0) & (pred == 0)).sum()
    return {
        "precision": float(tp / max(tp + fp, 1)),
        "recall": float(tp / max(tp + fn, 1)),
        "fpr": float(fp / max(fp + tn, 1)),
    }


def override_quality(y, base, final) -> dict:
    """Of the decisions tier 2 changed, how many it fixed vs. broke."""
    y, base, final = map(np.asarray, (y, base, final))
    changed = base != final
    return {
        "n_overrides": int(changed.sum()),
        "fixed": int((changed & (final == y)).sum()),
        "broke": int((changed & (base == y)).sum()),
    }


def p99_ms(latencies_ms) -> float:
    return float(np.percentile(np.asarray(latencies_ms, dtype=float), 99))


def single_item_latency_ms(score_fn, items: Any, n: int = 200, seed: int = 0) -> np.ndarray:
    """Time score_fn on one item at a time: the online setting, unlike batch ms/item."""
    from .cascade import take

    idx = np.random.default_rng(seed).choice(len(items), size=min(n, len(items)), replace=False)
    out = []
    for i in idx:
        one = take(items, np.array([i]))
        t0 = time.perf_counter()
        score_fn(one)
        out.append((time.perf_counter() - t0) * 1000)
    return np.array(out)
