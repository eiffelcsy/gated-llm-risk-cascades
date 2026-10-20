"""The shared protocol: fit on train, choose operating point and gate on val, report on test.

Both tracks call `run_protocol`, so their result tables are directly comparable.
Gate experiments (ablation, rolling refits) reuse the stored traces, so tier 2
never runs twice on the same item.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from . import metrics
from .cascade import POLICIES, Cascade, Tier1, Tier2, Trace, operating_point
from .gate import FEATURE_SETS, MODELS


@dataclass
class ProtocolRun:
    cascade: Cascade
    val: Trace
    test: Trace
    y_val: np.ndarray
    y_test: np.ndarray
    c_fn: float
    c_fp: float
    rows: list[dict]


def run_protocol(tier1: Tier1, tier2: Tier2, val_items: Any, y_val, test_items: Any, y_test,
                 c_fn: float, c_fp: float, budget: float, groups=None) -> ProtocolRun:
    """tier1 (and any supervised tier 2) must already be fit on train-period data only."""
    y_val, y_test = np.asarray(y_val), np.asarray(y_test)
    thr, low, high = operating_point(tier1.score(val_items), y_val, c_fn, c_fp, budget)
    cascade = Cascade(tier1, tier2, thr, low, high)
    val = cascade.trace(val_items)
    cascade.gate = cascade.fit_gate(val, y_val, c_fn, c_fp)
    test = cascade.trace(test_items)
    latency = metrics.single_item_latency_ms(tier1.score, test_items)

    finals = {p: cascade.decide(test, p) for p in POLICIES}
    # oracle: accept an override only when tier 2 is right -- the ceiling for any gate
    finals["oracle"] = np.where(test.disagreements() & (test.verdicts == y_test), test.verdicts, test.base)
    rows = [row(p, y_test, test, f, c_fn, c_fp, latency) for p, f in finals.items()]
    if groups is not None:
        rows += group_rows(finals, y_test, test.base, np.asarray(groups), c_fn, c_fp)
    return ProtocolRun(cascade, val, test, y_val, y_test, c_fn, c_fp, rows)


def gate_ablation(run: ProtocolRun) -> list[dict]:
    """Every gate model x feature set, refit on the same validation trace and scored on the same test trace."""
    out = []
    for model in MODELS:
        for features in FEATURE_SETS:
            gate = run.cascade.fit_gate(run.val, run.y_val, run.c_fn, run.c_fp, model=model, features=features)
            final = run.cascade.decide(run.test, "gated", gate=gate)
            out.append(row(f"gated[{model},{features}]", run.y_test, run.test, final, run.c_fn, run.c_fp))
    return out


def rolling_gate(run: ProtocolRun, t_val, t_test, starts, window: float, delay: float) -> np.ndarray:
    """Refit the gate at each start time on the most recent labelled window, then apply it until the next start.

    Labels are usable only once they are `delay` old, so the gate used from T
    is fit on items in [T - delay - window, T - delay). Returns test decisions.
    """
    both = _concat(run.val, run.test)
    y, t = np.r_[run.y_val, run.y_test], np.r_[t_val, t_test]
    final = both.base.copy()
    bounds = list(starts) + [np.inf]
    for t0, t1 in zip(bounds[:-1], bounds[1:]):
        fit = (t >= t0 - delay - window) & (t < t0 - delay)
        gate = run.cascade.fit_gate(both, y, run.c_fn, run.c_fp, mask=fit)
        apply = (t >= t0) & (t < t1)
        final[apply] = run.cascade.decide(both, "gated", gate=gate, mask=apply)[apply]
    return final[len(run.y_val):]


def _concat(a: Trace, b: Trace) -> Trace:
    return replace(a, scores=np.r_[a.scores, b.scores], base=np.r_[a.base, b.base],
                   routed=np.r_[a.routed, b.routed], verdicts=np.r_[a.verdicts, b.verdicts],
                   evidence=np.vstack([a.evidence, b.evidence]))


def row(policy: str, y, tr: Trace, final, c_fn: float, c_fp: float, tier1_latency_ms=None) -> dict:
    """One results row. An override pins the item's score to 0 or 1, so recall@FPR reflects the policy."""
    band = tr.routed
    final_scores = np.where(final != tr.base, final.astype(float), tr.scores)
    used_tier2 = policy != "tier1"
    return {
        "policy": policy,
        "cost_per_item": metrics.cost_weighted_loss(y, final, c_fn, c_fp),
        **metrics.decision_rates(y, final),
        "recall@1%fpr": metrics.recall_at_fpr(y, final_scores, 0.01),
        **metrics.override_quality(y, tr.base, final),
        "routed_share": tr.routed_share if used_tier2 else 0.0,
        "tier1_band_acc": float((tr.base[band] == y[band]).mean()) if band.any() else float("nan"),
        "tier2_band_acc": float((tr.verdicts[band] == y[band]).mean()) if band.any() else float("nan"),
        "tier2_abstain": float((tr.verdicts[band] < 0).mean()) if band.any() else float("nan"),
        "tier1_p99_ms": metrics.p99_ms(tier1_latency_ms) if tier1_latency_ms is not None else float("nan"),
        "tier2_ms_per_routed": tr.tier2_ms if used_tier2 else 0.0,
    }


def group_rows(finals: dict, y, base, groups, c_fn: float, c_fp: float) -> list[dict]:
    """Cost per item and override quality per group (e.g. month) for each policy."""
    out = []
    for policy, final in finals.items():
        for g in np.unique(groups):
            m = groups == g
            out.append({"policy": policy, "group": g,
                        "cost_per_item": metrics.cost_weighted_loss(y[m], final[m], c_fn, c_fp),
                        **metrics.override_quality(y[m], base[m], final[m])})
    return out


TABLE_COLS = ["policy", "cost_per_item", "recall", "fpr", "recall@1%fpr", "routed_share",
              "fixed", "broke", "tier1_band_acc", "tier2_band_acc", "tier2_abstain", "tier1_p99_ms"]


def format_table(track: str, rows: list[dict], cols=TABLE_COLS) -> str:
    """Markdown table of the headline rows (no per-group rows)."""
    head = [r for r in rows if "group" not in r]
    lines = [f"| track | {' | '.join(cols)} |", "|" + "---|" * (len(cols) + 1)]
    for r in head:
        cells = [f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]) for c in cols]
        lines.append(f"| {track} | {' | '.join(cells)} |")
    return "\n".join(lines)


def format_groups(rows: list[dict]) -> str:
    """Cost per item by group (e.g. month) for each policy."""
    grouped = [r for r in rows if "group" in r]
    if not grouped:
        return ""
    groups = sorted({r["group"] for r in grouped})
    policies = list(dict.fromkeys(r["policy"] for r in grouped))
    cost = {(r["policy"], r["group"]): r["cost_per_item"] for r in grouped}
    lines = ["| group | " + " | ".join(policies) + " |", "|" + "---|" * (len(policies) + 1)]
    for g in groups:
        lines.append(f"| {g} | " + " | ".join(f"{cost[(p, g)]:.4f}" for p in policies) + " |")
    return "\n".join(lines)
