import numpy as np
import pytest

from riskcascade.core import (Cascade, OverrideGate, gate_ablation, metrics, operating_point, rolling_gate,
                              run_protocol, splits)


class ArrT1:
    def __init__(self, s):
        self.s = s

    def score(self, items):
        return self.s[np.asarray(items)]


class ArrT2:
    def __init__(self, v, ev):
        self.v, self.ev = v, ev

    def judge(self, items, scores):
        i = np.asarray(items)
        return self.v[i], self.ev[i, None]


def toy(n=4000, seed=0):
    """Tier 1 is noisy; tier 2 is right more often when its evidence feature is high."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.2).astype(int)
    s = np.clip(y * 0.35 + rng.normal(0.35, 0.2, n), 0, 1)
    evidence = rng.random(n)
    right = rng.random(n) < 0.4 + 0.55 * evidence
    return y, s, np.where(right, y, 1 - y), evidence


def test_routing_and_policies():
    s = np.array([0.1, 0.3, 0.6, 0.9])
    c = Cascade(ArrT1(s), ArrT2(np.array([1, 1, 0, 0]), np.ones(4)), threshold=0.5, low=0.2, high=0.8)
    tr = c.trace(list(range(4)))
    assert tr.routed.tolist() == [False, True, True, False]
    assert c.decide(tr, "tier1").tolist() == [0, 0, 1, 1]
    assert c.decide(tr, "always").tolist() == [0, 1, 0, 1]  # only the band changes
    with pytest.raises(ValueError):
        c.decide(tr, "gated")  # no gate fitted


def test_abstain_never_overrides():
    s = np.array([0.3, 0.6])
    c = Cascade(ArrT1(s), ArrT2(np.array([-1, -1]), np.ones(2)), threshold=0.5, low=0.2, high=0.8)
    assert c.run([0, 1], policy="always").tolist() == [0, 1]


def test_gate_is_cost_aware():
    g = OverrideGate(threshold=0.5, c_fn=10, c_fp=1)
    g._constant = 0.5  # tier 2 right half the time
    # flipping to block risks a cheap FP (accept); flipping to allow risks a costly FN (reject)
    assert g.accept(np.array([0.4, 0.6]), np.array([1, 0]), np.zeros((2, 1))).tolist() == [True, False]


def test_gate_learns_evidence_and_beats_always():
    y, s, v, ev = toy()
    items = np.arange(len(y))
    val, test = items[: len(y) // 2], items[len(y) // 2:]
    run = run_protocol(ArrT1(s), ArrT2(v, ev), list(val), y[val], list(test), y[test], c_fn=10, c_fp=1, budget=0.4)
    rows = {r["policy"]: r for r in run.rows}
    assert rows["gated"]["cost_per_item"] < rows["always"]["cost_per_item"]
    assert rows["gated"]["broke"] < rows["always"]["broke"]
    assert rows["oracle"]["cost_per_item"] <= rows["gated"]["cost_per_item"]
    assert rows["oracle"]["broke"] == 0

    ablation = {r["policy"]: r for r in gate_ablation(run)}
    assert len(ablation) == 9
    # evidence is what makes the gate useful here
    assert ablation["gated[logreg,all]"]["cost_per_item"] < ablation["gated[logreg,score]"]["cost_per_item"]


def test_rolling_gate_respects_label_delay():
    y, s, v, ev = toy(2000)
    items, t = np.arange(2000), np.arange(2000, dtype=float)
    val, test = items[:1000], items[1000:]
    run = run_protocol(ArrT1(s), ArrT2(v, ev), list(val), y[val], list(test), y[test], 10, 1, 0.4)
    final = rolling_gate(run, t[val], t[test], starts=[1000, 1500], window=400, delay=100)
    assert final.shape == (1000,)
    assert set(np.unique(final)) <= {0, 1}


def test_operating_point_routes_budget():
    rng = np.random.default_rng(0)
    y = (rng.random(5000) < 0.1).astype(int)
    s = rng.beta(1, 8, 5000) + 0.3 * y
    thr, low, high = operating_point(s, y, 10, 1, budget=0.2)
    assert low <= thr < high
    assert ((s >= low) & (s < high)).mean() == pytest.approx(0.2, abs=0.01)


def test_metrics():
    y = np.array([1, 0, 1, 0]); p = np.array([1, 1, 0, 0])
    assert metrics.cost_weighted_loss(y, p, c_fn=10, c_fp=1) == pytest.approx(11 / 4)
    assert metrics.override_quality(y, base=np.zeros(4), final=p) == {"n_overrides": 2, "fixed": 1, "broke": 1}
    assert metrics.recall_at_fpr(np.array([1, 1, 0, 0]), np.array([0.9, 0.2, 0.1, 0.3]), 0.0) == 0.5
    # with FN 10x costlier, blocking everything above the lowest positive is optimal here
    assert metrics.best_threshold(np.array([1, 0, 1, 0]), np.array([0.9, 0.8, 0.3, 0.1]), 10, 1) == 0.3


def test_splits():
    fam = np.array(["benign"] * 30 + ["a"] * 10 + ["b"] * 10)
    tr, va, te = splits.held_out_family(fam, ["b"])
    assert set(fam[te]) >= {"b"} and "b" not in set(fam[tr]) | set(fam[va])
    assert len(set(tr) | set(va) | set(te)) == len(fam)
    tr, va, te = splits.temporal(np.arange(10), 5, 8)
    assert tr.max() < va.min() and va.max() < te.min()
