import numpy as np
import pytest

from riskcascade.core import Cascade, OverrideGate, metrics, splits


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


def test_routing_and_policies():
    s = np.array([0.1, 0.3, 0.6, 0.9])
    v = np.array([1, 1, 0, 0])
    ev = np.ones(4)
    c = Cascade(ArrT1(s), ArrT2(v, ev), low=0.2, high=0.8)
    r = c.run(list(range(4)), policy="tier1")
    assert r.final.tolist() == [0, 0, 1, 1]
    r = c.run(list(range(4)), policy="always")
    assert r.routed.tolist() == [False, True, True, False]
    assert r.final.tolist() == [0, 1, 0, 1]  # only the band changes


def test_gated_requires_gate():
    c = Cascade(ArrT1(np.array([0.5])), ArrT2(np.array([0]), np.ones(1)))
    with pytest.raises(ValueError):
        c.run([0], policy="gated")


def test_gate_learns_evidence_and_beats_always():
    from importlib.util import spec_from_file_location, module_from_spec
    from pathlib import Path

    spec = spec_from_file_location("run_eval", Path(__file__).parents[1] / "eval" / "run_eval.py")
    mod = module_from_spec(spec); spec.loader.exec_module(mod)
    rows = {r["policy"]: r for r in mod.demo()}
    assert rows["gated"]["cost_per_item"] < rows["always"]["cost_per_item"]
    assert rows["gated"]["broke"] < rows["always"]["broke"]


def test_metrics():
    y = np.array([1, 0, 1, 0]); p = np.array([1, 1, 0, 0])
    assert metrics.cost_weighted_loss(y, p, c_fn=10, c_fp=1) == pytest.approx(11 / 4)
    q = metrics.override_quality(y, base=np.array([0, 0, 0, 0]), final=p)
    assert q == {"n_overrides": 2, "fixed": 1, "broke": 1}
    assert metrics.recall_at_fpr(np.array([1, 1, 0, 0]), np.array([0.9, 0.2, 0.1, 0.3]), 0.0) == 0.5


def test_splits():
    fam = np.array(["benign"] * 30 + ["a"] * 10 + ["b"] * 10)
    tr, va, te = splits.held_out_family(fam, ["b"])
    assert set(fam[te]) >= {"b"} and "b" not in set(fam[tr]) | set(fam[va])
    assert len(set(tr) | set(va) | set(te)) == len(fam)
    t = np.arange(10)
    tr, va, te = splits.temporal(t, 5, 8)
    assert tr.max() < va.min() and va.max() < te.min()
