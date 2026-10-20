"""Transactions track end to end.

Time is split four ways (all boundaries in days, see configs/transactions.yaml):

  | tier-1 train | tier-2 train | val | test (drifted months) |
                 ^ investigator_days before train_end

- tier 1 (LightGBM) trains on the first block only, so its scores on the
  tier-2 training block are out-of-sample, like at deployment.
- supervised tier-2 arms and GRPO train on the second block.
- similar_cases indexes tier-1 rows old enough that their labels are visible to every later case.
- val picks the operating point and fits the gate; test is reported.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..core import gate_ablation, operating_point, rolling_gate, run_protocol, splits
from ..core.evaluate import group_rows, row
from .agent import CaseTools, InvestigationAgent
from .classifiers import EncoderInvestigator, EvidenceGBDT
from .data import DAY, load_ieee_cis, month_of, synthetic_transactions
from .features import build_features, feature_columns
from .scorer import LgbmScorer

ARMS = ("llm_agent", "llm_dossier", "encoder", "gbdt")


@dataclass
class Prepared:
    df: pd.DataFrame
    y: np.ndarray
    tier1_idx: np.ndarray
    tier2_idx: np.ndarray   # tier-2 training rows inside a widened uncertainty band
    val_idx: np.ndarray
    test_idx: np.ndarray
    scorer: LgbmScorer
    tools: CaseTools


def prepare(cfg: dict) -> Prepared:
    src = cfg["data"]["source"]
    df = build_features(synthetic_transactions() if src == "synthetic" else load_ieee_cis(cfg["data"]["ieee_cis"]))
    y, t = df["isFraud"].to_numpy(), df["TransactionDT"].to_numpy()

    sp, delay = cfg["split"], cfg["tier2"]["label_delay_days"] * DAY
    train_end, val_end = sp["train_end_day"] * DAY, sp["val_end_day"] * DAY
    t2_start = train_end - sp["investigator_days"] * DAY
    train, val, test = splits.temporal(t, train_end, val_end)
    tier1, tier2 = train[t[train] < t2_start], train[t[train] >= t2_start]

    scorer = LgbmScorer(feature_columns(df), **cfg["tier1"]).fit(df.iloc[tier1])
    tools = CaseTools(df, scorer, df.iloc[tier1[t[tier1] < t2_start - delay]],
                      label_delay_days=cfg["tier2"]["label_delay_days"])

    # train tier 2 where it will be used: near the threshold (band widened so there is enough data)
    c, budget = cfg["costs"], cfg["cascade"]["budget"] * cfg["tier2"]["train_band_mult"]
    _, low, high = operating_point(scorer.score(df.iloc[val]), y[val], c["c_fn"], c["c_fp"], budget)
    s2 = scorer.score(df.iloc[tier2])
    tier2 = tier2[(s2 >= low) & (s2 < high)]
    return Prepared(df, y, tier1, tier2, val, test, scorer, tools)


def build_tier2(p: Prepared, cfg: dict):
    t2, arm = cfg["tier2"], cfg["tier2"]["arm"]
    if arm.startswith("llm_"):
        return InvestigationAgent(p.tools, t2["llm"], t2.get("base_url"), mode=arm[4:],
                                  max_tool_calls=t2["max_tool_calls"], k_samples=t2["k_samples"])
    rows = p.df.iloc[p.tier2_idx]
    if arm == "encoder":
        model = EncoderInvestigator(p.tools, t2["encoder"], epochs=t2["encoder_epochs"])
    elif arm == "gbdt":
        model = EvidenceGBDT(p.tools)
    else:
        raise ValueError(f"tier2.arm must be one of {ARMS}")
    return model.fit(rows, p.scorer.score(rows), p.y[p.tier2_idx])


def run(cfg: dict) -> dict:
    p = prepare(cfg)
    tier2 = build_tier2(p, cfg)
    c, t = cfg["costs"], p.df["TransactionDT"].to_numpy()
    months = month_of(p.df["TransactionDT"].iloc[p.test_idx]).to_numpy()
    run_ = run_protocol(p.scorer, tier2, p.df.iloc[p.val_idx], p.y[p.val_idx], p.df.iloc[p.test_idx],
                        p.y[p.test_idx], c["c_fn"], c["c_fp"], cfg["cascade"]["budget"], groups=months)

    # gate experiment 1: model family x feature set, same traces
    ablation = gate_ablation(run_)

    # gate experiment 2: refit monthly on the latest labelled window vs. fit once on val
    starts = [m * 30 * DAY for m in np.unique(months)]
    final = rolling_gate(run_, t[p.val_idx], t[p.test_idx], starts,
                         window=cfg["gate"]["rolling_window_days"] * DAY,
                         delay=cfg["tier2"]["label_delay_days"] * DAY)
    run_.rows.append(row("gated_rolling", run_.y_test, run_.test, final, c["c_fn"], c["c_fp"]))
    run_.rows += group_rows({"gated_rolling": final}, run_.y_test, run_.test.base, months, c["c_fn"], c["c_fp"])

    return {"operating_point": dict(zip(("threshold", "low", "high"),
                                        (run_.cascade.threshold, run_.cascade.low, run_.cascade.high))),
            "rows": run_.rows, "gate_ablation": ablation, "tier2_log": tier2.log,
            "sizes": {k: len(getattr(p, f"{k}_idx")) for k in ("tier1", "tier2", "val", "test")}}
