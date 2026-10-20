import numpy as np
import pandas as pd
import pytest

pytest.importorskip("lightgbm")

from riskcascade.transactions.data import DAY, synthetic_transactions  # noqa: E402
from riskcascade.transactions.features import build_features  # noqa: E402
from riskcascade.transactions.posttrain import cost_reward  # noqa: E402


def tiny_frame():
    """Two accounts sharing one device, hand-checkable."""
    return pd.DataFrame({
        "TransactionID": [1, 2, 3, 4], "TransactionDT": [0, 100, 200, 5000],
        "TransactionAmt": [10.0, 20.0, 30.0, 40.0], "ProductCD": ["W"] * 4,
        "card1": [1, 1, 2, 1], "card2": [1, 1, 1, 1], "addr1": [1, 1, 1, 1],
        "P_emaildomain": ["a"] * 4, "DeviceInfo": ["d"] * 4, "isFraud": [0, 0, 1, 0]})


def test_features_use_only_the_past():
    f = build_features(tiny_frame())
    # device "d": no account before row 0, account A before row 1, A and B before row 3
    assert f["DeviceInfo_n_accounts"].tolist() == [0, 1, 1, 2]
    # account A (card1=1): rows 0, 1, 3. Velocity excludes the current row.
    assert f["n_tx_3600s"].tolist() == [0, 1, 0, 0]
    assert f["amt_sum_86400s"].tolist() == [0, 10, 0, 30]
    assert f["account_n_prev"].tolist() == [0, 1, 0, 2]


def test_tools_hide_future_and_late_labels():
    from riskcascade.transactions.agent import CaseTools
    from riskcascade.transactions.features import feature_columns
    from riskcascade.transactions.scorer import LgbmScorer

    df = build_features(synthetic_transactions(n_accounts=300, n_rings=4, days=60, drift_day=40))
    scorer = LgbmScorer(feature_columns(df), n_estimators=20).fit(df)
    tools = CaseTools(df, scorer, df.iloc[:500], label_delay_days=14)
    fraud = df[df.isFraud == 1]
    row = fraud.iloc[[len(fraud) // 2]]
    t, dev = row["TransactionDT"].iloc[0], row["DeviceInfo"].iloc[0]
    h = tools.entity_history(row, "DeviceInfo")
    past = df[(df.DeviceInfo == dev) & (df.TransactionDT < t)]
    assert h["past_transactions"] == len(past)
    assert h["known_fraud_transactions"] == int(past[past.TransactionDT < t - 14 * DAY].isFraud.sum())


def test_cost_reward_matches_decision_theory():
    assert cost_reward('{"verdict": 1}', 1, 10, 1) == 10
    assert cost_reward('{"verdict": 0}', 1, 10, 1) == -10
    assert cost_reward('{"verdict": 1}', 0, 10, 1) == -1
    assert cost_reward('{"verdict": -1}', 1, 10, 1) == 0
    assert cost_reward("no json here", 0, 10, 1) == -1


def test_pipeline_runs_with_gbdt_arm(monkeypatch):
    import yaml

    from riskcascade.transactions import pipeline

    monkeypatch.setattr(pipeline, "synthetic_transactions",
                        lambda: synthetic_transactions(n_accounts=1500, n_rings=15))
    cfg = yaml.safe_load(open("configs/transactions.yaml"))
    cfg["tier1"]["n_estimators"] = 100
    out = pipeline.run(cfg)
    policies = {r["policy"] for r in out["rows"]}
    assert {"tier1", "always", "gated", "oracle", "gated_rolling"} <= policies
    assert len(out["gate_ablation"]) == 9
    assert all(np.isfinite(r["cost_per_item"]) for r in out["rows"])
