"""Behavioural + entity-sharing features (the graph signal, without a GNN).

Entity sharing: for each transaction's card / address / email domain / device,
how many distinct accounts used that entity *before* this transaction.
Fraud rings show up as entities shared unusually widely, unusually fast.

Rule: every feature uses only rows strictly earlier in time order. Labels are
never used as features (they arrive late, as chargebacks). Leakage here would
invalidate the drift results.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ENTITY_COLS = ["card1", "addr1", "P_emaildomain", "DeviceInfo"]
ACCOUNT_PROXY = ["card1", "card2", "addr1"]  # IEEE-CIS has no user id; this is a common proxy

# raw IEEE-CIS columns worth passing straight to LightGBM (kept if present)
BASE_COLS = (["TransactionAmt", "ProductCD", "card1", "card2", "card3", "card4", "card5", "card6",
              "addr1", "addr2", "dist1", "P_emaildomain", "R_emaildomain", "DeviceType", "DeviceInfo"]
             + [f"C{i}" for i in range(1, 15)] + [f"D{i}" for i in range(1, 16)]
             + [f"M{i}" for i in range(1, 10)])


def add_account(df: pd.DataFrame) -> pd.DataFrame:
    df["account"] = df[ACCOUNT_PROXY].astype(str).agg("_".join, axis=1)
    return df


def add_entity_sharing(df: pd.DataFrame) -> pd.DataFrame:
    """<entity>_n_accounts: distinct accounts seen with this entity value before this row."""
    for col in ENTITY_COLS:
        first = (~df.duplicated([col, "account"])).astype(int)  # first time this account uses it
        df[f"{col}_n_accounts"] = first.groupby(df[col]).cumsum() - first
    return df


def add_velocity(df: pd.DataFrame, windows_s=(3600, 86400)) -> pd.DataFrame:
    """Per account: count and amount sum of earlier tx in trailing windows, time since the previous tx, tenure."""
    acct = df["account"].astype("category").cat.codes.to_numpy(np.int64)
    t = df["TransactionDT"].to_numpy(np.int64)
    order = np.lexsort((t, acct))  # by account, then time
    # one sortable key per row; the account offset is wider than any window, so windows never cross accounts
    key = acct[order] * (t.max() + max(windows_s) + 1) + t[order]
    csum = np.r_[0.0, np.cumsum(df["TransactionAmt"].to_numpy(float)[order])]
    pos = np.arange(len(df))
    for w in windows_s:
        start = np.searchsorted(key, key - w, side="left")  # first earlier row inside the window
        n, amt = np.empty(len(df)), np.empty(len(df))
        n[order], amt[order] = pos - start, csum[pos] - csum[start]
        df[f"n_tx_{w}s"], df[f"amt_sum_{w}s"] = n, amt
    df["secs_since_prev"] = df.groupby("account")["TransactionDT"].diff()
    df["account_n_prev"] = df.groupby("account").cumcount()  # tenure, label-free
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """df must be sorted by TransactionDT. Returns the frame with features added."""
    df = add_velocity(add_entity_sharing(add_account(df.copy())))
    for col in df.columns.intersection(BASE_COLS):
        if not pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].astype("category")
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    engineered = [c for c in df.columns if c.endswith("_n_accounts") or c.startswith(("n_tx_", "amt_sum_"))]
    return [c for c in BASE_COLS if c in df.columns] + engineered + ["secs_since_prev", "account_n_prev"]
