"""IEEE-CIS Fraud Detection (Kaggle): ~590K e-commerce transactions, time-ordered.

Use TransactionDT (seconds from a reference point) for splits.temporal:
the last months are the "drifted" test set.
"""
from __future__ import annotations

from pathlib import Path


def load_ieee_cis(root: str | Path):
    """Join train_transaction.csv and train_identity.csv on TransactionID.

    Returns a pandas DataFrame with columns incl. TransactionDT, isFraud.
    """
    import pandas as pd

    root = Path(root)
    tx = pd.read_csv(root / "train_transaction.csv")
    idn = pd.read_csv(root / "train_identity.csv")
    return tx.merge(idn, on="TransactionID", how="left").sort_values("TransactionDT")
