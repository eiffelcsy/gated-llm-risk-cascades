"""IEEE-CIS Fraud Detection (Kaggle): ~590K e-commerce transactions, time-ordered.

TransactionDT is seconds from a reference point; splits.temporal uses it so
the last months are the "drifted" test set.

`synthetic_transactions` produces a small frame with the same core columns so
the whole track runs without the Kaggle download.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DAY = 86400


def load_ieee_cis(root: str | Path) -> pd.DataFrame:
    """Join train_transaction.csv and train_identity.csv on TransactionID, sorted by time."""
    root = Path(root)
    tx = pd.read_csv(root / "train_transaction.csv")
    idn = pd.read_csv(root / "train_identity.csv")
    df = tx.merge(idn, on="TransactionID", how="left")
    return df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)


def month_of(dt: pd.Series) -> pd.Series:
    """30-day bucket index, used to report results by month."""
    return (dt // (30 * DAY)).astype(int)


def synthetic_transactions(n_accounts: int = 4000, n_rings: int = 40, days: int = 182,
                           drift_day: int = 121, seed: int = 0) -> pd.DataFrame:
    """Honest accounts plus fraud rings, with the overlaps that make real fraud hard.

    - Rings share devices across many stolen cards; but some honest devices
      (offices, households) are shared too, and some rings spoof generic devices.
    - Rings starting before `drift_day` run bursty fraud; rings after it switch
      to a quiet style (1-2 honest-looking transactions per card) tier 1 never saw.
    - Honest customers join throughout the period, and 30% of fraud is account
      takeover on an established card, so tenure is not proof either way.
    - Some honest buyers also burst.
    """
    rng = np.random.default_rng(seed)
    domains = np.array(["gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "icloud.com", "aol.com"])
    generic = np.array(["Windows", "iOS Device", "MacOS", "Trident/7.0", "SM-G960U"])
    offices = [f"office{i}" for i in range(150)]
    products, p_products = list("WCRHS"), [.70, .10, .08, .07, .05]
    rows, honest_cards = [], []

    def add(t, amt, product, card1, card2, addr1, email, device, fraud):
        rows.extend(zip(t, amt, product, *[[x] * len(t) for x in (card1, card2, addr1, email, device, fraud)]))

    def card():
        return dict(card1=rng.integers(1000, 18000), card2=rng.integers(100, 600), addr1=rng.integers(100, 540))

    for _ in range(n_accounts):
        n = rng.poisson(12) + 1
        joined = rng.uniform(-days, days) * DAY  # half were already customers when the data starts
        t = rng.uniform(max(joined, 0), days * DAY, n)
        if rng.random() < 0.1:  # bursty honest buyer
            t[: n // 2] = t[0] + rng.uniform(0, 6 * 3600, n // 2)
        u = rng.random()
        device = rng.choice(generic) if u < 0.55 else rng.choice(offices) if u < 0.65 else f"dev{rng.integers(10**7)}"
        honest_cards.append(card())
        add(t, rng.lognormal(3.8, 1.0, n), rng.choice(products, n, p=p_products), **honest_cards[-1],
            email=rng.choice(domains, p=[.45, .2, .12, .1, .08, .05]), device=device, fraud=0)

    for r in range(n_rings):
        start = rng.uniform(0, days - 21) * DAY
        quiet = start >= drift_day * DAY
        spoofed = rng.random() < 0.3  # ring hides behind generic device strings
        devices = list(rng.choice(generic, 2)) if spoofed else [f"ring{r}_dev{j}" for j in range(rng.integers(1, 4))]
        email = rng.choice(["protonmail.com", "mail.ru", "anonymous.com", *domains[:3]])
        for _ in range(rng.integers(10, 40)):  # one stolen card each: new, or taken over from a customer
            t0 = start + rng.uniform(0, 21 * DAY)
            if quiet:
                n = rng.integers(1, 3)
                t, amt = t0 + rng.uniform(0, 5 * DAY, n), rng.lognormal(3.9, 0.8, n)
                product = rng.choice(products, n, p=p_products)
            else:
                n = rng.integers(2, 7)
                t, amt = t0 + rng.uniform(0, 12 * 3600, n), rng.lognormal(4.4, 0.9, n)
                product = rng.choice(["W", "C"], n)
            victim = honest_cards[rng.integers(len(honest_cards))] if rng.random() < 0.3 else card()
            add(t, amt, product, **victim, email=email, device=rng.choice(devices), fraud=1)

    df = pd.DataFrame(rows, columns=["TransactionDT", "TransactionAmt", "ProductCD", "card1", "card2",
                                     "addr1", "P_emaildomain", "DeviceInfo", "isFraud"])
    df = df[df.TransactionDT < days * DAY]
    df["TransactionDT"] = df.TransactionDT.astype(int) + DAY  # IEEE-CIS starts at 86400
    df = df.sort_values("TransactionDT", kind="stable").reset_index(drop=True)
    df.insert(0, "TransactionID", np.arange(len(df)) + 2_987_000)
    return df
