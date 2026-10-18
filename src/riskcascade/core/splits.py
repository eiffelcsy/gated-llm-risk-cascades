"""The shared "unseen threat" protocol.

content track:      hold out whole attack families (e.g. train without
                    'low-contrast', test on it).
transactions track: train on early months, test on later ones.
Both return index arrays (train, val, test); val is used to fit the gate.
"""
from __future__ import annotations

import numpy as np


def held_out_family(families: np.ndarray, test_families: list[str], val_frac: float = 0.2,
                    benign_label: str = "benign", seed: int = 0):
    """Benign items are split randomly; listed attack families go only to test."""
    fam = np.asarray(families)
    rng = np.random.default_rng(seed)
    is_test_fam = np.isin(fam, test_families)
    benign = np.flatnonzero(fam == benign_label)
    seen = np.flatnonzero(~is_test_fam & (fam != benign_label))
    rng.shuffle(benign); rng.shuffle(seen)

    b_test, b_rest = np.array_split(benign, [len(benign) // 3])
    n_val_b = int(len(b_rest) * val_frac)
    n_val_s = int(len(seen) * val_frac)
    val = np.concatenate([b_rest[:n_val_b], seen[:n_val_s]])
    train = np.concatenate([b_rest[n_val_b:], seen[n_val_s:]])
    test = np.concatenate([b_test, np.flatnonzero(is_test_fam)])
    return train, val, test


def temporal(timestamps: np.ndarray, train_end: float, val_end: float):
    """train: t < train_end; val: train_end <= t < val_end; test: t >= val_end."""
    t = np.asarray(timestamps)
    if not train_end < val_end:
        raise ValueError("need train_end < val_end")
    return (np.flatnonzero(t < train_end),
            np.flatnonzero((t >= train_end) & (t < val_end)),
            np.flatnonzero(t >= val_end))
