"""Behavioural + entity-sharing features (the graph signal, without a GNN).

Entity-sharing: for each transaction's card / email domain / device / address,
how many distinct accounts used that entity *before* this transaction.
Fraud rings show up as entities shared unusually widely.

Rule: every feature must be computable from data strictly before the
transaction's timestamp. Leakage here would invalidate the drift results.
"""
from __future__ import annotations

ENTITY_COLS = ["card1", "addr1", "P_emaildomain", "DeviceInfo"]
ACCOUNT_PROXY = ["card1", "card2", "addr1"]  # IEEE-CIS has no user id; this is a common proxy


def add_entity_sharing(df):
    """TODO: for each entity column, cumulative distinct-account count up to (not incl.) each row."""
    raise NotImplementedError


def add_velocity(df, windows_s=(3600, 86400)):
    """TODO: per account proxy, transaction count and amount sum in trailing windows."""
    raise NotImplementedError
