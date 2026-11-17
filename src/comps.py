"""
Phase 1.4 — the naive comp tool (the thing to beat).

Plain Euclidean nearest neighbors in raw feature space, implemented from scratch
in NumPy (no sklearn — we want to understand the distance we're computing).

This is deliberately naive. It has two known weaknesses we will diagnose:
  1. On UN-standardized features, whichever column has the biggest raw numbers
     dominates the distance (a passing count in the dozens drowns out a 0–1 rate).
  2. Even standardized, CORRELATED features double-count: if five columns all
     measure "bigness," distance is secretly weighted 5x toward bigness.
Fixing (2) is exactly what PCA (Phase 2) does.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def nearest(
    name: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    k: int = 5,
    name_col: str = "PLAYER_NAME",
) -> pd.DataFrame:
    """
    Return the k nearest players to `name` by Euclidean distance over
    `feature_cols`. `df` carries identity columns + the feature columns.
    """
    X = df[feature_cols].to_numpy(dtype=float)

    matches = df.index[df[name_col].str.contains(name, case=False, na=False)]
    if len(matches) == 0:
        raise ValueError(f"no player matching {name!r}")
    qi = matches[0]
    q = X[df.index.get_loc(qi)]

    # Euclidean distance from the query row to every row: ||X - q||.
    dists = np.sqrt(((X - q) ** 2).sum(axis=1))

    out = df[[name_col]].copy()
    out["distance"] = dists
    out = out.drop(index=qi)                       # exclude the player themselves
    return out.nsmallest(k, "distance").reset_index(drop=True)
