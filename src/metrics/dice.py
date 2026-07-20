"""Dice, with the empty-empty convention stated explicitly.

A prediction and a ground truth that are both empty agree perfectly, so Dice
is 1.0. Returning 0.0 there would penalise a correct negative, and returning
NaN would silently drop the case from any mean. Both failure modes are easy to
introduce and hard to notice, so the convention is asserted in the tests.
"""

from __future__ import annotations

import numpy as np


def dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice over boolean masks.

    Returns 1.0 when both masks are empty, 0.0 when exactly one is.
    """
    p = np.asarray(pred).astype(bool)
    g = np.asarray(gt).astype(bool)
    if p.shape != g.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs gt {g.shape}")

    n_p, n_g = int(p.sum()), int(g.sum())
    if n_p == 0 and n_g == 0:
        return 1.0
    inter = int(np.logical_and(p, g).sum())
    return 2.0 * inter / (n_p + n_g)


def confusion(pred: np.ndarray, gt: np.ndarray) -> dict[str, int]:
    """Voxel-level confusion counts, for reporting alongside Dice."""
    p = np.asarray(pred).astype(bool)
    g = np.asarray(gt).astype(bool)
    if p.shape != g.shape:
        raise ValueError(f"shape mismatch: pred {p.shape} vs gt {g.shape}")
    return {
        "tp": int(np.logical_and(p, g).sum()),
        "fp": int(np.logical_and(p, ~g).sum()),
        "fn": int(np.logical_and(~p, g).sum()),
        "tn": int(np.logical_and(~p, ~g).sum()),
    }
