"""Raking (iterative proportional fitting): reweight panel cells so the panel's demographic margins match
the population's. Used daily, because panel composition shifts every time a data source joins or leaves."""

from __future__ import annotations

import numpy as np


def rake(
    counts: np.ndarray, cell_levels: list[np.ndarray], targets: list[np.ndarray], max_iter: int = 50, tol: float = 1e-8
) -> np.ndarray:
    """Return a weight per cell such that weighted counts reproduce each dimension's target shares.

    counts       members per cell (cells may be empty)
    cell_levels  for each dimension, the category index of every cell
    targets      for each dimension, population share of every category (sums to 1)
    Categories with no panel members cannot be matched; their target share is redistributed.
    """
    counts = np.asarray(counts, float)
    w = np.ones_like(counts)
    total = counts.sum()
    if total <= 0:
        return w
    for _ in range(max_iter):
        worst = 0.0
        for levels, target in zip(cell_levels, targets, strict=True):
            weighted = np.bincount(levels, weights=w * counts, minlength=len(target))
            present = weighted > 0
            goal = np.where(present, target, 0.0)
            goal = goal / goal.sum() * (w * counts).sum()
            factor = np.ones(len(target))
            factor[present] = goal[present] / weighted[present]
            w *= factor[levels]
            worst = max(worst, float(np.abs(factor[present] - 1).max()))
        if worst < tol:
            break
    return w


def raked_shares(counts: np.ndarray, cell_levels: list[np.ndarray], targets: list[np.ndarray], **kw) -> np.ndarray:
    """Population share each cell represents after raking (sums to 1 over non-empty cells)."""
    w = rake(counts, cell_levels, targets, **kw)
    mass = w * np.asarray(counts, float)
    return mass / mass.sum() if mass.sum() > 0 else mass
