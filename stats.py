"""Statistics for comparing policies: Wilson intervals, bootstrap CIs and paired differences."""

from __future__ import annotations

from typing import Tuple

import numpy as np


def wilson_interval(successes: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z ** 2 / n
    centre = (p + z ** 2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def bootstrap_ci(values, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0) -> Tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def paired_difference(candidate, baseline, n_boot: int = 2000, alpha: float = 0.05, seed: int = 0):
    """Mean of (candidate - baseline) over the same scenes, with a paired bootstrap interval."""
    diff = np.asarray(candidate, dtype=float) - np.asarray(baseline, dtype=float)
    low, high = bootstrap_ci(diff, n_boot=n_boot, alpha=alpha, seed=seed)
    return float(diff.mean()), low, high
