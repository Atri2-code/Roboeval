"""Run policies over a fixed set of scenes, summarise them, and gate on regressions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np

from .dataset import EVAL_SEED_START
from .graders import EpisodeResult, grade
from .rollout import rollout
from .scene import sample_scene
from .stats import bootstrap_ci, paired_difference, wilson_interval


def eval_scenes(n: int, with_obstacle: bool = False, start: int = EVAL_SEED_START):
    return [sample_scene(start + i, with_obstacle=with_obstacle) for i in range(n)]


def run_policy(make_policy: Callable[[int], Callable], scenes) -> List[EpisodeResult]:
    """make_policy(seed) builds a fresh policy per episode so stochastic policies are reproducible."""
    return [grade(rollout(scene, make_policy(scene.seed))) for scene in scenes]


def summarise(results: List[EpisodeResult]) -> Dict[str, float]:
    n = len(results)
    successes = sum(r.success for r in results)
    low, high = wilson_interval(successes, n)
    steps = [r.steps_to_success for r in results if r.success]
    eff = [r.path_efficiency for r in results if r.path_efficiency is not None]
    out = {
        "episodes": n,
        "success_rate": successes / n,
        "success_ci_low": low,
        "success_ci_high": high,
        "median_steps_to_success": float(np.median(steps)) if steps else None,
        "mean_path_efficiency": float(np.mean(eff)) if eff else None,
        "mean_effort": float(np.mean([r.effort for r in results])),
        "mean_smoothness": float(np.mean([r.smoothness for r in results])),
        "collision_rate": float(np.mean([r.collided for r in results])),
    }
    return out


@dataclass(frozen=True)
class GateDecision:
    passed: bool
    mean_difference: float
    ci_low: float
    ci_high: float
    reason: str


def regression_gate(candidate: List[EpisodeResult], baseline: List[EpisodeResult],
                    tolerance: float = 0.0, seed: int = 0) -> GateDecision:
    """Fail only when the candidate is significantly worse than the baseline on the same scenes.

    The comparison is on per-scene success (1 or 0) with a paired bootstrap interval. The gate fails when
    the whole interval lies below -tolerance, so noise on a small suite does not block a change, but a
    clear drop does.
    """
    assert [r.seed for r in candidate] == [r.seed for r in baseline], "candidate and baseline must use identical scenes"
    c = [float(r.success) for r in candidate]
    b = [float(r.success) for r in baseline]
    diff, low, high = paired_difference(c, b, seed=seed)
    if high < -tolerance:
        return GateDecision(False, diff, low, high,
                            f"candidate is worse than baseline: success change {diff:+.3f} (95% CI {low:+.3f} to {high:+.3f})")
    return GateDecision(True, diff, low, high,
                        f"no significant regression: success change {diff:+.3f} (95% CI {low:+.3f} to {high:+.3f})")
