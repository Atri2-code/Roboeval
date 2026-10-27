"""Graders: turn a trajectory into measurable outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .rollout import Trajectory
from .scene import TIMESTEP

SUCCESS_RADIUS = 0.02   # metres


@dataclass(frozen=True)
class EpisodeResult:
    seed: int
    success: bool
    steps_to_success: Optional[int]
    final_distance: float
    path_efficiency: Optional[float]   # straight-line distance / path length, successful episodes only
    effort: float                      # mean squared command
    smoothness: float                  # mean absolute change in command between steps (lower is smoother)
    collided: bool


def grade(traj: Trajectory) -> EpisodeResult:
    target = np.array(traj.scene.target)
    dist = np.linalg.norm(traj.ee - target, axis=1)
    reached = np.where(dist <= SUCCESS_RADIUS)[0]
    success = len(reached) > 0
    steps = int(reached[0]) if success else None

    efficiency = None
    if success:
        end = reached[0] + 1
        path = np.linalg.norm(np.diff(traj.ee[:end], axis=0), axis=1).sum()
        straight = np.linalg.norm(traj.ee[0] - target)
        efficiency = float(min(1.0, straight / path)) if path > 1e-9 else 1.0

    return EpisodeResult(
        seed=traj.scene.seed,
        success=success,
        steps_to_success=steps,
        final_distance=float(dist[-1]),
        path_efficiency=efficiency,
        effort=float(np.mean(traj.ctrl ** 2)),
        smoothness=float(np.mean(np.abs(np.diff(traj.ctrl, axis=0)))) if len(traj.ctrl) > 1 else 0.0,
        collided=bool(traj.collisions.any()),
    )


def seconds(steps: Optional[int]) -> Optional[float]:
    return None if steps is None else steps * TIMESTEP
