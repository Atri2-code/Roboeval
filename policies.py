"""Policies under evaluation. Each maps an observation dict to joint-velocity commands."""

from __future__ import annotations

import numpy as np

from .scene import CTRL_LIMIT, OBSTACLE_RADIUS


class RandomPolicy:
    """Sanity floor: uniform random commands."""

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs):
        return self.rng.uniform(-CTRL_LIMIT, CTRL_LIMIT, size=2)


class JacobianPolicy:
    """Damped-least-squares resolved-rate controller towards the target (the expert)."""

    def __init__(self, gain: float = 4.0, damping: float = 0.05):
        self.gain = gain
        self.damping = damping

    def task_velocity(self, obs):
        return self.gain * (obs["target"] - obs["ee"])

    def __call__(self, obs):
        j = obs["jac"]
        v = self.task_velocity(obs)
        jjt = j @ j.T + (self.damping ** 2) * np.eye(2)
        return j.T @ np.linalg.solve(jjt, v)


class AvoidPolicy(JacobianPolicy):
    """Jacobian controller plus a repulsive task-space velocity near the obstacle."""

    def __init__(self, gain: float = 4.0, damping: float = 0.05, margin: float = 0.18, strength: float = 1.5):
        super().__init__(gain, damping)
        self.margin = margin
        self.strength = strength

    def task_velocity(self, obs):
        v = super().task_velocity(obs)
        if "obstacle" in obs:
            away = obs["ee"] - obs["obstacle"]
            dist = np.linalg.norm(away) - OBSTACLE_RADIUS
            if dist < self.margin:
                v = v + self.strength * (self.margin - max(dist, 0.0)) / self.margin * away / (np.linalg.norm(away) + 1e-9)
        return v


class NoisyPolicy:
    """Wraps a policy and adds Gaussian command noise (a robustness probe)."""

    def __init__(self, base, sigma: float = 0.5, seed: int = 0):
        self.base = base
        self.sigma = sigma
        self.rng = np.random.default_rng(seed)

    def __call__(self, obs):
        return self.base(obs) + self.rng.normal(0.0, self.sigma, size=2)


def bc_features(obs) -> np.ndarray:
    """Features for the behaviour-cloned policy: joint angles and end-effector error."""
    err = obs["target"] - obs["ee"]
    return np.array([obs["q"][0], obs["q"][1], err[0], err[1]])


class LearnedPolicy:
    """Wraps a fitted scikit-learn regressor trained by behaviour cloning."""

    def __init__(self, model):
        self.model = model

    def __call__(self, obs):
        return self.model.predict(bc_features(obs)[None, :])[0]
