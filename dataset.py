"""Demonstration data and behaviour cloning.

The expert (JacobianPolicy) is rolled out on training scenes; its observations and commands become a
supervised dataset. Training scenes use seeds below TRAIN_SEED_LIMIT and evaluation scenes use seeds
from EVAL_SEED_START, so evaluation is always on scenes the learner has not seen.
"""

from __future__ import annotations

from typing import Tuple

import mujoco
import numpy as np
from sklearn.neural_network import MLPRegressor

from .policies import JacobianPolicy, LearnedPolicy, bc_features
from .rollout import MAX_STEPS, _observation
from .scene import CTRL_LIMIT, make_model, sample_scene

TRAIN_SEED_LIMIT = 10_000
EVAL_SEED_START = 100_000


def collect_demos(n_scenes: int, seed: int = 0, steps: int = MAX_STEPS) -> Tuple[np.ndarray, np.ndarray]:
    """Roll out the expert and return (features, commands) from every step."""
    expert = JacobianPolicy()
    xs, ys = [], []
    for k in range(n_scenes):
        scene_seed = seed + k
        assert scene_seed < TRAIN_SEED_LIMIT, "training seeds must stay below TRAIN_SEED_LIMIT"
        scene = sample_scene(scene_seed)
        model = make_model(scene)
        data = mujoco.MjData(model)
        data.qpos[:] = scene.init_q
        mujoco.mj_forward(model, data)
        for _ in range(steps):
            obs = _observation(model, data, scene)
            u = np.clip(expert(obs), -CTRL_LIMIT, CTRL_LIMIT)
            xs.append(bc_features(obs))
            ys.append(u)
            data.ctrl[:] = u
            mujoco.mj_step(model, data)
    return np.array(xs), np.array(ys)


def train_bc(x: np.ndarray, y: np.ndarray, seed: int = 0, hidden=(64, 64)) -> LearnedPolicy:
    model = MLPRegressor(hidden_layer_sizes=hidden, max_iter=300, random_state=seed, early_stopping=True)
    model.fit(x, y)
    return LearnedPolicy(model)
