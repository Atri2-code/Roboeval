"""Run a policy in a scene and record what happened."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import mujoco
import numpy as np

from .scene import CTRL_LIMIT, Scene, make_model

MAX_STEPS = 300
Policy = Callable[[Dict[str, np.ndarray]], np.ndarray]


@dataclass
class Trajectory:
    scene: Scene
    ee: np.ndarray            # (T+1, 2) end-effector positions, including the start
    ctrl: np.ndarray          # (T, 2) commands applied
    collisions: np.ndarray    # (T,) bool, arm touching the obstacle at each step


def _observation(model, data, scene: Scene) -> Dict[str, np.ndarray]:
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")
    jacp = np.zeros((3, model.nv))
    mujoco.mj_jacSite(model, data, jacp, None, ee_id)
    obs = {
        "q": data.qpos.copy(),
        "qvel": data.qvel.copy(),
        "ee": data.site_xpos[ee_id][:2].copy(),
        "target": np.array(scene.target),
        "jac": jacp[:2, :].copy(),
    }
    if scene.obstacle is not None:
        obs["obstacle"] = np.array(scene.obstacle)
    return obs


def _touching_obstacle(model, data) -> bool:
    obstacle = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle")
    if obstacle < 0:
        return False
    for i in range(data.ncon):
        c = data.contact[i]
        if obstacle in (c.geom1, c.geom2):
            return True
    return False


def rollout(scene: Scene, policy: Policy, max_steps: int = MAX_STEPS) -> Trajectory:
    model = make_model(scene)
    data = mujoco.MjData(model)
    data.qpos[:] = scene.init_q
    mujoco.mj_forward(model, data)

    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")
    ees = [data.site_xpos[ee_id][:2].copy()]
    ctrls, hits = [], []
    for _ in range(max_steps):
        obs = _observation(model, data, scene)
        u = np.clip(np.asarray(policy(obs), dtype=float), -CTRL_LIMIT, CTRL_LIMIT)
        data.ctrl[:] = u
        mujoco.mj_step(model, data)
        ees.append(data.site_xpos[ee_id][:2].copy())
        ctrls.append(u)
        hits.append(_touching_obstacle(model, data))
    return Trajectory(scene, np.array(ees), np.array(ctrls), np.array(hits, dtype=bool))
