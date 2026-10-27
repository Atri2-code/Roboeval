"""Procedural scene generation: a planar 2-link arm, a reach target and an optional obstacle.

Every scene is a pure function of its integer seed, so any result can be reproduced and
two policies can be compared on exactly the same scenes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import mujoco
import numpy as np

LINK_LENGTHS = (0.5, 0.4)
CTRL_LIMIT = 3.0          # rad/s, joint velocity command limit
TIMESTEP = 0.01
OBSTACLE_RADIUS = 0.06


@dataclass(frozen=True)
class Scene:
    seed: int
    target: Tuple[float, float]
    init_q: Tuple[float, float]
    obstacle: Optional[Tuple[float, float]] = None   # centre (x, y); radius is OBSTACLE_RADIUS

    @property
    def has_obstacle(self) -> bool:
        return self.obstacle is not None


def sample_scene(seed: int, with_obstacle: bool = False) -> Scene:
    """Sample a reachable target and, optionally, an obstacle that sits near the straight path."""
    rng = np.random.default_rng(seed)
    reach_min, reach_max = 0.25, 0.80
    radius = rng.uniform(reach_min, reach_max)
    angle = rng.uniform(-np.pi, np.pi)
    target = (float(radius * np.cos(angle)), float(radius * np.sin(angle)))

    init_q = (float(rng.uniform(-0.5, 0.5)), float(rng.uniform(0.6, 1.4)))

    obstacle = None
    if with_obstacle:
        # Place the obstacle roughly halfway to the target with a small sideways offset.
        direction = np.array(target) / max(np.linalg.norm(target), 1e-9)
        normal = np.array([-direction[1], direction[0]])
        centre = 0.55 * np.array(target) + rng.uniform(-0.05, 0.05) * normal
        obstacle = (float(centre[0]), float(centre[1]))
    return Scene(seed=seed, target=target, init_q=init_q, obstacle=obstacle)


def build_xml(scene: Scene) -> str:
    l1, l2 = LINK_LENGTHS
    obstacle_xml = ""
    if scene.obstacle is not None:
        ox, oy = scene.obstacle
        obstacle_xml = (
            f'<geom name="obstacle" type="cylinder" pos="{ox:.5f} {oy:.5f} 0" '
            f'size="{OBSTACLE_RADIUS} 0.05" rgba="0.8 0.2 0.2 1" contype="1" conaffinity="1"/>'
        )
    tx, ty = scene.target
    return f"""
<mujoco model="reach">
  <compiler angle="radian"/>
  <option timestep="{TIMESTEP}" gravity="0 0 0" integrator="implicitfast"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1" contype="0" conaffinity="0"/>
    {obstacle_xml}
    <site name="target" pos="{tx:.5f} {ty:.5f} 0" size="0.02" rgba="0 1 0 0.5"/>
    <body name="link1" pos="0 0 0">
      <joint name="j1" type="hinge" axis="0 0 1" range="-3.1 3.1" damping="0.5"/>
      <geom name="l1" type="capsule" fromto="0 0 0 {l1} 0 0" size="0.03" contype="1" conaffinity="1"/>
      <body name="link2" pos="{l1} 0 0">
        <joint name="j2" type="hinge" axis="0 0 1" range="-2.8 2.8" damping="0.5"/>
        <geom name="l2" type="capsule" fromto="0 0 0 {l2} 0 0" size="0.025" contype="1" conaffinity="1"/>
        <site name="ee" pos="{l2} 0 0" size="0.015"/>
      </body>
    </body>
  </worldbody>
  <actuator>
    <velocity name="a1" joint="j1" kv="20" ctrlrange="-{CTRL_LIMIT} {CTRL_LIMIT}"/>
    <velocity name="a2" joint="j2" kv="20" ctrlrange="-{CTRL_LIMIT} {CTRL_LIMIT}"/>
  </actuator>
</mujoco>
"""


def make_model(scene: Scene) -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(build_xml(scene))
