import numpy as np

from roboeval.graders import SUCCESS_RADIUS, grade
from roboeval.rollout import Trajectory
from roboeval.scene import Scene

SCENE = Scene(seed=1, target=(0.5, 0.0), init_q=(0.0, 1.0))


def traj(ee, ctrl=None, hits=None):
    ee = np.array(ee, dtype=float)
    T = len(ee) - 1
    ctrl = np.zeros((T, 2)) if ctrl is None else np.array(ctrl, dtype=float)
    hits = np.zeros(T, dtype=bool) if hits is None else np.array(hits, dtype=bool)
    return Trajectory(SCENE, ee, ctrl, hits)


def test_straight_line_success_has_unit_efficiency():
    ee = [[x, 0.0] for x in np.linspace(0.0, 0.5, 11)]
    r = grade(traj(ee))
    assert r.success and r.steps_to_success == 10
    assert r.path_efficiency == 1.0


def test_detour_lowers_path_efficiency():
    ee = [[0.0, 0.0], [0.25, 0.25], [0.5, 0.0]]
    r = grade(traj(ee))
    assert r.success
    assert 0.65 < r.path_efficiency < 0.75     # 0.5 / (2 * 0.3536)


def test_failure_reports_final_distance_and_no_steps():
    ee = [[0.0, 0.0], [0.1, 0.0], [0.2, 0.0]]
    r = grade(traj(ee))
    assert not r.success and r.steps_to_success is None and r.path_efficiency is None
    assert abs(r.final_distance - 0.3) < 1e-9


def test_success_radius_boundary():
    inside = grade(traj([[0.0, 0.0], [0.5 - SUCCESS_RADIUS + 1e-6, 0.0]]))
    outside = grade(traj([[0.0, 0.0], [0.5 - SUCCESS_RADIUS - 1e-6, 0.0]]))
    assert inside.success and not outside.success


def test_effort_smoothness_and_collision_flags():
    ctrl = [[1.0, 0.0], [1.0, 0.0], [-1.0, 0.0]]
    ee = [[0.0, 0.0]] * 4
    r = grade(traj(ee, ctrl=ctrl, hits=[False, True, False]))
    assert abs(r.effort - 0.5) < 1e-9          # mean of squared commands over 3 steps and 2 joints
    assert abs(r.smoothness - (0.0 + 2.0) / 2 / 2) < 1e-9    # mean over 2 diffs and 2 joints
    assert r.collided
