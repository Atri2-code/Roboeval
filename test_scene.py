import numpy as np
import mujoco

from roboeval.scene import LINK_LENGTHS, OBSTACLE_RADIUS, build_xml, make_model, sample_scene


def test_scene_is_a_pure_function_of_seed():
    assert sample_scene(7) == sample_scene(7)
    assert sample_scene(7) != sample_scene(8)
    assert sample_scene(7, with_obstacle=True) == sample_scene(7, with_obstacle=True)


def test_targets_are_reachable():
    reach = sum(LINK_LENGTHS)
    for seed in range(200):
        scene = sample_scene(seed)
        assert 0.25 - 1e-9 <= np.hypot(*scene.target) <= 0.80 + 1e-9 < reach


def test_obstacle_sits_between_base_and_target_and_clear_of_target():
    for seed in range(200):
        scene = sample_scene(seed, with_obstacle=True)
        assert scene.has_obstacle
        d_target = np.linalg.norm(np.array(scene.obstacle) - np.array(scene.target))
        assert d_target > OBSTACLE_RADIUS + 0.05


def test_plain_scene_has_no_obstacle_geom():
    model = make_model(sample_scene(3))
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle") < 0
    model = make_model(sample_scene(3, with_obstacle=True))
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "obstacle") >= 0


def test_joint_limits_are_in_radians():
    # Regression test: MuJoCo reads angles as degrees unless the compiler is told otherwise.
    model = make_model(sample_scene(1))
    assert model.jnt_range[0, 1] > 3.0
    assert "angle=\"radian\"" in build_xml(sample_scene(1))
