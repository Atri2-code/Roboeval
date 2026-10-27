import numpy as np
import pytest

from roboeval.benchmark import eval_scenes, run_policy, summarise
from roboeval.cli import main
from roboeval.dataset import EVAL_SEED_START, TRAIN_SEED_LIMIT, collect_demos, train_bc
from roboeval.policies import JacobianPolicy, NoisyPolicy, RandomPolicy
from roboeval.rollout import MAX_STEPS, rollout
from roboeval.scene import sample_scene


def test_rollout_shapes():
    traj = rollout(sample_scene(5), JacobianPolicy(), max_steps=50)
    assert traj.ee.shape == (51, 2) and traj.ctrl.shape == (50, 2) and traj.collisions.shape == (50,)


def test_rollout_is_deterministic():
    a = rollout(sample_scene(5), JacobianPolicy(), max_steps=80)
    b = rollout(sample_scene(5), JacobianPolicy(), max_steps=80)
    np.testing.assert_array_equal(a.ee, b.ee)


def test_expert_beats_random_by_a_wide_margin():
    scenes = eval_scenes(12)
    expert = summarise(run_policy(lambda s: JacobianPolicy(), scenes))["success_rate"]
    random = summarise(run_policy(lambda s: RandomPolicy(seed=s), scenes))["success_rate"]
    assert expert >= 0.6 and random == 0.0


def test_noisy_policy_is_reproducible_per_seed():
    obs = {"target": np.array([0.3, 0.2]), "ee": np.zeros(2), "jac": np.eye(2)}
    a = NoisyPolicy(JacobianPolicy(), sigma=0.5, seed=3)(obs)
    b = NoisyPolicy(JacobianPolicy(), sigma=0.5, seed=3)(obs)
    np.testing.assert_array_equal(a, b)


def test_train_and_eval_seed_ranges_do_not_overlap():
    assert TRAIN_SEED_LIMIT <= EVAL_SEED_START
    with pytest.raises(AssertionError):
        collect_demos(2, seed=TRAIN_SEED_LIMIT - 1)


def test_demo_dataset_shape_and_behaviour_cloning_fits():
    x, y = collect_demos(3, seed=0)
    assert x.shape == (3 * MAX_STEPS, 4) and y.shape == (3 * MAX_STEPS, 2)
    policy = train_bc(x, y, seed=0, hidden=(8,))
    obs = {"q": np.array([0.1, 1.0]), "ee": np.array([0.4, 0.2]), "target": np.array([0.3, 0.3])}
    assert policy(obs).shape == (2,)


def test_cli_runs_end_to_end_and_writes_results(tmp_path):
    out = tmp_path / "r.json"
    code = main(["--episodes", "4", "--train-scenes", "2", "--output", str(out),
                 "--candidate", "jacobian", "--baseline", "jacobian"])
    assert code == 0 and out.exists()
