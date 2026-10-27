import numpy as np
import pytest

from roboeval.benchmark import regression_gate, summarise
from roboeval.graders import EpisodeResult
from roboeval.stats import bootstrap_ci, paired_difference, wilson_interval


def result(seed, success):
    return EpisodeResult(seed, success, 10 if success else None, 0.0 if success else 0.5,
                         0.9 if success else None, 1.0, 0.1, False)


def make(successes):
    return [result(i, s) for i, s in enumerate(successes)]


def test_wilson_interval_contains_estimate_and_stays_in_bounds():
    for k, n in [(0, 10), (5, 10), (10, 10), (84, 100)]:
        low, high = wilson_interval(k, n)
        assert 0.0 <= low <= k / n <= high <= 1.0
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_wilson_interval_narrows_with_more_data():
    w_small = np.subtract(*wilson_interval(8, 10)[::-1])
    w_large = np.subtract(*wilson_interval(800, 1000)[::-1])
    assert w_large < w_small


def test_bootstrap_ci_brackets_the_mean_and_is_reproducible():
    values = np.random.default_rng(0).normal(5, 1, 200)
    low, high = bootstrap_ci(values, seed=1)
    assert low < values.mean() < high
    assert bootstrap_ci(values, seed=1) == (low, high)


def test_paired_difference_of_identical_series_is_zero():
    mean, low, high = paired_difference([1, 0, 1, 1], [1, 0, 1, 1])
    assert mean == low == high == 0.0


def test_gate_passes_on_identical_policies():
    a = make([True] * 30 + [False] * 10)
    assert regression_gate(a, a).passed


def test_gate_fails_on_clear_regression():
    baseline = make([True] * 40)
    candidate = make([True] * 20 + [False] * 20)
    decision = regression_gate(candidate, baseline)
    assert not decision.passed and decision.ci_high < 0


def test_gate_tolerates_noise_on_a_tiny_suite():
    baseline = make([True, True, True, True, False])
    candidate = make([True, True, True, False, False])
    assert regression_gate(candidate, baseline).passed       # one scene difference is not significant


def test_gate_tolerance_can_absorb_a_real_but_acceptable_drop():
    baseline = make([True] * 100)
    candidate = make([True] * 90 + [False] * 10)
    assert not regression_gate(candidate, baseline, tolerance=0.0).passed
    assert regression_gate(candidate, baseline, tolerance=0.15).passed


def test_gate_requires_identical_scenes():
    with pytest.raises(AssertionError):
        regression_gate(make([True, True]), [result(5, True), result(6, True)])


def test_summarise_reports_rates():
    s = summarise(make([True, True, False, False]))
    assert s["success_rate"] == 0.5 and s["episodes"] == 4
    assert s["success_ci_low"] < 0.5 < s["success_ci_high"]
    assert s["median_steps_to_success"] == 10
