# roboeval-mujoco

An evaluation harness for robot-policy rollouts in MuJoCo. It generates scenes, rolls policies out in simulation, grades each episode, compares policies with confidence intervals on identical scenes, and fails a build when a change is measurably worse. The idea: if you cannot measure a change, it does not ship.

The robot is deliberately small (a planar 2-link arm reaching a target, with an optional obstacle), so the whole loop runs on a laptop CPU in about a minute. The point is the evaluation machinery, which is what carries over to larger stacks.

## What it does

| Stage | Where | What happens |
|---|---|---|
| Scene generation | `roboeval/scene.py` | A scene is a pure function of an integer seed: random reachable target, start pose and, in the obstacle suite, a cylinder near the straight path. Same seed, same scene, always. |
| Simulated rollouts | `roboeval/rollout.py` | 300 control steps in MuJoCo per episode, recording end-effector path, commands and obstacle contacts. |
| Policies | `roboeval/policies.py` | Random, a damped-least-squares Jacobian controller (the expert), the same with command noise, the same with an obstacle-repulsion term, and a behaviour-cloned neural policy. |
| Datasets and training | `roboeval/dataset.py` | Expert demonstrations on training scenes (seeds below 10,000) train an MLP. Evaluation scenes start at seed 100,000, so the learner is always tested on scenes it has not seen. |
| Graders | `roboeval/graders.py` | Success (end-effector within 2 cm), steps to success, path efficiency, command effort, command smoothness, collision. |
| Statistics | `roboeval/stats.py` | Wilson intervals for success rates, bootstrap intervals for continuous metrics, paired bootstrap for policy-versus-policy differences. |
| Regression gate | `roboeval/benchmark.py` | Fails only if the candidate is significantly worse than the baseline on the same scenes, so noise on a small suite does not block a change but a clear drop does. Exit code 1 on failure. |

## Run it

```bash
pip install -r requirements.txt
pytest -q                                              # 27 tests, about 3 seconds
python -m roboeval.cli --episodes 100 --suite plain --train-scenes 150
python -m roboeval.cli --episodes 100 --suite obstacle --candidate avoid --baseline jacobian
```

Results are deterministic: rerunning the plain suite reproduced its numbers exactly. Raw outputs are in `results/`.

## Results (100 evaluation scenes per suite, seed 0)

**Plain reaching**

| Policy | Success | 95% CI | Median steps | Path efficiency |
|---|---|---|---|---|
| random | 0.00 | 0.00 to 0.04 | n/a | n/a |
| jacobian (expert) | 0.84 | 0.76 to 0.90 | 104 | 0.92 |
| jacobian with command noise | 0.85 | 0.77 to 0.91 | 124 | 0.80 |
| jacobian with obstacle repulsion | 0.84 | 0.76 to 0.90 | 104 | 0.92 |
| behaviour-cloned MLP | 0.80 | 0.71 to 0.87 | 108 | 0.92 |

Gate, behaviour-cloned versus expert: **FAIL**. Success changes by -0.040 (95% CI -0.080 to -0.010). The learned policy copies the expert's path quality but fails on a few more scenes, and the harness says so with a margin rather than a hunch. Noise barely changes success but makes paths about 13% less efficient (0.92 to 0.80) and slower.

**Obstacle suite**

| Policy | Success | 95% CI | Collision rate |
|---|---|---|---|
| random | 0.00 | 0.00 to 0.04 | 0.22 |
| jacobian | 0.48 | 0.38 to 0.58 | 0.61 |
| jacobian with command noise | 0.50 | 0.40 to 0.60 | 0.64 |
| jacobian with obstacle repulsion | 0.28 | 0.20 to 0.37 | 0.58 |
| behaviour-cloned MLP | 0.48 | 0.38 to 0.58 | 0.61 |

Gate, repulsion versus plain Jacobian: **FAIL**. Success changes by -0.200 (95% CI -0.280 to -0.130).

This is the result the harness exists to produce. Adding an obstacle-repulsion term looks like an improvement and is not: it pushes only the end-effector away from the obstacle, so the links still hit it (collision rate 0.58 against 0.61), and the push drags the arm into poses it cannot leave, so success drops from 0.48 to 0.28. A change that looked sensible without measurement would have shipped.

## Known limitations

- A planar 2-link arm, not a manipulation or locomotion benchmark. It is not Isaac Lab, ManiSkill or LeRobot, and no claim is made about those stacks.
- The expert fails on about 16% of plain scenes. In a 40-scene check the failed episodes stalled with the end-effector short of the target, which looks like joint limits and singular poses. I have not diagnosed this further.
- Success does not penalise collisions; collisions are reported separately.
- The behaviour-cloned policy is a small scikit-learn MLP trained on 150 expert scenes (45,000 samples). It is not tuned.
- Policies see exact joint state and target position; there is no vision or sensor noise.
- Explanations for why the repulsion policy fails are hypotheses from the numbers, not separately tested.

## Next steps

- Link-aware repulsion (use the elbow position, not just the end-effector) and re-run the gate.
- Report "safe success" (reached without collision) as a headline metric.
- Diagnose the expert's failures by plotting stalled poses against joint limits.
- Swap in a gymnasium-style environment so the same graders and gate can run against Isaac Lab or LeRobot tasks.

## Layout

```
roboeval/   scene, rollout, policies, graders, stats, dataset, benchmark, cli
tests/      27 tests: scene determinism, graders, statistics, gate logic, rollouts, dataset, CLI
results/    plain.json and obstacle.json from the runs above
```

## Author

Atrija Haldar, MSc Engineering, Technology and Business Management, University of Leeds. [LinkedIn](https://www.linkedin.com/in/atrija-haldar-196a3b221)
