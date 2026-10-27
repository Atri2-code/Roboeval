"""Command line: run the benchmark, write results, and exit non-zero on a regression.

    python -m roboeval.cli --episodes 100 --suite plain
    python -m roboeval.cli --episodes 100 --suite obstacle --candidate avoid --baseline jacobian
"""

from __future__ import annotations

import argparse
import json
import sys

from .benchmark import eval_scenes, regression_gate, run_policy, summarise
from .dataset import collect_demos, train_bc
from .policies import AvoidPolicy, JacobianPolicy, NoisyPolicy, RandomPolicy


def build_policies(train_scenes: int, seed: int):
    x, y = collect_demos(train_scenes, seed=seed)
    bc = train_bc(x, y, seed=seed)
    return {
        "random": lambda s: RandomPolicy(seed=s),
        "jacobian": lambda s: JacobianPolicy(),
        "jacobian_noisy": lambda s: NoisyPolicy(JacobianPolicy(), sigma=1.0, seed=s),
        "avoid": lambda s: AvoidPolicy(),
        "behaviour_cloned": lambda s: bc,
    }, len(x)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=int, default=100)
    ap.add_argument("--suite", choices=["plain", "obstacle"], default="plain")
    ap.add_argument("--train-scenes", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--candidate", default="behaviour_cloned")
    ap.add_argument("--baseline", default="jacobian")
    ap.add_argument("--tolerance", type=float, default=0.0)
    ap.add_argument("--output", default=None, help="write JSON results here")
    args = ap.parse_args(argv)

    policies, n_samples = build_policies(args.train_scenes, args.seed)
    scenes = eval_scenes(args.episodes, with_obstacle=(args.suite == "obstacle"))

    results = {name: run_policy(make, scenes) for name, make in policies.items()}
    summaries = {name: summarise(r) for name, r in results.items()}

    print(f"suite={args.suite} episodes={args.episodes} behaviour-cloning samples={n_samples}")
    header = f"{'policy':<18}{'success':>9}{'95% CI':>16}{'steps':>8}{'path eff':>10}{'collide':>9}"
    print(header)
    for name, s in summaries.items():
        steps = "-" if s["median_steps_to_success"] is None else f"{s['median_steps_to_success']:.0f}"
        eff = "-" if s["mean_path_efficiency"] is None else f"{s['mean_path_efficiency']:.2f}"
        ci = f"[{s['success_ci_low']:.2f}, {s['success_ci_high']:.2f}]"
        print(f"{name:<18}{s['success_rate']:>9.2f}{ci:>16}{steps:>8}{eff:>10}{s['collision_rate']:>9.2f}")

    decision = regression_gate(results[args.candidate], results[args.baseline], tolerance=args.tolerance, seed=args.seed)
    print(f"\ngate ({args.candidate} vs {args.baseline}): {'PASS' if decision.passed else 'FAIL'}: {decision.reason}")

    if args.output:
        payload = {"suite": args.suite, "episodes": args.episodes, "summaries": summaries,
                   "gate": {"candidate": args.candidate, "baseline": args.baseline, **decision.__dict__}}
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
    return 0 if decision.passed else 1


if __name__ == "__main__":
    sys.exit(main())
