"""Evaluate trained actors on the test days.

    python -m evaluation.evaluate_rl --case C1 --model_dir pretrained/proposed/C1/seed1

Test day k is generated with np.random.seed(first_day + k); the paper uses 20261005-20261024.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from evaluation.ledger import evaluate
from evaluation.policy import DeterministicPolicy
from power_envs import CASES

FIRST_TEST_DAY = 20261005


def violation_hours(day, tol=1e-8):
    """Hours with a positive local or global cost (N_v)."""
    return sum((s["local_score"] > tol) or (s["global_score"] > tol) for s in day["steps"])


def summary(days):
    return dict(days=len(days),
                reward=float(np.mean([d["reward"] for d in days])),
                cost=float(np.mean([d["cost"] for d in days])),
                global_cost=float(np.mean([d["global_violation_score"] for d in days])),
                local_cost=float(np.mean([d["local_violation_score"] for d in days])),
                violation_hours=float(np.mean([violation_hours(d) for d in days])))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", choices=tuple(CASES), required=True)
    p.add_argument("--model_dir", type=Path, required=True)
    p.add_argument("--first_day", type=int, default=FIRST_TEST_DAY)
    p.add_argument("--days", type=int, default=20)
    p.add_argument("--label", default="proposed")
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    torch.set_num_threads(1)
    results = []
    for day in range(args.first_day, args.first_day + args.days):
        np.random.seed(day)
        env = CASES[args.case]()
        obs = env.reset()
        r = evaluate(args.case, day, args.label, env=env, obs=obs, controller=DeterministicPolicy(env, args.model_dir))
        results.append(r)
        print(f"{args.case} day={day} reward={r['reward']:.3f} cost={r['cost']:.2f} "
              f"C_g={r['global_violation_score']:.4f} C_l={r['local_violation_score']:.4f} N_v={violation_hours(r)}",
              flush=True)
    print(json.dumps(summary(results), indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
