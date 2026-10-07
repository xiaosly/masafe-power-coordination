"""Run decentralized or distributed MPC on the test days.

    python -m baselines.run --case C1 --method dist_mpc --output results/evaluation/C1/dist_mpc.json

dec_mpc   each agent optimizes its own model for the remaining day; no communication
dist_mpc  the same models coordinated by ADMM; one communication round per iteration
Requires Gurobi (gurobipy).
"""
import argparse
import json
import logging
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from baselines.admm import ADMMPlanner
from evaluation.evaluate_rl import FIRST_TEST_DAY, summary, violation_hours
from evaluation.ledger import evaluate
from power_envs import CASES

# C2: relaxed battery modes, over-relaxation, fixed penalties; C1, C3: residual balancing
ADMM_SETTINGS = {"C2": dict(max_iter=500, relax=True, adapt_iters=0, alpha=1.6,
                            rho0={"p": 100.0, "q": 100.0, "v": 1e4}),
                 "C1": dict(max_iter=300, relax=False, adapt_iters=50, alpha=1.0),
                 "C3": dict(max_iter=300, relax=False, adapt_iters=50, alpha=1.0, eps_abs=1e-3, eps_rel=1e-4)}
logging.getLogger("pandapower").setLevel(logging.ERROR)


def controller_for(case, method):
    if method == "dist_mpc":
        return ADMMPlanner(case, **ADMM_SETTINGS[case])
    return ADMMPlanner(case, **{**ADMM_SETTINGS[case], "max_iter": 0})


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--case", choices=tuple(CASES), required=True)
    p.add_argument("--method", choices=("dec_mpc", "dist_mpc"), required=True)
    p.add_argument("--first_day", type=int, default=FIRST_TEST_DAY)
    p.add_argument("--days", type=int, default=20)
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    results = []
    for day in range(args.first_day, args.first_day + args.days):
        np.random.seed(day)
        env = CASES[args.case]()
        obs = env.reset()
        r = evaluate(args.case, day, args.method, env=env, obs=obs, controller=controller_for(args.case, args.method))
        rounds = [s["diagnostics"].get("communication_rounds", 0) for s in r["steps"]]
        seconds = [s["diagnostics"]["admm_sequential_s"] for s in r["steps"]]
        r["mean_rounds_per_decision"] = float(np.mean(rounds))
        r["mean_seconds_per_decision"] = float(np.mean(seconds))
        results.append(r)
        print(f"{args.case} {args.method} day={day} reward={r['reward']:.3f} cost={r['cost']:.2f} "
              f"C_g={r['global_violation_score']:.4f} C_l={r['local_violation_score']:.4f} N_v={violation_hours(r)} "
              f"rounds={np.mean(rounds):.1f} s/decision={np.mean(seconds):.3f}", flush=True)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(results, indent=1))
    print(json.dumps(summary(results), indent=2))


if __name__ == "__main__":
    main()
