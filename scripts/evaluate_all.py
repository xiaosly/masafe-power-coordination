"""Evaluate the trained (or pretrained) policies on the 20 test days, optionally with the MPC baselines.

    python scripts/evaluate_all.py                 # all four methods in results/runs/
    python scripts/evaluate_all.py --pretrained    # pretrained/proposed/<case>/seed<k>
    python scripts/evaluate_all.py --mpc           # also decentralized and distributed MPC (Gurobi)

Results: results/evaluation/<case>/<method>/seed<k>.json and results/evaluation/<case>/<mpc>.json
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safe_marl.methods import METHODS


def evaluation_command(case, method, seed, models, output):
    return [sys.executable, "-m", "evaluation.evaluate_rl", "--case", case,
            "--label", method, "--model_dir", str(models), "--output", str(output)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", nargs="+", choices=("C1", "C2", "C3"), default=["C1", "C2", "C3"])
    p.add_argument("--methods", nargs="+", choices=METHODS)
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--pretrained", action="store_true")
    p.add_argument("--mpc", action="store_true")
    p.add_argument("--mpc-only", action="store_true")
    p.add_argument("--jobs", type=int, default=8)
    p.add_argument("--runs", type=Path, default=ROOT / "results/runs")
    p.add_argument("--output_root", type=Path, default=ROOT / "results/evaluation")
    args = p.parse_args()
    methods = [] if args.mpc_only else (args.methods or (["proposed"] if args.pretrained else list(METHODS)))
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    jobs = []
    for case in args.cases:
        for method in methods:
            for seed in args.seeds:
                models = (ROOT / "pretrained" / method / case / f"seed{seed}" if args.pretrained
                          else args.runs.resolve() / case / method / f"seed{seed}" / "models")
                out = args.output_root.resolve() / case / method / f"seed{seed}.json"
                jobs.append(evaluation_command(case, method, seed, models, out))
        if args.mpc or args.mpc_only:
            for method in ("dec_mpc", "dist_mpc"):
                out = args.output_root.resolve() / case / f"{method}.json"
                jobs.append([sys.executable, "-m", "baselines.run", "--case", case, "--method", method,
                             "--output", str(out)])

    def launch(command):
        code = subprocess.call(command, cwd=ROOT, env=env, stdout=subprocess.DEVNULL)
        print(f"exit {code}: {' '.join(command[2:])}", flush=True)
        return code

    with ThreadPoolExecutor(args.jobs) as pool:
        sys.exit(max(pool.map(launch, jobs), default=0))


if __name__ == "__main__":
    main()
