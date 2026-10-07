"""Train all four RL methods on all cases and seeds.

    python scripts/train_all.py [--cases C1 C2 C3] [--seeds 1 2 3 4 5] [--jobs 4]

Each run uses one learner and 4 rollout processes; --jobs runs are executed in parallel.
Logs: results/logs/train_<batch>/<case>_<method>_seed<k>.log
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safe_marl.methods import METHODS, TRAIN_MODULES


def training_command(case, method, seed, output_root, steps=480096, workers=4):
    return [sys.executable, "-m", TRAIN_MODULES[method], "--case", case, "--method", method,
            "--seed", str(seed), "--output_dir", str(output_root / case / method / f"seed{seed}"),
            "--num_env_steps", str(steps), "--n_rollout_threads", str(workers)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cases", nargs="+", choices=("C1", "C2", "C3"), default=["C1", "C2", "C3"])
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--output_root", type=Path, default=ROOT / "results/runs")
    p.add_argument("--num_env_steps", type=int, default=480096)
    p.add_argument("--n_rollout_threads", type=int, default=4)
    args = p.parse_args()
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    jobs = [(case, method, seed) for seed in args.seeds for case in args.cases for method in args.methods]
    output_root = args.output_root.resolve()
    (ROOT / "results/logs").mkdir(parents=True, exist_ok=True)
    log_dir = Path(tempfile.mkdtemp(prefix="train_", dir=ROOT / "results/logs"))

    def launch(job):
        case, method, seed = job
        command = ["nice", "-n", "10", *training_command(case, method, seed, output_root,
                                                         args.num_env_steps, args.n_rollout_threads)]
        with open(log_dir / f"{case}_{method}_seed{seed}.log", "w") as log:
            code = subprocess.call(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        print(f"{case} {method} seed {seed}: exit {code}", flush=True)
        return code

    with ThreadPoolExecutor(args.jobs) as pool:
        sys.exit(max(pool.map(launch, jobs), default=0))


if __name__ == "__main__":
    main()
