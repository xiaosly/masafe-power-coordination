"""Plot all four RL methods, with mean +/- one standard error across training seeds.

    python scripts/plot_training_curves.py          # new runs in results/runs/
    python scripts/plot_training_curves.py --paper  # archived runs used for Figure 2

Each seed is smoothed over 201 episodes. PF uses its unpenalized environment reward.
"""
import argparse
import csv
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
from scipy.io import loadmat

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from safe_marl.methods import LABELS, METHODS

CASES = ("C1", "C2", "C3")
PANELS = (("reward", "Reward $R$"), ("global_cost", "Global cost $C_g$"), ("local_cost", "Local cost $C_l$"))
PAPER_EVENTS = (("C1", "global_cost", "unified", 18684, "conflict"),
                ("C2", "global_cost", "unified", 8971, "conflict"))


def moving_average(x, window):
    half = window // 2
    csum = np.concatenate([[0.0], np.cumsum(x)])
    idx = np.arange(len(x))
    lo, hi = np.clip(idx - half, 0, len(x)), np.clip(idx + half + 1, 0, len(x))
    return (csum[hi] - csum[lo]) / (hi - lo)


def load(path):
    data = loadmat(path)
    reward = data["raw_rewards"] if "raw_rewards" in data else data["rewards"]
    return dict(reward=np.ravel(reward).astype(float),
                global_cost=np.ravel(data["global_costs"]).astype(float),
                local_cost=np.asarray(data["local_costs_per_agent"], float).sum(axis=1))


def paper_runs(archive, case, method):
    return [{metric: np.asarray(archive[f"{case}/{method}/seed{seed}/{metric}"], float)
             for metric, _ in PANELS} for seed in range(1, 6)]


def statistics(runs, window):
    n = min(len(run["reward"]) for run in runs)
    result = {}
    for metric, _ in PANELS:
        curves = np.stack([moving_average(run[metric][:n], window) for run in runs])
        mean = curves.mean(axis=0)
        se = curves.std(axis=0, ddof=1) / np.sqrt(len(runs)) if len(runs) > 1 else np.zeros(n)
        result[metric] = (mean, mean - se if metric == "reward" else np.maximum(mean - se, 0), mean + se)
    return result


def mpc_rewards(path):
    values = {case: [] for case in CASES}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            if row["method"] == "dist_mpc":
                values[row["case"]].append(float(row["reward"]))
    return {case: float(np.mean(days)) for case, days in values.items()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=ROOT / "results/runs")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--paper", action="store_true")
    parser.add_argument("--archive", type=Path, default=ROOT / "results/paper/training_curves.npz",
                        help="curve archive used with --paper; download with scripts/download_curves.sh")
    parser.add_argument("--usetex", action="store_true", help="use the paper LaTeX fonts (requires a local TeX installation)")
    parser.add_argument("--out", type=Path, default=ROOT / "results/figures/training_curves")
    parser.add_argument("--window", type=int, default=201)
    parser.add_argument("--mpc-reference", type=Path, default=ROOT / "results/paper/mpc_per_day.csv")
    parser.add_argument("--no-mpc-reference", action="store_true")
    args = parser.parse_args(argv)
    archive = np.load(args.archive) if args.paper else None
    stats = {}
    for case in args.cases:
        stats[case] = {}
        for method in args.methods:
            runs = (paper_runs(archive, case, method) if args.paper else
                    [load(path) for path in sorted((args.runs / case / method).glob("seed*/training_metrics_*.mat"))])
            stats[case][method] = statistics(runs, args.window)
            print(f"{case} {LABELS[method]}: {len(runs)} seeds, {len(runs[0]['reward'])} episodes")
    reference = None if args.no_mpc_reference else mpc_rewards(args.mpc_reference)
    from scripts.figure2_render import plot
    names = dict(proposed="proposed", mappo_pf="penalty", mappo_lag="unified", mappo_vanilla="unsafe")
    render_stats = {case: {names[method]: values for method, values in methods.items()}
                    for case, methods in stats.items()}
    events = [event for event in PAPER_EVENTS if event[0] in args.cases]
    gap_vs = "unified" if args.paper and {"proposed", "mappo_lag"}.issubset(args.methods) else None
    plot(render_stats, args.out, gap_vs=gap_vs, mpc_refs=reference,
         events=events if args.paper and "mappo_lag" in args.methods else (), use_tex=args.usetex,
         xlim=(0.0, 2.0) if args.paper else None)


if __name__ == "__main__":
    main()
