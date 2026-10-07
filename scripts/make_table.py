"""Table I from the evaluation results.

    python scripts/make_table.py            # results/evaluation/ (scripts/evaluate_all.py)
    python scripts/make_table.py --paper    # results/paper/*.csv (runs reported in the paper)

RL: mean over the test days of each training seed, then over the seeds. t: computation time per
decision for all agents, largest of the three cases.
"""
import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CASES = ("C1", "C2", "C3")
RL = (("proposed", "Proposed"), ("mappo_pf", "MAPPO-PF"), ("mappo_lag", "MAPPO-Lag"), ("mappo_vanilla", "MAPPO-Vanilla"))
MPC = (("dec_mpc", "Decentralized MPC"), ("dist_mpc", "Distributed MPC (ADMM)"))
METRICS = ("reward", "global_cost", "local_cost", "violation_hours")


def fmt(x, reward=False):
    if reward:
        return f"{x:.1f}".replace("-", "−")
    if x == 0:
        return "0"
    if abs(x) >= 100:
        return f"{x:.0f}"
    if abs(x) >= 10:
        return f"{x:.1f}"
    if abs(x) >= 0.1:
        return f"{x:.2f}"
    return f"{x:.1g}"


def fmt_time(seconds):
    return f"{seconds:.0e}" if seconds < 0.01 else f"{seconds:.2g}"


def from_paper():
    rows = {}
    with (ROOT / "results/paper/rl_per_seed.csv").open() as f:
        for r in csv.DictReader(f):
            rows.setdefault((r["case"], r["method"]), []).append({k: float(r[k]) for k in METRICS})
    with (ROOT / "results/paper/mpc_per_day.csv").open() as f:
        for r in csv.DictReader(f):
            rows.setdefault((r["case"], r["method"]), []).append(
                {k: float(r[k]) for k in METRICS + ("rounds_per_decision", "seconds_per_decision")})
    with (ROOT / "results/paper/decision_time.csv").open() as f:
        for r in csv.DictReader(f):
            for method, _ in RL:
                for row in rows.get((r["case"], method), []):
                    row["seconds_per_decision"] = float(r["seconds_per_decision"])
    return rows


def from_evaluation():
    from evaluation.evaluate_rl import summary, violation_hours
    rows = {}
    for case in CASES:
        for method, _ in RL:
            for path in sorted((ROOT / "results/evaluation" / case / method).glob("seed*.json")):
                days = json.loads(path.read_text())
                row = summary(days)
                row["seconds_per_decision"] = float(np.mean([d["control_seconds"] for d in days])) / 24
                rows.setdefault((case, method), []).append(row)
        for method, _ in MPC:
            path = ROOT / "results/evaluation" / case / f"{method}.json"
            if path.exists():
                for day in json.loads(path.read_text()):
                    rows.setdefault((case, method), []).append(dict(
                        reward=day["reward"], global_cost=day["global_violation_score"],
                        local_cost=day["local_violation_score"], violation_hours=violation_hours(day),
                        rounds_per_decision=day["mean_rounds_per_decision"],
                        seconds_per_decision=day["mean_seconds_per_decision"]))
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--paper", action="store_true")
    args = p.parse_args()
    rows = from_paper() if args.paper else from_evaluation()
    print("| Method | " + " | ".join(f"{c} R̄ | {c} C_g | {c} C_l" for c in CASES) + " | N_v (h/d) | rounds | t (s) |")
    print("|---" * (2 + 3 * len(CASES) + 2) + "|")
    for method, label in RL + MPC:
        if not any((case, method) in rows for case in CASES):
            continue
        cells, nv, rounds, seconds = [], [], [], []
        for case in CASES:
            values = rows.get((case, method))
            if not values:
                cells += ["", "", ""]
                nv.append("")
                continue
            mean = {k: float(np.mean([v[k] for v in values])) for k in values[0]}
            cells += [fmt(mean["reward"], True), fmt(mean["global_cost"]), fmt(mean["local_cost"])]
            nv.append(f"{mean['violation_hours']:.0f}")
            rounds.append(mean.get("rounds_per_decision", 0.0))
            if "seconds_per_decision" in mean:
                seconds.append(mean["seconds_per_decision"])
        print(f"| {label} | " + " | ".join(cells) + f" | {'/'.join(nv)} | {'/'.join(f'{r:.0f}' for r in rounds)} | "
              f"{fmt_time(max(seconds)) if seconds else ''} |")


if __name__ == "__main__":
    main()
