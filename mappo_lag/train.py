"""Train MAPPO-Lag on one case.

    python -m mappo_lag.train --case C1 --seed 1

Outputs: results/runs/<case>/mappo_lag/seed<k>/.
"""
import json
from pathlib import Path

import numpy as np
import setproctitle
import torch

from mappo_lag.core.config import get_config

ROOT = Path(__file__).resolve().parents[1]
ENV_NAMES = {"C1": "power_s2", "C2": "power_s1", "C3": "power_s3"}


def build_parser():
    parser = get_config()
    parser.description = __doc__
    parser.add_argument("--case", choices=tuple(ENV_NAMES), required=True)
    parser.add_argument("--method", choices=("mappo_lag",), default="mappo_lag")
    parser.add_argument("--output_dir", type=Path)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    args.env_name = ENV_NAMES[args.case]
    args.num_agents = {"C1": 3, "C2": 5, "C3": 12}[args.case]
    run_dir = (args.output_dir or ROOT / "results/runs" / args.case / args.method / f"seed{args.seed}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (run_dir / "config.json").write_text(json.dumps(settings, indent=2))
    torch.set_num_threads(args.n_training_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    setproctitle.setproctitle(f"safe-marl-mappo_lag-{args.case}-seed{args.seed}")

    from mappo_lag.core.envs.env_wrappers import ShareDummyVecEnv, ShareSubprocVecEnv
    from mappo_lag.core.runner.separated.power_runner_lagr import PowerRunner
    from mappo_lag.env import PowerEnvS1, PowerEnvS2, PowerEnvS3

    env_class = {"C1": PowerEnvS2, "C2": PowerEnvS1, "C3": PowerEnvS3}[args.case]
    env_fns = [lambda: env_class(args) for _ in range(args.n_rollout_threads)]
    envs = ShareDummyVecEnv(env_fns) if len(env_fns) == 1 else ShareSubprocVecEnv(env_fns)
    try:
        runner = PowerRunner(dict(all_args=args, envs=envs, eval_envs=None,
                                  num_agents=args.num_agents, device=device, run_dir=run_dir))
        runner.run()
        runner.writter.close()
        (run_dir / f"training_metrics_{args.env_name}.mat").replace(run_dir / f"training_metrics_{args.case}.mat")
    finally:
        envs.close()


if __name__ == "__main__":
    main()
