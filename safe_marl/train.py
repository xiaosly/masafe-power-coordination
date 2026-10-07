"""Train one RL method on one case.

    python -m safe_marl.train --case C1 --seed 1

The defaults are the settings of the paper. Outputs: results/runs/<case>/<method>/seed<k>/
(configuration, training curves, TensorBoard logs, actor checkpoints).
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import setproctitle
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mappo_lagrangian.env_wrapper import MARLEnv
from mappo_lagrangian.vec_env import DummyVecEnv, SubprocVecEnv
from power_envs import CASES
from safe_marl.methods import METHODS


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--case", choices=tuple(CASES), required=True)
    p.add_argument("--method", choices=METHODS, default="proposed")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--output_dir", type=Path)
    p.add_argument("--cuda", action="store_true")
    p.add_argument("--n_training_threads", type=int, default=1)
    # rollouts
    p.add_argument("--num_env_steps", type=int, default=480096)
    p.add_argument("--episode_length", type=int, default=24)
    p.add_argument("--n_rollout_threads", type=int, default=4)
    # networks
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--layer_N", type=int, default=2)
    p.add_argument("--gain", type=float, default=0.01)
    p.add_argument("--std_x_coef", type=float, default=1.0)
    p.add_argument("--std_y_coef", type=float, default=0.5)
    # optimization
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--critic_lr", type=float, default=5e-4)
    p.add_argument("--opti_eps", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--ppo_epoch", type=int, default=5)
    p.add_argument("--num_mini_batch", type=int, default=1)
    p.add_argument("--clip_param", type=float, default=0.2)
    p.add_argument("--entropy_coef", type=float, default=0.008)
    p.add_argument("--value_loss_coef", type=float, default=1.0)
    p.add_argument("--max_grad_norm", type=float, default=10.0)
    p.add_argument("--huber_delta", type=float, default=10.0)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae_lambda", type=float, default=0.95)
    # Lagrange multipliers
    p.add_argument("--lambda_local_init", type=float, default=0.5)
    p.add_argument("--lambda_global_init", type=float, default=0.78)
    p.add_argument("--lambda_lr", type=float, default=5e-4)
    # logging
    p.add_argument("--log_interval", type=int, default=100)
    p.add_argument("--save_interval", type=int, default=1000)
    return p


def main(argv=None):
    dispatch = argparse.ArgumentParser(add_help=False)
    dispatch.add_argument("--method", choices=METHODS, default="proposed")
    method, _ = dispatch.parse_known_args(argv)
    if method.method == "mappo_lag":
        from mappo_lag.train import main as train_lag
        return train_lag(argv)
    from safe_marl.runner import LocalGlobalRunner

    args = build_parser().parse_args(argv)
    run_dir = (args.output_dir or ROOT / "results" / "runs" / args.case / args.method / f"seed{args.seed}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(args.n_training_threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.cuda and torch.cuda.is_available():
        device = torch.device("cuda:0")
        torch.cuda.manual_seed_all(args.seed)
    else:
        device = torch.device("cpu")
    setproctitle.setproctitle(f"safe-marl-{args.method}-{args.case}-seed{args.seed}")
    settings = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    (run_dir / "config.json").write_text(json.dumps(settings, indent=2))

    def make_env(rank):
        def initialize():
            np.random.seed(args.seed + 10000 * rank)
            return MARLEnv(args.case)
        return initialize

    env_fns = [make_env(rank) for rank in range(args.n_rollout_threads)]
    envs = DummyVecEnv(env_fns) if len(env_fns) == 1 else SubprocVecEnv(env_fns)
    try:
        LocalGlobalRunner(args, envs, device, run_dir).run()
    finally:
        envs.close()


if __name__ == "__main__":
    main()
