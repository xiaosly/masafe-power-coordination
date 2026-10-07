import numpy as np
import torch

from evaluation.policy import DeterministicPolicy
from mappo_lagrangian.networks import Actor
from power_envs import CASES
from safe_marl.train import build_parser


def test_deterministic_policy_matches_actor_mean(tmp_path):
    args = build_parser().parse_args(["--case", "C1"])
    torch.manual_seed(0)
    np.random.seed(0)
    env = CASES["C1"]()
    obs = env.reset()
    actors = [Actor(args, env.obs_dim, env.act_dim) for _ in env.agent_ids]
    for j, actor in enumerate(actors):
        torch.save(actor.state_dict(), tmp_path / f"actor_agent{j}.pt")
    actions, _ = DeterministicPolicy(env, tmp_path)(env, obs)
    for actor, i in zip(actors, env.agent_ids):
        with torch.no_grad():
            mean, _ = actor(obs[i][None], deterministic=True)
        np.testing.assert_allclose(actions[i], np.clip((mean.numpy()[0] + 1) / 2, 0, 1), atol=1e-6)
