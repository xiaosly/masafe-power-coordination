"""Rollout collection with one policy and one buffer per agent."""
from pathlib import Path

import numpy as np
import torch
from tensorboardX import SummaryWriter

from mappo_lagrangian.buffer import ReplayBuffer
from mappo_lagrangian.policy import Policy


def _t2n(x):
    return x.detach().cpu().numpy()


class Runner:
    def __init__(self, args, envs, device, run_dir):
        self.args = args
        self.envs = envs
        self.device = device
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.num_env_steps = args.num_env_steps
        self.log_interval = args.log_interval
        self.save_interval = args.save_interval
        self.run_dir = Path(run_dir)
        self.writter = SummaryWriter(str(self.run_dir / "logs"))
        self.save_dir = self.run_dir / "models"
        self.save_dir.mkdir(parents=True, exist_ok=True)
        obs_dim, share_obs_dim, act_dim, self.num_agents = envs.dims
        self.policy = [Policy(args, obs_dim, share_obs_dim, act_dim, device) for _ in range(self.num_agents)]
        self.buffer = [ReplayBuffer(args, obs_dim, share_obs_dim, act_dim) for _ in range(self.num_agents)]

    def warmup(self):
        obs, share_obs = self.envs.reset()
        for agent_id, buffer in enumerate(self.buffer):
            buffer.share_obs[0] = share_obs[:, agent_id].copy()
            buffer.obs[0] = obs[:, agent_id].copy()

    @torch.no_grad()
    def collect(self, step):
        """Values, actions, log-probabilities and cost predictions, shape (threads, agents, dim)."""
        outputs = [policy.get_actions(buffer.share_obs[step], buffer.obs[step])
                   for policy, buffer in zip(self.policy, self.buffer)]
        return [np.array([_t2n(out[k]) for out in outputs]).transpose(1, 0, 2) for k in range(5)]

    def insert(self, obs, share_obs, rewards, global_costs, local_costs, dones,
               values, actions, action_log_probs, global_preds, local_preds):
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[np.all(dones, axis=1)] = 0.0
        for agent_id, buffer in enumerate(self.buffer):
            buffer.insert(share_obs[:, agent_id], obs[:, agent_id], actions[:, agent_id],
                          action_log_probs[:, agent_id], values[:, agent_id], rewards[:, agent_id], masks[:, agent_id],
                          global_costs[:, agent_id], global_preds[:, agent_id],
                          local_costs[:, agent_id], local_preds[:, agent_id])

    def save(self):
        for agent_id, policy in enumerate(self.policy):
            torch.save(policy.actor.state_dict(), str(self.save_dir / f"actor_agent{agent_id}.pt"))
            torch.save(policy.critic.state_dict(), str(self.save_dir / f"critic_agent{agent_id}.pt"))
