"""Rollout storage of one agent (episode_length steps x n_rollout_threads days)."""
import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, args, obs_dim, share_obs_dim, act_dim):
        T, N = args.episode_length, args.n_rollout_threads
        self.episode_length = T
        self.gamma = args.gamma
        self.gae_lambda = args.gae_lambda

        self.share_obs = np.zeros((T + 1, N, share_obs_dim), dtype=np.float32)
        self.obs = np.zeros((T + 1, N, obs_dim), dtype=np.float32)
        self.actions = np.zeros((T, N, act_dim), dtype=np.float32)
        self.action_log_probs = np.zeros((T, N, act_dim), dtype=np.float32)
        self.rewards = np.zeros((T, N, 1), dtype=np.float32)
        self.value_preds = np.zeros((T + 1, N, 1), dtype=np.float32)
        self.returns = np.zeros_like(self.value_preds)
        self.global_costs = np.zeros_like(self.rewards)
        self.global_preds = np.zeros_like(self.value_preds)
        self.global_returns = np.zeros_like(self.value_preds)
        self.local_costs = np.zeros_like(self.rewards)
        self.local_preds = np.zeros_like(self.value_preds)
        self.local_returns = np.zeros_like(self.value_preds)
        self.masks = np.ones((T + 1, N, 1), dtype=np.float32)
        self.active_masks = np.ones_like(self.masks)
        self.factor = None
        self.step = 0

    def update_factor(self, factor):
        self.factor = factor.copy()

    def insert(self, share_obs, obs, actions, action_log_probs, value_preds, rewards, masks,
               global_costs, global_preds, local_costs, local_preds):
        self.share_obs[self.step + 1] = share_obs.copy()
        self.obs[self.step + 1] = obs.copy()
        self.actions[self.step] = actions.copy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()
        self.global_costs[self.step] = global_costs.copy()
        self.global_preds[self.step] = global_preds.copy()
        self.local_costs[self.step] = local_costs.copy()
        self.local_preds[self.step] = local_preds.copy()
        self.step = (self.step + 1) % self.episode_length

    def after_update(self):
        self.share_obs[0] = self.share_obs[-1].copy()
        self.obs[0] = self.obs[-1].copy()
        self.masks[0] = self.masks[-1].copy()

    def compute_returns(self, rewards, preds, returns, next_value, normalizer):
        """GAE returns of one channel; preds are normalized by `normalizer`."""
        preds[-1] = next_value
        gae = 0
        for step in reversed(range(rewards.shape[0])):
            delta = rewards[step] + self.gamma * normalizer.denormalize(preds[step + 1]) * self.masks[step + 1] \
                - normalizer.denormalize(preds[step])
            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
            returns[step] = gae + normalizer.denormalize(preds[step])

    def feed_forward_generator(self, reward_adv, global_adv, local_adv, num_mini_batch):
        batch_size = self.rewards.shape[0] * self.rewards.shape[1]
        mini_batch_size = batch_size // num_mini_batch
        rand = torch.randperm(batch_size).numpy()
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

        share_obs = self.share_obs[:-1].reshape(-1, self.share_obs.shape[-1])
        obs = self.obs[:-1].reshape(-1, self.obs.shape[-1])
        actions = self.actions.reshape(-1, self.actions.shape[-1])
        action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        global_preds = self.global_preds[:-1].reshape(-1, 1)
        global_returns = self.global_returns[:-1].reshape(-1, 1)
        local_preds = self.local_preds[:-1].reshape(-1, 1)
        local_returns = self.local_returns[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        factor = self.factor.reshape(-1, self.factor.shape[-1])
        reward_adv = reward_adv.reshape(-1, 1)
        global_adv = global_adv.reshape(-1, 1)
        local_adv = local_adv.reshape(-1, 1)

        for idx in sampler:
            yield (share_obs[idx], obs[idx], actions[idx], action_log_probs[idx], active_masks[idx], factor[idx],
                   value_preds[idx], returns[idx], reward_adv[idx],
                   global_preds[idx], global_returns[idx], global_adv[idx],
                   local_preds[idx], local_returns[idx], local_adv[idx])
