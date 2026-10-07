"""Networks of one agent: actor, reward critic and cost critics."""
import torch

from mappo_lagrangian.networks import Actor, Critic


class Policy:
    def __init__(self, args, obs_dim, share_obs_dim, act_dim, device=torch.device("cpu")):
        self.actor = Actor(args, obs_dim, act_dim, device)
        self.critic = Critic(args, share_obs_dim, device)
        self.cost1_critic = Critic(args, share_obs_dim, device)
        self.cost2_critic = Critic(args, share_obs_dim, device)
        self.local_cost_critic = Critic(args, obs_dim, device)

        def adam(module, lr):
            return torch.optim.Adam(module.parameters(), lr=lr, eps=args.opti_eps, weight_decay=args.weight_decay)

        self.actor_optimizer = adam(self.actor, args.lr)
        self.critic_optimizer = adam(self.critic, args.critic_lr)
        self.cost2_optimizer = adam(self.cost2_critic, args.critic_lr)
        self.local_cost_optimizer = adam(self.local_cost_critic, args.critic_lr)

    def get_actions(self, share_obs, obs):
        actions, log_probs = self.actor(obs)
        values = self.critic(share_obs)
        global_preds = self.cost2_critic(share_obs)
        local_preds = self.local_cost_critic(obs)
        return values, actions, log_probs, global_preds, local_preds

    def evaluate_actions(self, share_obs, obs, actions, active_masks):
        log_probs, entropy = self.actor.evaluate_actions(obs, actions, active_masks)
        values = self.critic(share_obs)
        global_values = self.cost2_critic(share_obs)
        local_values = self.local_cost_critic(obs)
        return values, log_probs, entropy, global_values, local_values
