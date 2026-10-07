"""PPO trainer with separate local and global Lagrange multipliers (docs/algorithm.md).

    hybrid advantage  A_h = A_r - lambda_i^l * A_{i,l} - lambda^g * A_g
    multipliers       lambda <- lambda + eta_lambda * J   (budgets d = 0)
"""
import numpy as np
import torch
from torch import nn

from mappo_lagrangian.popart import PopArt


def joint_ratio(new_log_probs, old_log_probs):
    """Probability ratio of an agent's complete action vector."""
    return (new_log_probs - old_log_probs).sum(-1, keepdim=True).exp()


def discounted_episode_cost(costs, gamma):
    """Discounted daily cost, averaged over the parallel days of the rollout."""
    weights = gamma ** np.arange(costs.shape[0])
    return float((costs[..., 0] * weights[:, None]).sum(axis=0).mean())


class LocalGlobalTrainer:
    def __init__(self, args, policy, device=torch.device("cpu")):
        self.policy = policy
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.method = args.method
        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta
        self.gamma = args.gamma
        self.lambda_lr = args.lambda_lr
        constrained = self.method == "proposed"
        self.lambda_local = float(args.lambda_local_init) if constrained else 0.0
        self.lambda_global = float(args.lambda_global_init) if constrained else 0.0
        self.value_normalizer = PopArt(1, device=device)
        self.local_normalizer = PopArt(1, device=device)
        self.global_normalizer = PopArt(1, device=device)

    def update_duals(self, buffer):
        """One multiplier step per iteration, before the policy updates."""
        local = discounted_episode_cost(buffer.local_costs, self.gamma)
        global_cost = discounted_episode_cost(buffer.global_costs, self.gamma)
        if self.method == "proposed":
            self.lambda_local += self.lambda_lr * local
            self.lambda_global += self.lambda_lr * global_cost
        return local, global_cost

    @torch.no_grad()
    def update_normalizer(self, normalizer, critic, optimizer, returns, predictions):
        """PopArt update of one channel that preserves the critic's unnormalized outputs."""
        raw_predictions = normalizer.denormalize(predictions)
        old_mean, old_var = normalizer.running_mean_var()
        old_mean, old_std = old_mean.clone(), old_var.sqrt().clone()
        normalizer(returns.reshape(-1, 1), train=True)
        new_mean, new_var = normalizer.running_mean_var()
        new_std = new_var.sqrt()
        scale = old_std / new_std
        critic.v_out.weight.mul_(scale[:, None])
        critic.v_out.bias.copy_((old_std * critic.v_out.bias + old_mean - new_mean) / new_std)
        for parameter in (critic.v_out.weight, critic.v_out.bias):
            state = optimizer.state.get(parameter, {})
            if "exp_avg" in state:
                state["exp_avg"].mul_(scale)
                state["exp_avg_sq"].mul_(scale.square())
                if "max_exp_avg_sq" in state:
                    state["max_exp_avg_sq"].mul_(scale.square())
        predictions[...] = normalizer(raw_predictions, train=False).cpu().numpy()

    def value_loss(self, values, old_values, returns, masks, normalizer):
        target = normalizer(returns, train=False)
        clipped = old_values + (values - old_values).clamp(-self.clip_param, self.clip_param)
        original = torch.nn.functional.huber_loss(values, target, reduction="none", delta=self.huber_delta)
        clipped_loss = torch.nn.functional.huber_loss(clipped, target, reduction="none", delta=self.huber_delta)
        loss = torch.maximum(original, clipped_loss)
        return (loss * masks).sum() / masks.sum().clamp_min(1)

    def train(self, buffer):
        channels = (
            (buffer.returns, buffer.value_preds, self.value_normalizer,
             self.policy.critic, self.policy.critic_optimizer),
            (buffer.global_returns, buffer.global_preds, self.global_normalizer,
             self.policy.cost2_critic, self.policy.cost2_optimizer),
            (buffer.local_returns, buffer.local_preds, self.local_normalizer,
             self.policy.local_cost_critic, self.policy.local_cost_optimizer),
        )
        enabled = (True, self.method == "proposed", self.method == "proposed")
        advantages = [returns[:-1] - norm.denormalize(preds[:-1]) if use else np.zeros_like(returns[:-1])
                      for use, (returns, preds, norm, _, _) in zip(enabled, channels)]
        active = buffer.active_masks[:-1] > 0
        # each channel is centered; all channels share the scale of the hybrid advantage
        centered = [adv - adv[active].mean() for adv in advantages]
        hybrid = centered[0] - self.lambda_global * centered[1] - self.lambda_local * centered[2]
        scale = max(float(hybrid[active].std()), 1e-5)
        reward_adv, global_adv, local_adv = [adv / scale for adv in centered]
        for use, (returns, preds, normalizer, critic, optimizer) in zip(enabled, channels):
            if use:
                self.update_normalizer(normalizer, critic, optimizer, returns[:-1], preds[:-1])
        info = {}
        updates = 0
        for _ in range(self.ppo_epoch):
            for sample in buffer.feed_forward_generator(reward_adv, global_adv, local_adv, self.num_mini_batch):
                for key, value in self.ppo_update(sample).items():
                    info[key] = info.get(key, 0.0) + value
                updates += 1
        info = {key: value / updates for key, value in info.items()}
        info.update(lambda_local=self.lambda_local, lambda_global=self.lambda_global,
                    reward_adv_std=float(centered[0][active].std()),
                    local_penalty_adv_std=float(self.lambda_local * centered[2][active].std()),
                    global_penalty_adv_std=float(self.lambda_global * centered[1][active].std()))
        return info

    def ppo_update(self, sample):
        (share_obs, obs, actions, old_log_probs, active, factor, old_values, returns, reward_adv,
         old_global_values, global_returns, global_adv, old_local_values, local_returns, local_adv) = sample
        to_tensor = lambda array: torch.as_tensor(array, **self.tpdv)
        old_log_probs, reward_adv, global_adv, local_adv, factor, active = map(
            to_tensor, (old_log_probs, reward_adv, global_adv, local_adv, factor, active))
        values, log_probs, entropy, global_values, local_values = self.policy.evaluate_actions(
            share_obs, obs, actions, active)
        ratio = joint_ratio(log_probs, old_log_probs)
        hybrid = reward_adv - self.lambda_global * global_adv - self.lambda_local * local_adv
        surrogate = torch.minimum(ratio * hybrid,
                                  ratio.clamp(1 - self.clip_param, 1 + self.clip_param) * hybrid)
        actor_loss = -(factor * surrogate * active).sum() / active.sum().clamp_min(1)
        self.policy.actor_optimizer.zero_grad()
        (actor_loss - self.entropy_coef * entropy).backward()
        actor_grad = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        self.policy.actor_optimizer.step()
        losses = {}
        for name, values_, old_, targets, norm, critic, optimizer in (
            ("value", values, old_values, returns, self.value_normalizer,
             self.policy.critic, self.policy.critic_optimizer),
            ("global_cost", global_values, old_global_values, global_returns, self.global_normalizer,
             self.policy.cost2_critic, self.policy.cost2_optimizer),
            ("local_cost", local_values, old_local_values, local_returns, self.local_normalizer,
             self.policy.local_cost_critic, self.policy.local_cost_optimizer),
        ):
            if name != "value" and self.method != "proposed":
                continue
            loss = self.value_loss(values_, to_tensor(old_), to_tensor(targets), active, norm)
            optimizer.zero_grad()
            (loss * self.value_loss_coef).backward()
            nn.utils.clip_grad_norm_(critic.parameters(), self.max_grad_norm)
            optimizer.step()
            losses[name + "_loss"] = float(loss.detach())
        delta = (log_probs - old_log_probs).sum(-1, keepdim=True).detach()
        losses.update(policy_loss=float(actor_loss.detach()), dist_entropy=float(entropy.detach()),
                      actor_grad_norm=float(actor_grad), ratio=float(ratio.detach().mean()),
                      approximate_kl=float((delta.exp() - 1 - delta).mean()))
        return losses
