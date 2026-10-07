import numpy as np
import torch
import torch.nn as nn
from mappo_lag.core.utils.util import get_gard_norm, huber_loss, mse_loss
from mappo_lag.core.utils.popart import PopArt
from mappo_lag.core.algorithms.utils.util import check

class R_MAPPO_Lagr:
    """
    Trainer class for MAPPO-L to update policies.
    :param args: (argparse.Namespace) arguments containing relevant model, policy, and env information.
    :param policy: (R_MAPPO_Policy) policy to update.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    :param precompute: Use an 'input' for the linearization constant instead of true_linear_leq_constraint.
                           If present, overrides surrogate
                           When using precompute, the last input is the precomputed linearization constant

    :param attempt_(in)feasible_recovery: deals with cases where x=0 is infeasible point but problem still feasible
                                                               (where optimization problem is entirely infeasible)

    :param revert_to_last_safe_point: Behavior protocol for situation when optimization problem is entirely infeasible.
                                          Specifies that we should just reset the parameters to the last point
                                          that satisfied constraint.
    """

    def __init__(self,
                 args,
                 policy, hvp_approach=None, attempt_feasible_recovery=False,
                 attempt_infeasible_recovery=False, revert_to_last_safe_point=False, delta_bound=0.02, safety_bound=10,
                 _backtrack_ratio=0.8, _max_backtracks=15, _constraint_name_1="trust_region",
                 _constraint_name_2="safety_region", linesearch_infeasible_recovery=True, accept_violation=False,
                 device=torch.device("cpu")):
        self.args = args
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy

        self._damping = 0.00001

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        self.data_chunk_length = args.data_chunk_length
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta
        self.gamma = args.gamma

        self._use_recurrent_policy = args.use_recurrent_policy
        self._use_naive_recurrent = args.use_naive_recurrent_policy
        self._use_max_grad_norm = args.use_max_grad_norm
        self._use_clipped_value_loss = args.use_clipped_value_loss
        self._use_huber_loss = args.use_huber_loss
        self._use_popart = args.use_popart
        self._use_value_active_masks = args.use_value_active_masks
        self._use_policy_active_masks = args.use_policy_active_masks

        self.attempt_feasible_recovery = attempt_feasible_recovery
        self.attempt_infeasible_recovery = attempt_infeasible_recovery
        self.revert_to_last_safe_point = revert_to_last_safe_point
        num_slices = 1
        self._max_quad_constraint_val = delta_bound
        self._max_lin_constraint_val = safety_bound
        self._backtrack_ratio = _backtrack_ratio
        self._max_backtracks = _max_backtracks
        self._constraint_name_1 = _constraint_name_1
        self._constraint_name_2 = _constraint_name_2
        self._linesearch_infeasible_recovery = linesearch_infeasible_recovery
        self._accept_violation = accept_violation

        self.lagrangian_coef = args.lagrangian_coef_rate
        self.lamda_lagr1 = args.lamda_lagr1
        self.lamda_lagr2 = args.lamda_lagr2
        self.safety_bound = args.safety_bound

        self.lamda_local = args.lamda_local
        self.local_safety_bound = args.local_safety_bound


        self._hvp_approach = hvp_approach

        if self._use_popart:
            self.value_normalizer = PopArt(1, device=self.device)
        else:
            self.value_normalizer = None

    def cal_value_loss(self, values, value_preds_batch, return_batch, active_masks_batch):
        """
        Calculate value function loss.
        :param values: (torch.Tensor) value function predictions.
        :param value_preds_batch: (torch.Tensor) "old" value  predictions from data batch (used for value clip loss)
        :param return_batch: (torch.Tensor) reward to go returns.
        :param active_masks_batch: (torch.Tensor) denotes if agent is active or dead at a given timesep.

        :return value_loss: (torch.Tensor) value function loss.
        """
        if self._use_popart:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
            error_clipped = self.value_normalizer(return_batch) - value_pred_clipped
            error_original = self.value_normalizer(return_batch) - values
        else:
            value_pred_clipped = value_preds_batch + (values - value_preds_batch).clamp(-self.clip_param,
                                                                                        self.clip_param)
            error_clipped = return_batch - value_pred_clipped
            error_original = return_batch - values

        if self._use_huber_loss:
            value_loss_clipped = huber_loss(error_clipped, self.huber_delta)
            value_loss_original = huber_loss(error_original, self.huber_delta)
        else:
            value_loss_clipped = mse_loss(error_clipped)
            value_loss_original = mse_loss(error_original)

        if self._use_clipped_value_loss:
            value_loss = torch.max(value_loss_original, value_loss_clipped)
        else:
            value_loss = value_loss_original

        if self._use_value_active_masks:
            value_loss = (value_loss * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            value_loss = value_loss.mean()

        return value_loss

    def _get_flat_grad(self, y: torch.Tensor, model: nn.Module, **kwargs) -> torch.Tensor:

        grads = torch.autograd.grad(y, model.parameters(), **kwargs, allow_unused=True)

        _grads = []
        for val in grads:
            if val != None:
                _grads.append(val);

        return torch.cat([grad.reshape(-1) for grad in _grads])

    def _conjugate_gradients(self, b: torch.Tensor, flat_kl_grad: torch.Tensor, nsteps: int = 10,
                             residual_tol: float = 1e-10) -> torch.Tensor:
        x = torch.zeros_like(b)
        r, p = b.clone(), b.clone()


        rdotr = r.dot(r)
        for i in range(nsteps):
            z = self.cal_second_hessian(p, flat_kl_grad)
            alpha = rdotr / p.dot(z)
            x += alpha * p
            r -= alpha * z
            new_rdotr = r.dot(r)
            if new_rdotr < residual_tol:
                break
            p = r + new_rdotr / rdotr * p
            rdotr = new_rdotr
        return x

    def cal_second_hessian(self, v: torch.Tensor, flat_kl_grad: torch.Tensor) -> torch.Tensor:
        """Matrix vector product."""

        kl_v = (flat_kl_grad * v).sum()
        flat_kl_grad_grad = self._get_flat_grad(
            kl_v, self.policy.actor, retain_graph=True).detach()
        return flat_kl_grad_grad + v * self._damping

    def _set_from_flat_params(self, model: nn.Module, flat_params: torch.Tensor) -> nn.Module:
        prev_ind = 0
        for param in model.parameters():
            flat_size = int(np.prod(list(param.size())))
            param.data.copy_(
                flat_params[prev_ind:prev_ind + flat_size].view(param.size()))
            prev_ind += flat_size
        return model

    def ppo_update(self, sample, update_actor=True, precomputed_eval=None,
                   precomputed_threshold=None,
                   diff_threshold=False):
        """Update the actor and critics from one rollout mini-batch."""

        share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, \
        value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, \
        adv_targ, available_actions_batch, factor_batch, \
            cost1_preds_batch, cost1_returns_barch, rnn_states_cost1_batch, cost1_adv_targ, aver_episode_costs1,\
            cost2_preds_batch, cost2_returns_barch, rnn_states_cost2_batch,cost2_adv_targ, aver_episode_costs2,\
            local_cost_preds_batch, local_cost_return_batch, rnn_states_local_cost_batch, local_cost_adv_targ, aver_episode_local_costs = sample

        old_action_log_probs_batch = check(old_action_log_probs_batch).to(**self.tpdv)
        adv_targ = check(adv_targ).to(**self.tpdv)
        cost1_adv_targ = check(cost1_adv_targ).to(**self.tpdv)
        cost2_adv_targ = check(cost2_adv_targ).to(**self.tpdv)
        value_preds_batch = check(value_preds_batch).to(**self.tpdv)
        return_batch = check(return_batch).to(**self.tpdv)
        active_masks_batch = check(active_masks_batch).to(**self.tpdv)
        factor_batch = check(factor_batch).to(**self.tpdv)
        cost1_returns_barch = check(cost1_returns_barch).to(**self.tpdv)
        cost2_returns_barch = check(cost2_returns_barch).to(**self.tpdv)
        local_cost_adv_targ = check(local_cost_adv_targ).to(**self.tpdv)
        local_cost_return_batch = check(local_cost_return_batch).to(**self.tpdv)
        local_cost_preds_batch = check(local_cost_preds_batch).to(**self.tpdv)

        cost1_preds_batch = check(cost1_preds_batch).to(**self.tpdv)
        cost2_preds_batch = check(cost2_preds_batch).to(**self.tpdv)
        rnn_states_local_cost_batch = check(rnn_states_local_cost_batch).to(**self.tpdv)

        aver_episode_costs1_t = torch.as_tensor(aver_episode_costs1, **self.tpdv)
        aver_episode_costs2_t = torch.as_tensor(aver_episode_costs2, **self.tpdv)
        aver_episode_local_costs_t = torch.as_tensor(aver_episode_local_costs, **self.tpdv)


        values, action_log_probs, dist_entropy, cost1_values, cost2_values, local_cost_values = self.policy.evaluate_actions(share_obs_batch,
                                                                                           obs_batch,
                                                                                           rnn_states_batch,
                                                                                           rnn_states_critic_batch,
                                                                                           actions_batch,
                                                                                           masks_batch,
                                                                                           available_actions_batch,
                                                                                           active_masks_batch,
                                                                                           rnn_states_cost1_batch,
                                                   rnn_states_cost2_batch,
                                                   rnn_states_local_cost_batch)


        adv_targ_hybrid =  adv_targ - self.lamda_lagr1*cost1_adv_targ - self.lamda_lagr2*cost2_adv_targ - self.lamda_local * local_cost_adv_targ


        imp_weights = torch.exp(action_log_probs - old_action_log_probs_batch)

        surr1 = imp_weights * adv_targ_hybrid
        surr2 = torch.clamp(imp_weights, 1.0 - self.clip_param, 1.0 + self.clip_param) * adv_targ_hybrid

        if self._use_policy_active_masks:
            policy_action_loss = (-torch.sum(factor_batch * torch.min(surr1, surr2),
                                             dim=-1,
                                             keepdim=True) * active_masks_batch).sum() / active_masks_batch.sum()
        else:
            policy_action_loss = -torch.sum(factor_batch * torch.min(surr1, surr2), dim=-1, keepdim=True).mean()

        policy_loss = policy_action_loss

        self.policy.actor_optimizer.zero_grad()

        if update_actor:
            (policy_loss - dist_entropy * self.entropy_coef).backward()

        if self._use_max_grad_norm:
            actor_grad_norm = nn.utils.clip_grad_norm_(self.policy.actor.parameters(), self.max_grad_norm)
        else:
            actor_grad_norm = get_gard_norm(self.policy.actor.parameters())

        self.policy.actor_optimizer.step()


        delta_lamda_lagr1 = -(( aver_episode_costs1_t.mean() - self.safety_bound) * (1 - self.gamma) + (imp_weights * cost1_adv_targ)).mean().detach()
        delta_lamda_lagr2 = -((aver_episode_costs2_t.mean() - self.safety_bound) * (1 - self.gamma) + (
                    imp_weights * cost2_adv_targ)).mean().detach()

        R_Relu = torch.nn.ReLU()
        new_lamda_lagr1 = R_Relu(self.lamda_lagr1 - (delta_lamda_lagr1 * self.lagrangian_coef))
        new_lamda_lagr2 = R_Relu(
            self.lamda_lagr2 - (delta_lamda_lagr2 * self.lagrangian_coef))

        self.lamda_lagr1 = new_lamda_lagr1
        self.lamda_lagr2 = new_lamda_lagr2

        delta_lamda_local = -((aver_episode_local_costs_t.mean() - self.local_safety_bound) * (1 - self.gamma) +
                              (imp_weights * local_cost_adv_targ)).mean().detach()
        new_lamda_local = R_Relu(self.lamda_local - (delta_lamda_local * self.lagrangian_coef))
        self.lamda_local = new_lamda_local


        value_loss = self.cal_value_loss(values, value_preds_batch, return_batch, active_masks_batch)
        self.policy.critic_optimizer.zero_grad()
        (value_loss * self.value_loss_coef).backward()
        if self._use_max_grad_norm:
            critic_grad_norm = nn.utils.clip_grad_norm_(self.policy.critic.parameters(), self.max_grad_norm)
        else:
            critic_grad_norm = get_gard_norm(self.policy.critic.parameters())
        self.policy.critic_optimizer.step()


        cost1_loss = self.cal_value_loss(cost1_values, cost1_preds_batch, cost1_returns_barch, active_masks_batch)
        self.policy.cost1_optimizer.zero_grad()
        (cost1_loss * self.value_loss_coef).backward()

        cost2_loss = self.cal_value_loss(cost2_values, cost2_preds_batch, cost2_returns_barch, active_masks_batch)
        self.policy.cost2_optimizer.zero_grad()
        (cost2_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            cost1_grad_norm = nn.utils.clip_grad_norm_(self.policy.cost1_critic.parameters(), self.max_grad_norm)
            cost2_grad_norm = nn.utils.clip_grad_norm_(self.policy.cost2_critic.parameters(), self.max_grad_norm)
        else:
            cost1_grad_norm = get_gard_norm(self.policy.cost1_critic.parameters())
            cost2_grad_norm = get_gard_norm(self.policy.cost2_critic.parameters())
        self.policy.cost1_optimizer.step()
        self.policy.cost2_optimizer.step()

        local_cost_loss = self.cal_value_loss(local_cost_values, local_cost_preds_batch, local_cost_return_batch, active_masks_batch)
        self.policy.local_cost_optimizer.zero_grad()
        (local_cost_loss * self.value_loss_coef).backward()

        if self._use_max_grad_norm:
            local_cost_grad_norm = nn.utils.clip_grad_norm_(self.policy.local_cost_critic.parameters(), self.max_grad_norm)
        else:
            local_cost_grad_norm = get_gard_norm(self.policy.local_cost_critic.parameters())
        self.policy.local_cost_optimizer.step()

        return value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, cost1_loss, cost1_grad_norm, cost2_loss, cost2_grad_norm, local_cost_loss, local_cost_grad_norm

    def train(self, buffer, update_actor=True):
        """
        Perform a training update using minibatch GD.
        :param buffer: (SharedReplayBuffer) buffer containing training data.
        :param update_actor: (bool) whether to update actor network.

        :return train_info: (dict) contains information regarding training update (e.g. loss, grad norms, etc).
        """
        if self._use_popart:
            advantages = buffer.returns[:-1] - self.value_normalizer.denormalize(buffer.value_preds[:-1])
        else:
            advantages = buffer.returns[:-1] - buffer.value_preds[:-1]
        advantages_copy = advantages.copy()
        advantages_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_advantages = np.nanmean(advantages_copy)
        std_advantages = np.nanstd(advantages_copy)
        advantages = (advantages - mean_advantages) / (std_advantages + 1e-5)

        if self._use_popart:
            cost1_adv = buffer.cost1_returns[:-1] - self.value_normalizer.denormalize(buffer.cost1_preds[:-1])
            cost2_adv = buffer.cost2_returns[:-1] - self.value_normalizer.denormalize(buffer.cost2_preds[:-1])
            local_cost_adv = buffer.local_cost_returns[:-1] - self.value_normalizer.denormalize(buffer.local_cost_preds[:-1])
        else:
            cost1_adv = buffer.cost1_returns[:-1] - buffer.cost1_preds[:-1]
            cost2_adv = buffer.cost2_returns[:-1] - buffer.cost2_preds[:-1]
            local_cost_adv = buffer.local_cost_returns[:-1] - buffer.local_cost_preds[:-1]
        cost1_adv_copy = cost1_adv.copy()
        cost2_adv_copy = cost2_adv.copy()
        local_cost_adv_copy = local_cost_adv.copy()
        cost1_adv_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        cost2_adv_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        local_cost_adv_copy[buffer.active_masks[:-1] == 0.0] = np.nan
        mean_cost1_adv = np.nanmean(cost1_adv_copy)
        mean_cost2_adv = np.nanmean(cost2_adv_copy)
        mean_local_cost_adv = np.nanmean(local_cost_adv_copy)
        std_cost1_adv = np.nanstd(cost1_adv_copy)
        std_cost2_adv = np.nanstd(cost2_adv_copy)
        std_local_cost_adv = np.nanstd(local_cost_adv_copy)
        cost1_adv = (cost1_adv - mean_cost1_adv) / (std_cost1_adv + 1e-5)
        cost2_adv = (cost2_adv - mean_cost2_adv) / (std_cost2_adv + 1e-5)
        local_cost_adv = (local_cost_adv - mean_local_cost_adv) / (std_local_cost_adv + 1e-5)

        train_info = {}

        train_info['value_loss'] = 0
        train_info['policy_loss'] = 0
        train_info['dist_entropy'] = 0
        train_info['actor_grad_norm'] = 0
        train_info['critic_grad_norm'] = 0
        train_info['ratio'] = 0
        train_info['cost1_grad_norm'] = 0
        train_info['cost1_loss'] = 0
        train_info['cost2_grad_norm'] = 0
        train_info['cost2_loss'] = 0
        train_info['local_cost_grad_norm'] = 0
        train_info['local_cost_loss'] = 0

        for _ in range(self.ppo_epoch):
            if self._use_naive_recurrent:
                data_generator = buffer.naive_recurrent_generator(advantages, self.num_mini_batch, cost_adv)
            else:
                data_generator = buffer.feed_forward_generator(advantages, self.num_mini_batch, cost1_adv=cost1_adv,cost2_adv=cost2_adv,
                                                              local_cost_adv=local_cost_adv)

            for sample in data_generator:

                value_loss, critic_grad_norm, policy_loss, dist_entropy, actor_grad_norm, imp_weights, cost1_loss, cost1_grad_norm, \
                    cost2_loss, cost2_grad_norm, local_cost_loss, local_cost_grad_norm = self.ppo_update(sample, update_actor, precomputed_threshold=None,
                                      diff_threshold=False)

                train_info['value_loss'] += value_loss.item()
                train_info['policy_loss'] += policy_loss.item()
                train_info['dist_entropy'] += dist_entropy.item()
                train_info['actor_grad_norm'] += actor_grad_norm
                train_info['critic_grad_norm'] += critic_grad_norm
                train_info['ratio'] += imp_weights.mean()
                train_info['cost1_loss'] += cost1_loss.item()
                train_info['cost1_grad_norm'] += cost1_grad_norm
                train_info['cost2_loss'] += cost2_loss.item()
                train_info['cost2_grad_norm'] += cost2_grad_norm
                train_info['local_cost_loss'] += local_cost_loss.item()
                train_info['local_cost_grad_norm'] += local_cost_grad_norm

        num_updates = self.ppo_epoch * self.num_mini_batch

        for k in train_info.keys():
            train_info[k] /= num_updates

        return train_info

    def prep_training(self):
        self.policy.actor.train()
        self.policy.critic.train()
        self.policy.cost1_critic.train()
        self.policy.cost2_critic.train()
        self.policy.local_cost_critic.train()

    def prep_rollout(self):
        self.policy.actor.eval()
        self.policy.critic.eval()
        self.policy.cost1_critic.eval()
        self.policy.cost2_critic.eval()
        self.policy.local_cost_critic.eval()
