import torch
from mappo_lag.core.algorithms.r_mappo.algorithm.r_actor_critic import R_Actor, R_Critic
from mappo_lag.core.utils.util import update_linear_schedule


class MACPPOPolicy:
    """
    Safe MAPPO Policy  class. Wraps actor and critic networks to compute actions and value function predictions.

    :param args: (argparse.Namespace) arguments containing relevant model and policy information.
    :param obs_space: (Box) observation space.
    :param cent_obs_space: (Box) value function input space (centralized input for MAPPO, decentralized for IPPO).
    :param action_space: (Box) action space.
    :param device: (torch.device) specifies the device to run on (cpu/gpu).
    """

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay

        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        self.actor = R_Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = R_Critic(args, self.share_obs_space, self.device)
        self.cost1_critic = R_Critic(args, self.share_obs_space, self.device)
        self.cost2_critic = R_Critic(args, self.share_obs_space, self.device)
        self.local_cost_critic = R_Critic(args, self.obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)
        self.cost1_optimizer = torch.optim.Adam(self.cost1_critic.parameters(),
                                               lr=self.critic_lr,
                                               eps=self.opti_eps,
                                               weight_decay=self.weight_decay)
        self.cost2_optimizer = torch.optim.Adam(self.cost2_critic.parameters(),
                                               lr=self.critic_lr,
                                               eps=self.opti_eps,
                                               weight_decay=self.weight_decay)
        self.local_cost_optimizer = torch.optim.Adam(self.local_cost_critic.parameters(),
                                 lr=self.critic_lr,
                                 eps=self.opti_eps,
                                 weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)
        update_linear_schedule(self.cost1_optimizer, episode, episodes, self.critic_lr)
        update_linear_schedule(self.cost2_optimizer, episode, episodes, self.critic_lr)
        update_linear_schedule(self.local_cost_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, masks, available_actions=None,
                    deterministic=False, rnn_states_cost1=None, rnn_states_cost2=None,
                    rnn_states_local_cost=None):
        """
        Compute actions and value function predictions for the given inputs.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.

        :return values: (torch.Tensor) value function predictions.
        :return actions: (torch.Tensor) actions to take.
        :return action_log_probs: (torch.Tensor) log probabilities of chosen actions.
        :return rnn_states_actor: (torch.Tensor) updated actor network RNN states.
        :return rnn_states_critic: (torch.Tensor) updated critic network RNN states.
        """
        actions, action_log_probs, rnn_states_actor = self.actor(obs,
                                                                 rnn_states_actor,
                                                                 masks,
                                                                 available_actions,
                                                                 deterministic)

        values, rnn_states_critic = self.critic(cent_obs, rnn_states_critic, masks)
        if rnn_states_cost1 is None and rnn_states_local_cost is None:
            return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic
        else:
            cost1_preds, rnn_states_cost1 = self.cost1_critic(cent_obs, rnn_states_cost1, masks)
            cost2_preds, rnn_states_cost2 = self.cost2_critic(cent_obs, rnn_states_cost2, masks)
            local_cost_preds, rnn_states_local_cost = self.local_cost_critic(obs, rnn_states_local_cost, masks)
            return values, actions, action_log_probs, rnn_states_actor, rnn_states_critic, cost1_preds, rnn_states_cost1, \
                cost2_preds, rnn_states_cost2, local_cost_preds, rnn_states_local_cost


    def get_values(self, cent_obs, rnn_states_critic, masks):
        """
        Get value function predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        return values

    def get_cost1_values(self, cent_obs, rnn_states_cost1, masks):
        """
        Get constraint cost predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        cost1_preds, _ = self.cost1_critic(cent_obs, rnn_states_cost1, masks)
        return cost1_preds
    def get_cost2_values(self, cent_obs, rnn_states_cost2, masks):
        """
        Get constraint cost predictions.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.

        :return values: (torch.Tensor) value function predictions.
        """
        cost2_preds, _ = self.cost2_critic(cent_obs, rnn_states_cost2, masks)
        return cost2_preds

    def get_local_cost_values(self, obs, rnn_states_local_cost, masks):
        local_cost_preds, _ = self.local_cost_critic(obs, rnn_states_local_cost, masks)
        return local_cost_preds

    def evaluate_actions(self, cent_obs, obs, rnn_states_actor, rnn_states_critic, action, masks,
                         available_actions=None, active_masks=None, rnn_states_cost1=None, rnn_states_cost2=None,
                         rnn_states_local_cost=None):
        """
        Get action logprobs / entropy and value function predictions for actor update.
        :param cent_obs (np.ndarray): centralized input to the critic.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param rnn_states_critic: (np.ndarray) if critic is RNN, RNN states for critic.
        :param action: (np.ndarray) actions whose log probabilites and entropy to compute.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param active_masks: (torch.Tensor) denotes whether an agent is active or dead.

        :return values: (torch.Tensor) value function predictions.
        :return action_log_probs: (torch.Tensor) log probabilities of the input actions.
        :return dist_entropy: (torch.Tensor) action distribution entropy for the given inputs.
        """
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     rnn_states_actor,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks)

        values, _ = self.critic(cent_obs, rnn_states_critic, masks)
        if rnn_states_cost1 is None and rnn_states_local_cost is None:
            return values, action_log_probs, dist_entropy
        else:
            cost1_values, _ = self.cost1_critic(cent_obs, rnn_states_cost1, masks)
            cost2_values, _ = self.cost2_critic(cent_obs, rnn_states_cost2, masks)
            local_cost_values, _ = self.local_cost_critic(obs, rnn_states_local_cost, masks)
            return values, action_log_probs, dist_entropy, cost1_values, cost2_values, local_cost_values

    def act(self, obs, rnn_states_actor, masks, available_actions=None, deterministic=False):
        """
        Compute actions using the given inputs.
        :param obs (np.ndarray): local agent inputs to the actor.
        :param rnn_states_actor: (np.ndarray) if actor is RNN, RNN states for actor.
        :param masks: (np.ndarray) denotes points at which RNN states should be reset.
        :param available_actions: (np.ndarray) denotes which actions are available to agent
                                  (if None, all actions available)
        :param deterministic: (bool) whether the action should be mode of distribution or should be sampled.
        """
        actions, _, rnn_states_actor = self.actor(obs, rnn_states_actor, masks, available_actions, deterministic)
        return actions, rnn_states_actor
