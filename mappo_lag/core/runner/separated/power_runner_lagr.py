import time
import numpy as np
import torch
import os
from scipy.io import savemat

from mappo_lag.core.runner.separated.base_runner_mappo_lagr import Runner

def _t2n(x):
    return x.detach().cpu().numpy()


class PowerRunner(Runner):
    """MAPPO-Lag rollout and training loop for the three power-system cases."""

    def __init__(self, config):

        super(PowerRunner, self).__init__(config)
        self.num_agents = config['num_agents']

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        train_episode_rewards = [0 for _ in range(self.n_rollout_threads)]
        train_episode_costs1 = [0 for _ in range(self.n_rollout_threads)]
        train_episode_costs2 = [0 for _ in range(self.n_rollout_threads)]

        train_episode_local_costs = np.zeros((self.n_rollout_threads, self.num_agents))

        done_episodes_rewards = []
        done_episodes_costs1 = []
        done_episodes_costs2 = []

        done_episodes_local_costs_per_agent = []

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):

                values, actions, action_log_probs, rnn_states, rnn_states_critic, cost1_preds, \
                    rnn_states_cost1, cost2_preds, rnn_states_cost2, local_cost_preds, rnn_states_local_cost = self.collect(step)


                obs, share_obs, rewards, costs1, costs2, local_costs, dones, infos, _ = self.envs.step(actions)

                dones_env = np.all(dones, axis=1)


                reward_env = np.mean(rewards, axis=1).flatten()
                cost1_env = np.mean(costs1, axis=1).flatten()
                cost2_env = np.mean(costs2, axis=1).flatten()

                local_cost_per_agent = local_costs.reshape(self.n_rollout_threads, self.num_agents)

                train_episode_rewards += reward_env
                train_episode_costs1 += cost1_env
                train_episode_costs2 += cost2_env
                train_episode_local_costs += local_cost_per_agent

                for t in range(self.n_rollout_threads):
                    if dones_env[t]:
                        per_agent_str = " | ".join(
                            f"Agent{i} LC: {train_episode_local_costs[t, i]:.4f}"
                            for i in range(self.num_agents)
                        )
                        print(
                            f"Train Done | Episode {episode} "
                            f"| Reward: {train_episode_rewards[t]:.2f} "
                            f"| Global Cost: {train_episode_costs2[t]:.4f} "
                            f"|| {per_agent_str}"
                        )
                        done_episodes_rewards.append(train_episode_rewards[t])
                        done_episodes_costs1.append(train_episode_costs1[t])
                        done_episodes_costs2.append(train_episode_costs2[t])
                        done_episodes_local_costs_per_agent.append(
                            train_episode_local_costs[t].copy()
                        )

                        train_episode_rewards[t] = 0
                        train_episode_costs1[t] = 0
                        train_episode_costs2[t] = 0
                        train_episode_local_costs[t] = 0

                if getattr(self.all_args, "unified_cost_objective", False):
                    opt_costs1 = np.zeros_like(costs1, dtype=np.float32)
                    opt_local_costs = np.zeros_like(local_costs, dtype=np.float32)
                    if getattr(self.all_args, "merge_local_global_cost", False):
                        opt_costs2 = (costs1 + costs2).astype(np.float32)
                    else:
                        opt_costs2 = np.array(costs2, dtype=np.float32)
                else:
                    opt_costs1, opt_costs2, opt_local_costs = costs1, costs2, local_costs

                data = obs, share_obs, rewards, opt_costs1, opt_costs2, opt_local_costs, dones, infos, \
                    values, actions, action_log_probs, \
                    rnn_states, rnn_states_critic, cost1_preds, rnn_states_cost1, cost2_preds, rnn_states_cost2, \
                    local_cost_preds, rnn_states_local_cost


                self.insert(data)


            if (episode > 0 and episode % 100 == 0) or episode == episodes - 1:

                per_agent_arr = np.array(done_episodes_local_costs_per_agent) \
                    if len(done_episodes_local_costs_per_agent) > 0 else np.zeros((0, self.num_agents))
                data_dict = {
                    "rewards":      np.array(done_episodes_rewards),
                    "global_costs": np.array(done_episodes_costs2),
                    "local_costs_per_agent": per_agent_arr,
                }

                for i in range(self.num_agents):
                    data_dict[f"local_cost_agent{i}"] = per_agent_arr[:, i] if per_agent_arr.shape[0] > 0 else np.array([])
                save_path = os.path.join(self.run_dir, f"training_metrics_{self.all_args.env_name}.mat")
                savemat(save_path, data_dict)


            self.compute()
            train_infos = self.train()

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads


            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            if episode % self.log_interval == 0 and len(done_episodes_rewards) > 0:
                n_recent = min(len(done_episodes_rewards), 100)
                aver_episode_rewards = np.mean(done_episodes_rewards[-n_recent:])
                aver_episode_costs1  = np.mean(done_episodes_costs1[-n_recent:])
                aver_episode_costs2  = np.mean(done_episodes_costs2[-n_recent:])


                recent_per_agent = np.array(done_episodes_local_costs_per_agent[-n_recent:])
                aver_episode_local_costs = recent_per_agent.mean(axis=0) if recent_per_agent.shape[0] > 0 \
                    else np.zeros(self.num_agents)


                self.return_aver_cost1(aver_episode_costs1)
                self.return_aver_cost2(aver_episode_costs2)
                self.return_aver_local_cost(float(aver_episode_local_costs.mean()))


                agent_lc_str = "  ".join(
                    f"Agent{i}: {aver_episode_local_costs[i]:.4f}"
                    for i in range(self.num_agents)
                )
                print(
                    f"[{self.all_args.env_name}] "
                    f"Ep {episode:>6d}/{episodes} | "
                    f"Steps {total_num_steps:>9d} | "
                    f"Reward: {aver_episode_rewards:>8.3f} | "
                    f"Global Cost: {aver_episode_costs2:>8.4f} | "
                    f"Local Costs [{agent_lc_str}]"
                )


            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def return_aver_cost1(self, aver_episode_costs1):
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].return_aver_insert1(aver_episode_costs1)

    def return_aver_cost2(self, aver_episode_costs2):
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].return_aver_insert2(aver_episode_costs2)

    def return_aver_local_cost(self, aver_episode_local_costs):
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].return_aver_insert_local(aver_episode_local_costs)

    def warmup(self):

        obs, share_obs, _ = self.envs.reset()
        if not self.use_centralized_V:
            share_obs = obs

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].share_obs[0] = share_obs[:, agent_id].copy()
            self.buffer[agent_id].obs[0] = obs[:, agent_id].copy()

    @torch.no_grad()
    def collect(self, step):
        value_collector = []
        action_collector = []
        action_log_prob_collector = []
        rnn_state_collector = []
        rnn_state_critic_collector = []

        cost1_preds_collector = []
        rnn_states_cost1_collector = []
        cost2_preds_collector = []
        rnn_states_cost2_collector = []
        local_cost_preds_collector = []
        rnn_states_local_cost_collector = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic, cost1_pred, rnn_state_cost1 \
                , cost2_pred, rnn_state_cost2, local_cost_pred, rnn_state_local_cost = self.trainer[
                agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                             self.buffer[agent_id].obs[step],
                                             self.buffer[agent_id].rnn_states[step],
                                             self.buffer[agent_id].rnn_states_critic[step],
                                             self.buffer[agent_id].masks[step],
                                             rnn_states_cost1=self.buffer[agent_id].rnn_states_cost1[step],
                                             rnn_states_cost2=self.buffer[agent_id].rnn_states_cost2[step],
                                             rnn_states_local_cost=self.buffer[agent_id].rnn_states_local_cost[step])

            value_collector.append(_t2n(value))
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))
            rnn_state_collector.append(_t2n(rnn_state))
            rnn_state_critic_collector.append(_t2n(rnn_state_critic))

            cost1_preds_collector.append(_t2n(cost1_pred))
            rnn_states_cost1_collector.append(_t2n(rnn_state_cost1))
            cost2_preds_collector.append(_t2n(cost2_pred))
            rnn_states_cost2_collector.append(_t2n(rnn_state_cost2))
            local_cost_preds_collector.append(_t2n(local_cost_pred))
            rnn_states_local_cost_collector.append(_t2n(rnn_state_local_cost))

        values = np.array(value_collector).transpose(1, 0, 2)
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)
        rnn_states = np.array(rnn_state_collector).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_state_critic_collector).transpose(1, 0, 2, 3)

        cost1_preds = np.array(cost1_preds_collector).transpose(1, 0, 2)
        rnn_states_cost1 = np.array(rnn_states_cost1_collector).transpose(1, 0, 2, 3)
        cost2_preds = np.array(cost2_preds_collector).transpose(1, 0, 2)
        rnn_states_cost2 = np.array(rnn_states_cost2_collector).transpose(1, 0, 2, 3)
        local_cost_preds = np.array(local_cost_preds_collector).transpose(1, 0, 2)
        rnn_states_local_cost = np.array(rnn_states_local_cost_collector).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, cost1_preds, rnn_states_cost1, \
            cost2_preds, rnn_states_cost2, local_cost_preds, rnn_states_local_cost

    def insert(self, data):
        obs, share_obs, rewards, costs1, costs2, local_costs, dones, infos, \
            values, actions, action_log_probs, rnn_states, rnn_states_critic, cost1_preds, rnn_states_cost1, cost2_preds, rnn_states_cost2, \
            local_cost_preds, rnn_states_local_cost = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states_critic.shape[2:]), dtype=np.float32)

        rnn_states_cost1[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states_cost1.shape[2:]), dtype=np.float32)
        rnn_states_cost2[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states_cost2.shape[2:]), dtype=np.float32)
        rnn_states_local_cost[dones_env == True] = np.zeros(
            ((dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states_local_cost.shape[2:]),
            dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        if not self.use_centralized_V:
            share_obs = obs

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].insert(share_obs[:, agent_id], obs[:, agent_id], rnn_states[:, agent_id],
                                         rnn_states_critic[:, agent_id], actions[:, agent_id],
                                         action_log_probs[:, agent_id],
                                         values[:, agent_id], rewards[:, agent_id], masks[:, agent_id], None,
                                         active_masks[:, agent_id], None,
                                         costs1=costs1[:, agent_id],
                                         cost1_preds=cost1_preds[:, agent_id],
                                         rnn_states_cost1=rnn_states_cost1[:, agent_id],
                                         costs2=costs2[:, agent_id],
                                         cost2_preds=cost2_preds[:, agent_id],
                                         rnn_states_cost2=rnn_states_cost2[:, agent_id],
                                         local_costs=local_costs[:, agent_id],
                                         local_cost_preds=local_cost_preds[:, agent_id],
                                         rnn_states_local_cost=rnn_states_local_cost[:, agent_id])

    @torch.no_grad()
    def eval(self, total_num_steps=0):
        """Evaluate deterministic actions and report daily rewards and costs."""
        eval_episodes = self.render_episodes
        all_eval_rewards = []
        all_eval_costs1 = []
        all_eval_costs2 = []

        print(f"Starting generic evaluation for {eval_episodes} episodes...")

        for episode in range(eval_episodes):
            eval_obs, eval_share_obs, _ = self.envs.reset()

            eval_rnn_states = np.zeros(
                (self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size),
                dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

            eval_episode_rewards = np.zeros(self.n_eval_rollout_threads, dtype=np.float32)
            eval_episode_costs1 = np.zeros(self.n_eval_rollout_threads, dtype=np.float32)
            eval_episode_costs2 = np.zeros(self.n_eval_rollout_threads, dtype=np.float32)

            for step in range(self.episode_length):
                eval_actions_collector = []

                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].prep_rollout()
                    eval_actions, temp_rnn_state = \
                        self.trainer[agent_id].policy.act(eval_obs[:, agent_id],
                                                          eval_rnn_states[:, agent_id],
                                                          eval_masks[:, agent_id],
                                                          deterministic=True)
                    eval_rnn_states[:, agent_id] = _t2n(temp_rnn_state)
                    eval_actions_collector.append(_t2n(eval_actions))

                eval_actions = np.array(eval_actions_collector).transpose(1, 0, 2)
                eval_obs, eval_share_obs, eval_rewards, eval_costs1, eval_costs2, _, eval_dones, _, _ = self.envs.step(eval_actions)


                eval_episode_rewards += np.mean(eval_rewards, axis=1).flatten()
                eval_episode_costs1 += np.mean(eval_costs1, axis=1).flatten()
                eval_episode_costs2 += np.mean(eval_costs2, axis=1).flatten()

                eval_dones_env = np.all(eval_dones, axis=1)
                eval_rnn_states[eval_dones_env == True] = np.zeros(
                    ((eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size),
                    dtype=np.float32)

                eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
                eval_masks[eval_dones_env == True] = np.zeros(((eval_dones_env == True).sum(), self.num_agents, 1),
                                                              dtype=np.float32)

            all_eval_rewards.extend(eval_episode_rewards)
            all_eval_costs1.extend(eval_episode_costs1)
            all_eval_costs2.extend(eval_episode_costs2)

            for t in range(self.n_eval_rollout_threads):
                print(f"Eval Episode {episode}, Thread {t} | Reward: {eval_episode_rewards[t]:.4f} "
                      f"| Cost1: {eval_episode_costs1[t]:.4f} | Cost2: {eval_episode_costs2[t]:.4f}")

        if len(all_eval_rewards) > 0:
            avg_reward = np.mean(all_eval_rewards)
            avg_cost1 = np.mean(all_eval_costs1)
            avg_cost2 = np.mean(all_eval_costs2)
            print(f"Evaluation Finished. Avg Reward: {avg_reward:.4f}, "
                  f"Avg Cost1: {avg_cost1:.4f}, Avg Cost2: {avg_cost2:.4f}\n")

