"""Training loop: rollouts, multiplier updates and sequential agent updates."""
import json
import time

import numpy as np
import torch
from scipy.io import savemat

from mappo_lagrangian.runner import Runner, _t2n
from safe_marl.trainer import LocalGlobalTrainer, joint_ratio

METHODS = ("proposed", "mappo_pf", "mappo_vanilla")


def constraint_objective(rewards, global_cost, local_cost, method):
    """Reward and cost signals used for training; the logged costs are the physical ones."""
    if method == "proposed":
        return rewards, global_cost, local_cost
    if method == "mappo_pf":
        return rewards - local_cost - global_cost, np.zeros_like(global_cost), np.zeros_like(local_cost)
    return rewards, np.zeros_like(global_cost), np.zeros_like(local_cost)


class LocalGlobalRunner(Runner):
    def __init__(self, args, envs, device, run_dir):
        super().__init__(args, envs, device, run_dir)
        self.trainer = [LocalGlobalTrainer(args, policy, device=device) for policy in self.policy]
        self.method = args.method
        self.metric_history = []
        self.raw_reward_history = []
        self.update_history = []

    @torch.no_grad()
    def compute(self):
        for trainer, buffer in zip(self.trainer, self.buffer):
            policy = trainer.policy
            buffer.compute_returns(buffer.rewards, buffer.value_preds, buffer.returns,
                                   _t2n(policy.critic(buffer.share_obs[-1])), trainer.value_normalizer)
            buffer.compute_returns(buffer.global_costs, buffer.global_preds, buffer.global_returns,
                                   _t2n(policy.cost2_critic(buffer.share_obs[-1])), trainer.global_normalizer)
            buffer.compute_returns(buffer.local_costs, buffer.local_preds, buffer.local_returns,
                                   _t2n(policy.local_cost_critic(buffer.obs[-1])), trainer.local_normalizer)

    @torch.no_grad()
    def action_log_probs(self, trainer, buffer):
        log_probs, _ = trainer.policy.actor.evaluate_actions(
            buffer.obs[:-1].reshape(-1, buffer.obs.shape[-1]),
            buffer.actions.reshape(-1, buffer.actions.shape[-1]),
            buffer.active_masks[:-1].reshape(-1, 1))
        return log_probs

    def train(self):
        dual_costs = [trainer.update_duals(buffer) for trainer, buffer in zip(self.trainer, self.buffer)]
        factor = np.ones((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)
        train_infos = [None] * self.num_agents
        # agents are updated one after another in a random order; each update is weighted
        # by the probability ratios of the agents updated before it
        for agent in torch.randperm(self.num_agents).tolist():
            trainer, buffer = self.trainer[agent], self.buffer[agent]
            buffer.update_factor(factor)
            old_log_probs = self.action_log_probs(trainer, buffer)
            info = trainer.train(buffer)
            new_log_probs = self.action_log_probs(trainer, buffer)
            factor *= joint_ratio(new_log_probs, old_log_probs).cpu().numpy().reshape(
                self.episode_length, self.n_rollout_threads, 1)
            info.update(local_episode_cost=dual_costs[agent][0], global_episode_cost=dual_costs[agent][1])
            train_infos[agent] = info
            buffer.after_update()
        return train_infos

    def save_metrics(self):
        rewards = np.asarray([row[0] for row in self.metric_history])
        global_costs = np.asarray([row[1] for row in self.metric_history])
        locals_ = np.asarray([row[2] for row in self.metric_history])
        data = dict(rewards=rewards, global_costs=global_costs, local_costs_per_agent=locals_,
                    raw_rewards=np.asarray(self.raw_reward_history))
        for agent in range(self.num_agents):
            data[f"local_cost_agent{agent}"] = locals_[:, agent]
        for metric in ("lambda_local", "lambda_global", "reward_adv_std", "local_penalty_adv_std",
                       "global_penalty_adv_std", "approximate_kl", "local_episode_cost", "global_episode_cost"):
            data[metric] = np.asarray([[info[metric] for info in row] for row in self.update_history])
        data["update_iterations"] = np.arange(len(self.update_history))
        savemat(self.run_dir / f"training_metrics_{self.args.case}.mat", data)

    def save_checkpoint(self, iteration):
        self.save()
        snapshot = self.run_dir / "checkpoints" / f"iteration_{iteration:05d}"
        snapshot.mkdir(parents=True, exist_ok=True)
        states = []
        for agent, trainer in enumerate(self.trainer):
            policy = trainer.policy
            torch.save(policy.actor.state_dict(), snapshot / f"actor_agent{agent}.pt")
            state = dict(lambda_local=trainer.lambda_local, lambda_global=trainer.lambda_global,
                         actor=policy.actor.state_dict(), actor_optimizer=policy.actor_optimizer.state_dict())
            for name, model, optimizer, normalizer in (
                ("reward", policy.critic, policy.critic_optimizer, trainer.value_normalizer),
                ("global", policy.cost2_critic, policy.cost2_optimizer, trainer.global_normalizer),
                ("local", policy.local_cost_critic, policy.local_cost_optimizer, trainer.local_normalizer),
            ):
                state[name] = dict(critic=model.state_dict(), optimizer=optimizer.state_dict(),
                                   normalizer=normalizer.state_dict())
            states.append(state)
        torch.save(dict(iteration=iteration, agents=states, torch_rng_state=torch.get_rng_state(),
                        numpy_rng_state=np.random.get_state()), self.save_dir / "training_state.pt")

    def run(self):
        self.warmup()
        iterations = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        started = time.time()
        for iteration in range(iterations):
            episode_reward = np.zeros(self.n_rollout_threads)
            raw_episode_reward = np.zeros(self.n_rollout_threads)
            episode_global = np.zeros(self.n_rollout_threads)
            episode_local = np.zeros((self.n_rollout_threads, self.num_agents))
            for step in range(self.episode_length):
                values, actions, log_probs, global_preds, local_preds = self.collect(step)
                obs, share_obs, rewards, global_costs, local_costs, dones = self.envs.step(actions)
                raw_episode_reward += np.mean(rewards, axis=1).ravel()
                episode_global += np.mean(global_costs, axis=1).ravel()
                episode_local += local_costs.reshape(self.n_rollout_threads, self.num_agents)
                rewards, opt_global, opt_local = constraint_objective(rewards, global_costs, local_costs, self.method)
                episode_reward += np.mean(rewards, axis=1).ravel()
                self.insert(obs, share_obs, rewards, opt_global, opt_local, dones,
                            values, actions, log_probs, global_preds, local_preds)
            for thread in range(self.n_rollout_threads):
                self.metric_history.append((episode_reward[thread], episode_global[thread],
                                            episode_local[thread].copy()))
                self.raw_reward_history.append(raw_episode_reward[thread])
            self.compute()
            self.update_history.append(self.train())
            if iteration % self.log_interval == 0 or iteration == iterations - 1:
                recent = self.metric_history[-100:]
                status = dict(method=self.method, iteration=iteration, total_iterations=iterations,
                              env_steps=(iteration + 1) * self.episode_length * self.n_rollout_threads,
                              reward=float(np.mean([row[0] for row in recent])),
                              global_cost=float(np.mean([row[1] for row in recent])),
                              local_cost_total=float(np.mean([sum(row[2]) for row in recent])),
                              lambda_local=[trainer.lambda_local for trainer in self.trainer],
                              lambda_global=self.trainer[0].lambda_global,
                              elapsed_seconds=time.time() - started)
                print(f"[{self.args.case}] {json.dumps(status)}", flush=True)
                (self.run_dir / "status.json").write_text(json.dumps(status, indent=2))
                self.writter.add_scalar("train/reward", status["reward"], status["env_steps"])
                self.save_metrics()
            if iteration % self.save_interval == 0 or iteration == iterations - 1:
                self.save_checkpoint(iteration)
        self.writter.close()
