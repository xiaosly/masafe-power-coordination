"""Multi-agent training interface of the power-system cases."""
import numpy as np

from power_envs import CASES

COST_KEYS = {"C1": ("dso_voltage_violations", "line_overload"),
             "C2": ("local_voltage_violations", "global_voltage_violation"),
             "C3": ("local_violations", "global_carbon_violation")}


class MARLEnv:
    """Maps [-1, 1]^d policy outputs to the [0, 1]^d set-points of the environment.

    share_obs: concatenation of all agents' observations (input of the reward and global-cost critics).
    """

    def __init__(self, case):
        self.case = case
        self.env = CASES[case]()
        self.agent_keys = self.env.agent_ids
        self.n = self.env.n_agents
        self.dims = (self.env.obs_dim, self.n * self.env.obs_dim, self.env.act_dim, self.n)

    def shared(self, obs_n):
        state = np.asarray(obs_n, dtype=np.float32).reshape(-1).tolist()
        return [state.copy() for _ in range(self.n)]

    def reset(self):
        obs = self.env.reset()
        obs_n = [obs[k].tolist() for k in self.agent_keys]
        return obs_n, self.shared(obs_n)

    def step(self, action_n):
        actions = {k: np.clip((np.array(action_n[i]) + 1.0) / 2.0, 0.0, 1.0) for i, k in enumerate(self.agent_keys)}
        obs, rewards, done, info = self.env.step(actions)
        obs_n = [obs[k].tolist() for k in self.agent_keys]
        local_key, global_key = COST_KEYS[self.case]
        reward_n = [[rewards[k]] for k in self.agent_keys]
        global_n = [[info[global_key]]] * self.n
        local_n = [[info[local_key][k]] for k in self.agent_keys]
        return obs_n, self.shared(obs_n), reward_n, global_n, local_n, [done] * self.n
