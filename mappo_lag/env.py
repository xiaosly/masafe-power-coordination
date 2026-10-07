"""Power-system interface used by MAPPO-Lag."""
import numpy as np
from power_envs.c1_tso_dso import TSODSOEnv
from power_envs.c2_microgrids import MGVoltageEnv
from power_envs.c3_carbon import CarbonMarketEnv

from mappo_lag.spaces import Box

class _BasePowerWrapper(object):
    """Adapt power-system observations, actions and costs to the trainer interface."""
    def __init__(self, args, base_env_class):


        self.env = base_env_class()
        self.unified_cost_objective = getattr(args, 'unified_cost_objective', False)
        self.constraint_mode = getattr(args, 'constraint_mode', 'lagrangian')
        self.penalty_coeff = getattr(args, 'penalty_coeff', 1.0)

        self.n = self.env.n_agents
        self.n_agents = self.n
        self.agent_keys = self.env.agent_ids


        self.action_space = [Box(low=-1.0, high=1.0, shape=(self.env.act_dim,), dtype=np.float32) for _ in range(self.n)]
        self.observation_space = [Box(low=-np.inf, high=np.inf, shape=(self.env.obs_dim,), dtype=np.float32) for _ in range(self.n)]
        self.share_observation_space = [Box(low=-np.inf, high=np.inf, shape=(self.env.obs_dim,), dtype=np.float32) for _ in range(self.n)]

        self._obs = None

    def step(self, action_n):
        """
        MAPPO-Lagrangian wrapper step:
        Requires return: obs_n, shareobs, reward_n, cost_n1, cost_n2, local_cost_n, done_n, info_n, available_cmds
        """


        actions_dict = {}
        for i, key in enumerate(self.agent_keys):
            actions_dict[key] = np.clip((np.array(action_n[i]) + 1.0) / 2.0, 0.0, 1.0)


        obs_dict, rewards_dict, done_flag, info_dict = self.env.step(actions_dict)
        self._obs = obs_dict


        obs_n = [obs_dict[k].tolist() for k in self.agent_keys]
        shareobs = obs_n

        reward_n = [[rewards_dict[k]] for k in self.agent_keys]


        costs1_n, costs2_n = self._extract_costs(info_dict)


        local_cost_n = costs1_n


        if self.constraint_mode == 'penalty':

            for i in range(self.n):
                reward_n[i][0] -= self.penalty_coeff * (costs1_n[i][0] + costs2_n[i][0])


        done_n = [done_flag] * self.n
        info_n = [{"agent_id": i, "individual_reward": rewards_dict[k]} for i, k in enumerate(self.agent_keys)]

        return obs_n, shareobs, reward_n, costs1_n, costs2_n, local_cost_n, done_n, info_n, self.get_avail_actions()

    def _extract_costs(self, info):
        raise NotImplementedError

    def reset(self, **kwargs):
        obs_dict = self.env.reset()
        self._obs = obs_dict
        obs_n = [obs_dict[k].tolist() for k in self.agent_keys]
        share_obs = obs_n
        return obs_n, share_obs, self.get_avail_actions()

    def get_avail_actions(self):

        return None

    def close(self):
        pass


class PowerEnvS1(_BasePowerWrapper):
    """C2: microgrid and distribution-network voltage constraints."""
    def __init__(self, args):
        super().__init__(args, MGVoltageEnv)

    def _extract_costs(self, info):

        cost1 = [[info['local_voltage_violations'][k]] for k in self.agent_keys]

        global_v = info['global_voltage_violation']
        cost2 = [[global_v]] * self.n
        return cost1, cost2


class PowerEnvS2(_BasePowerWrapper):
    """C1: DSO voltage constraints and the transmission import constraint."""
    def __init__(self, args):
        super().__init__(args, TSODSOEnv)

    def _extract_costs(self, info):

        cost1 = [[info['dso_voltage_violations'][k]] for k in self.agent_keys]

        global_overload = info['line_overload']
        cost2 = [[global_overload]] * self.n
        return cost1, cost2


class PowerEnvS3(_BasePowerWrapper):
    """C3: individual and community carbon constraints."""
    def __init__(self, args):
        super().__init__(args, CarbonMarketEnv)

    def _extract_costs(self, info):

        cost1 = [[info['local_violations'][k]] for k in self.agent_keys]

        global_carbon = info['global_carbon_violation']
        cost2 = [[global_carbon]] * self.n
        return cost1, cost2
