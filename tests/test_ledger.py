import numpy as np
import pytest

from evaluation.ledger import evaluate
from power_envs import CASES


class RandomController:
    def __init__(self, seed=0):
        self.rng = np.random.RandomState(seed)

    def __call__(self, env, obs):
        return {i: self.rng.rand(env.act_dim) for i in env.agent_ids}, {}


def rewards_from_costs(case, env, costs):
    """Shared reward implied by the hourly operating costs."""
    if case == "C3":
        return [-0.008 * c for c in costs[:-1]]
    hours = range(24)
    if case == "C1":
        demand = sum(env.dso_total_base_load.values())
        return [0.003 * (env.grid_tou[t] * demand * env.current_load_profile[t] - costs[t]) for t in hours]
    baseline = [env.grid_tou[max(t - 1, 0)] * sum(sum(env.mg_base_p[i]) * env.mg_load_profiles[i][max(t - 1, 0)]
                                                 for i in env.agent_ids) for t in hours]
    return [0.008 * (baseline[t] - costs[t]) for t in hours]


@pytest.mark.parametrize("case", CASES)
def test_cost_matches_reward(case):
    np.random.seed(20261005)
    env = CASES[case]()
    obs = env.reset()
    rewards = []
    step = env.step

    def recording_step(actions):
        out = step(actions)
        rewards.append(np.mean(list(out[1].values())))
        return out

    env.step = recording_step
    result = evaluate(case, 20261005, "random", env=env, obs=obs, controller=RandomController())
    costs = [s["cost"] for s in result["steps"]]
    expected = rewards_from_costs(case, env, costs)
    np.testing.assert_allclose(rewards[:len(expected)], expected, rtol=1e-6, atol=1e-6)
