import numpy as np
import pytest

from power_envs import CASES

DIMS = {"C1": (3, 8, 4), "C2": (5, 8, 4), "C3": (12, 11, 5)}


@pytest.mark.parametrize("case", CASES)
def test_episode_shapes_and_costs(case):
    np.random.seed(0)
    env = CASES[case]()
    n, obs_dim, act_dim = DIMS[case]
    obs = env.reset()
    assert len(env.agent_ids) == n and env.obs_dim == obs_dim and env.act_dim == act_dim
    assert all(o.shape == (obs_dim,) for o in obs.values())
    rng = np.random.RandomState(1)
    for t in range(24):
        obs, rewards, done, info = env.step({i: rng.rand(act_dim) for i in env.agent_ids})
        assert done == (t == 23)
        assert set(rewards) == set(env.agent_ids)


@pytest.mark.parametrize("case", CASES)
def test_profiles_depend_only_on_the_seed(case):
    profiles = []
    for _ in range(2):
        np.random.seed(20261005)
        env = CASES[case]()
        env.reset()
        loads = env.current_load_profiles if case == "C3" else {0: env.current_load_profile}
        profiles.append((env.current_pv_profile.copy(), {k: v.copy() for k, v in loads.items()}))
    assert np.array_equal(profiles[0][0], profiles[1][0])
    assert all(np.array_equal(profiles[0][1][k], profiles[1][1][k]) for k in profiles[0][1])
