"""Parallel rollout workers (adapted from OpenAI Baselines through MAPPO)."""
from multiprocessing import Pipe, Process

import numpy as np


class CloudpickleWrapper:
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            result = env.step(data)
            if np.all(result[5]):
                env.reset()
            remote.send(result)
        elif cmd == "reset":
            remote.send(env.reset())
        elif cmd == "dims":
            remote.send(env.dims)
        elif cmd == "close":
            remote.close()
            break


class SubprocVecEnv:
    """One environment per subprocess."""

    def __init__(self, env_fns):
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in env_fns])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for work_remote, remote, env_fn in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(("dims", None))
        self.dims = self.remotes[0].recv()

    def reset(self):
        for remote in self.remotes:
            remote.send(("reset", None))
        obs, share_obs = zip(*[remote.recv() for remote in self.remotes])
        return np.stack(obs), np.stack(share_obs)

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        results = zip(*[remote.recv() for remote in self.remotes])
        return tuple(np.stack(x) for x in results)

    def close(self):
        for remote in self.remotes:
            remote.send(("close", None))
        for p in self.ps:
            p.join()


class DummyVecEnv:
    """Environments in the main process."""

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        self.dims = self.envs[0].dims

    def reset(self):
        obs, share_obs = map(np.array, zip(*[env.reset() for env in self.envs]))
        return obs, share_obs

    def step(self, actions):
        results = [env.step(a) for a, env in zip(actions, self.envs)]
        obs, share_obs, rewards, global_costs, local_costs, dones = map(np.array, zip(*results))
        for i, done in enumerate(dones):
            if np.all(done):
                obs[i], share_obs[i] = self.envs[i].reset()
        return obs, share_obs, rewards, global_costs, local_costs, dones

    def close(self):
        pass
