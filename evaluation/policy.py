"""Deterministic policies: the mean action of each agent's actor on its own observation."""
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


class DeterministicPolicy:
    """Loads actor_agent{j}.pt and maps the [-1, 1] mean action to the [0, 1] set-points."""

    def __init__(self, env, model_dir):
        self.ids = env.agent_ids
        self.weights = [torch.load(Path(model_dir) / f"actor_agent{j}.pt", map_location="cpu", weights_only=True)
                        for j in range(env.n_agents)]

    @staticmethod
    def mean(obs, w):
        x = torch.as_tensor(obs, dtype=torch.float32).reshape(1, -1)
        x = F.layer_norm(x, (x.shape[-1],), w["base.feature_norm.weight"], w["base.feature_norm.bias"])
        depth = sum(1 for k in w if k.startswith("base.mlp.fc2.") and k.endswith(".0.weight"))
        for stem in ("base.mlp.fc1", *(f"base.mlp.fc2.{k}" for k in range(depth))):
            x = F.relu(F.linear(x, w[stem + ".0.weight"], w[stem + ".0.bias"]))
            x = F.layer_norm(x, (x.shape[-1],), w[stem + ".2.weight"], w[stem + ".2.bias"])
        return F.linear(x, w["act.action_out.fc_mean.weight"], w["act.action_out.fc_mean.bias"]).numpy()[0]

    def __call__(self, env, obs):
        with torch.inference_mode():
            return {i: np.clip((self.mean(obs[i], w) + 1) / 2, 0, 1)
                    for i, w in zip(self.ids, self.weights)}, {}
