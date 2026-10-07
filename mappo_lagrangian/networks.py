"""Actor and critic networks."""
import copy

import numpy as np
import torch
import torch.nn as nn


def init(module, weight_init, bias_init, gain=1):
    weight_init(module.weight.data, gain=gain)
    bias_init(module.bias.data)
    return module


def check(x):
    return torch.from_numpy(x) if type(x) == np.ndarray else x


class MLPLayer(nn.Module):
    """(layer_N + 1) x [Linear, ReLU, LayerNorm] with orthogonal initialization."""

    def __init__(self, input_dim, hidden_size, layer_N):
        super().__init__()
        self._layer_N = layer_N
        gain = nn.init.calculate_gain("relu")

        def init_(m):
            return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)

        self.fc1 = nn.Sequential(init_(nn.Linear(input_dim, hidden_size)), nn.ReLU(), nn.LayerNorm(hidden_size))
        self.fc_h = nn.Sequential(init_(nn.Linear(hidden_size, hidden_size)), nn.ReLU(), nn.LayerNorm(hidden_size))
        self.fc2 = nn.ModuleList([copy.deepcopy(self.fc_h) for _ in range(layer_N)])

    def forward(self, x):
        x = self.fc1(x)
        for i in range(self._layer_N):
            x = self.fc2[i](x)
        return x


class MLPBase(nn.Module):
    """LayerNorm on the input followed by the MLP."""

    def __init__(self, input_dim, hidden_size, layer_N):
        super().__init__()
        self.feature_norm = nn.LayerNorm(input_dim)
        self.mlp = MLPLayer(input_dim, hidden_size, layer_N)

    def forward(self, x):
        return self.mlp(self.feature_norm(x))


class FixedNormal(torch.distributions.Normal):
    def log_probs(self, actions):
        return super().log_prob(actions)

    def mode(self):
        return self.mean


class DiagGaussian(nn.Module):
    """Linear mean and state-independent std = std_y_coef * sigmoid(log_std / std_x_coef)."""

    def __init__(self, num_inputs, num_outputs, gain, std_x_coef, std_y_coef):
        super().__init__()
        self.std_x_coef = std_x_coef
        self.std_y_coef = std_y_coef
        self.fc_mean = init(nn.Linear(num_inputs, num_outputs), nn.init.orthogonal_,
                            lambda x: nn.init.constant_(x, 0), gain)
        self.log_std = nn.Parameter(torch.ones(num_outputs) * std_x_coef)

    def forward(self, x):
        action_std = torch.sigmoid(self.log_std / self.std_x_coef) * self.std_y_coef
        return FixedNormal(self.fc_mean(x), action_std)


class ACTLayer(nn.Module):
    def __init__(self, inputs_dim, action_dim, gain, std_x_coef, std_y_coef):
        super().__init__()
        self.action_out = DiagGaussian(inputs_dim, action_dim, gain, std_x_coef, std_y_coef)

    def forward(self, x):
        return self.action_out(x)


class Actor(nn.Module):
    """Gaussian policy on the agent's local observation."""

    def __init__(self, args, obs_dim, act_dim, device=torch.device("cpu")):
        super().__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.base = MLPBase(obs_dim, args.hidden_size, args.layer_N)
        self.act = ACTLayer(args.hidden_size, act_dim, args.gain, args.std_x_coef, args.std_y_coef)
        self.to(device)

    def forward(self, obs, deterministic=False):
        dist = self.act(self.base(check(obs).to(**self.tpdv)))
        actions = dist.mode() if deterministic else dist.sample()
        return actions, dist.log_probs(actions)

    def evaluate_actions(self, obs, actions, active_masks):
        dist = self.act(self.base(check(obs).to(**self.tpdv)))
        log_probs = dist.log_probs(check(actions).to(**self.tpdv))
        active_masks = check(active_masks).to(**self.tpdv)
        entropy = (dist.entropy() * active_masks).sum() / active_masks.sum()
        return log_probs, entropy


class Critic(nn.Module):
    """Scalar value function."""

    def __init__(self, args, input_dim, device=torch.device("cpu")):
        super().__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.base = MLPBase(input_dim, args.hidden_size, args.layer_N)
        self.v_out = init(nn.Linear(args.hidden_size, 1), nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.to(device)

    def forward(self, x):
        return self.v_out(self.base(check(x).to(**self.tpdv)))
