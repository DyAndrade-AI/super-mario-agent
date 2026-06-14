"""
Dueling DQN network for discrete-action visual control.
"""
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from .cnn_backbone import create_cnn_backbone


class NoisyLinear(nn.Module):
    """Factorized Gaussian noisy linear layer for train-time exploration."""

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        mu_range = 1.0 / self.in_features ** 0.5
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / self.in_features ** 0.5)
        self.bias_sigma.data.fill_(self.std_init / self.out_features ** 0.5)

    @staticmethod
    def _scale_noise(size: int, device: torch.device) -> torch.Tensor:
        noise = torch.randn(size, device=device)
        return noise.sign().mul_(noise.abs().sqrt_())

    def reset_noise(self):
        epsilon_in = self._scale_noise(self.in_features, self.weight_mu.device)
        epsilon_out = self._scale_noise(self.out_features, self.weight_mu.device)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu

        return F.linear(x, weight, bias)


def _linear(
    in_features: int,
    out_features: int,
    use_noisy: bool,
    noisy_std_init: float,
) -> nn.Module:
    if use_noisy:
        return NoisyLinear(in_features, out_features, std_init=noisy_std_init)
    return nn.Linear(in_features, out_features)


class DuelingDQNNetwork(nn.Module):
    """
    CNN encoder plus dueling value/advantage heads.

    Q(s, a) = V(s) + A(s, a) - mean_a A(s, a)
    """

    def __init__(
        self,
        observation_space,
        action_space,
        cnn_features: Optional[list] = None,
        hidden_dim: int = 512,
        dueling_hidden_dim: int = 512,
        use_noisy_nets: bool = True,
        noisy_std_init: float = 0.5,
        use_batch_norm: bool = False,
        use_layer_norm: bool = True,
        init_scale: float = 1.0,
    ):
        super().__init__()

        if cnn_features is None:
            cnn_features = [32, 64, 64]

        obs_shape = observation_space.shape
        if len(obs_shape) != 3:
            raise ValueError(f"DQN espera observaciones 3D, recibio shape={obs_shape}")

        input_channels = obs_shape[-1] if obs_shape[-1] in (1, 3, 4) else obs_shape[0]
        self.num_actions = action_space.n
        self.use_noisy_nets = use_noisy_nets

        self.cnn = create_cnn_backbone(
            input_channels=input_channels,
            feature_dims=cnn_features,
            use_batch_norm=use_batch_norm,
            use_layer_norm=use_layer_norm,
        )

        self.feature_layer = nn.Sequential(
            nn.Linear(self.cnn.output_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity(),
        )

        self.value_stream = nn.Sequential(
            _linear(hidden_dim, dueling_hidden_dim, use_noisy_nets, noisy_std_init),
            nn.ReLU(),
            nn.LayerNorm(dueling_hidden_dim) if use_layer_norm else nn.Identity(),
            _linear(dueling_hidden_dim, 1, use_noisy_nets, noisy_std_init),
        )
        self.advantage_stream = nn.Sequential(
            _linear(hidden_dim, dueling_hidden_dim, use_noisy_nets, noisy_std_init),
            nn.ReLU(),
            nn.LayerNorm(dueling_hidden_dim) if use_layer_norm else nn.Identity(),
            _linear(dueling_hidden_dim, self.num_actions, use_noisy_nets, noisy_std_init),
        )

        self._init_weights(init_scale)

    def _init_weights(self, scale: float):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=scale)
                nn.init.constant_(module.bias, 0.0)
            elif isinstance(module, nn.Conv2d):
                nn.init.orthogonal_(module.weight, gain=scale)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.cnn(obs)
        features = features.reshape(features.shape[0], -1)
        features = self.feature_layer(features)

        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        return value + advantage - advantage.mean(dim=1, keepdim=True)

    def reset_noise(self):
        for module in self.modules():
            if isinstance(module, NoisyLinear):
                module.reset_noise()

    @torch.no_grad()
    def get_action_deterministic(self, obs: torch.Tensor) -> torch.Tensor:
        return self(obs).argmax(dim=1)


def create_dueling_dqn(observation_space, action_space, **kwargs) -> DuelingDQNNetwork:
    """Factory for the Q-network used by Dueling DQN."""
    return DuelingDQNNetwork(observation_space, action_space, **kwargs)
