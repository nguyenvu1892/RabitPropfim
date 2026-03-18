"""
Actor-Critic Networks for SAC — Squashed Gaussian policy + twin Q-networks.

Actor: Produces mean + log_std for a squashed Gaussian (tanh) action distribution.
Critic: Twin Q-networks for double Q-learning (reduces overestimation bias).

Designed to work with MultiTimeframeEncoder as the shared backbone.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
EPS = 1e-6


class Actor(nn.Module):
    """
    SAC Actor — Squashed Gaussian policy.

    Output: tanh-squashed action sampled from N(mu, sigma).
    Action space: [confidence, risk_fraction, sl_mult, tp_mult]
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 4,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [256, 256]

        # Build MLP
        layers: list[nn.Module] = []
        prev_dim = state_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim

        self.trunk = nn.Sequential(*layers)

        # Separate heads for mean and log_std
        self.mean_head = nn.Linear(prev_dim, action_dim)
        self.log_std_head = nn.Linear(prev_dim, action_dim)

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            state: (batch, state_dim) — encoded state representation
            deterministic: If True, return mean (no sampling)

        Returns:
            (action, log_prob) — both (batch, action_dim)
        """
        features = self.trunk(state)

        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, LOG_SIG_MIN, LOG_SIG_MAX)
        std = log_std.exp()

        # Squashed Gaussian
        normal = Normal(mean, std)

        if deterministic:
            action_pre = mean
        else:
            action_pre = normal.rsample()  # Reparameterization trick

        # Tanh squash
        action = torch.tanh(action_pre)

        # Log probability with correction for tanh squashing
        # log π(a|s) = log N(u|μ,σ) - Σ log(1 - tanh²(u))
        log_prob = normal.log_prob(action_pre)
        log_prob -= torch.log(1.0 - action.pow(2) + EPS)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob


class QNetwork(nn.Module):
    """Single Q-network: Q(s, a) → scalar."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 4,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        hidden_dims = hidden_dims or [256, 256]

        layers: list[nn.Module] = []
        prev_dim = state_dim + action_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))

        self.net = nn.Sequential(*layers)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Q(s, a) → (batch, 1)"""
        x = torch.cat([state, action], dim=-1)
        return self.net(x)


class TwinQCritic(nn.Module):
    """
    Twin Q-networks for SAC (Clipped Double-Q trick).

    Uses min(Q1, Q2) to reduce overestimation bias.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int = 4,
        hidden_dims: list[int] | None = None,
    ) -> None:
        super().__init__()
        self.q1 = QNetwork(state_dim, action_dim, hidden_dims)
        self.q2 = QNetwork(state_dim, action_dim, hidden_dims)

    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (Q1, Q2) — both (batch, 1)"""
        return self.q1(state, action), self.q2(state, action)

    def min_q(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """min(Q1, Q2) — used as target in SAC update."""
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)
