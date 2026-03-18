"""
SAC Trainer — Soft Actor-Critic training loop with PER and Curriculum.

Integrates:
- PER Buffer for prioritized replay
- Curriculum Manager for progressive difficulty
- Twin Q-Critic with target networks
- Automatic entropy tuning (temperature α)
- Gradient clipping
- Periodic evaluation
- Checkpoint saving

All hyperparameters from train_hyperparams.yaml.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from models.actor_critic import Actor, TwinQCritic
from training_pipeline.per_buffer import PERBuffer

logger = logging.getLogger(__name__)


@dataclass
class TrainMetrics:
    """Metrics tracked during training."""
    step: int = 0
    episode: int = 0
    actor_loss: float = 0.0
    critic_loss: float = 0.0
    alpha_loss: float = 0.0
    alpha_value: float = 0.0
    mean_reward: float = 0.0
    mean_episode_length: float = 0.0
    curriculum_stage: str = ""
    buffer_size: int = 0

    def to_dict(self) -> dict[str, float]:
        return {
            "step": self.step,
            "episode": self.episode,
            "actor_loss": self.actor_loss,
            "critic_loss": self.critic_loss,
            "alpha_loss": self.alpha_loss,
            "alpha_value": self.alpha_value,
            "mean_reward": self.mean_reward,
            "mean_episode_length": self.mean_episode_length,
            "buffer_size": self.buffer_size,
        }


class SACTrainer:
    """
    Soft Actor-Critic training loop.

    Manages the full training cycle including:
    - Environment interaction and data collection
    - SAC policy optimization with PER
    - Target network soft updates
    - Automatic entropy tuning
    - Evaluation and checkpointing
    """

    def __init__(
        self,
        actor: Actor,
        critic: TwinQCritic,
        state_dim: int,
        action_dim: int = 4,
        config: dict | None = None,
        device: str = "cpu",
    ) -> None:
        """
        Args:
            actor: SAC Actor network
            critic: SAC Twin Q-Critic
            state_dim: Observation dimension
            action_dim: Action dimension
            config: Dict from train_hyperparams.yaml
            device: PyTorch device
        """
        config = config or {}
        self.device = torch.device(device)

        # Networks
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

        # Target critic (copy, no gradient)
        self.target_critic = TwinQCritic(
            state_dim, action_dim,
            config.get("critic_hidden_dims", [256, 256])
        ).to(self.device)
        self._hard_update_target()

        # Hyperparameters
        self.lr = config.get("learning_rate", 3e-4)
        self.gamma = config.get("gamma", 0.99)
        self.tau = config.get("tau", 0.005)
        self.batch_size = config.get("batch_size", 256)
        self.warmup_steps = config.get("warmup_steps", 10_000)
        self.grad_clip = config.get("gradient_clip_norm", 1.0)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr)

        # Automatic entropy tuning
        target_entropy = config.get("target_entropy", "auto")
        if target_entropy == "auto":
            self.target_entropy = -float(action_dim)
        else:
            self.target_entropy = float(target_entropy)

        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_optimizer = optim.Adam(
            [self.log_alpha], lr=config.get("alpha_lr", 3e-4)
        )

        # PER Buffer
        self.buffer = PERBuffer(
            capacity=config.get("buffer_size", 1_000_000),
            alpha=config.get("per_alpha", 0.6),
            beta_start=config.get("per_beta_start", 0.4),
            beta_frames=config.get("per_beta_frames", 500_000),
            state_dim=state_dim,
            action_dim=action_dim,
        )

        # Tracking
        self.total_steps = 0
        self.total_episodes = 0
        self._episode_rewards: list[float] = []

    @property
    def alpha(self) -> torch.Tensor:
        """Current entropy temperature."""
        return self.log_alpha.exp()

    def _hard_update_target(self) -> None:
        """Copy critic weights to target critic."""
        self.target_critic.load_state_dict(self.critic.state_dict())

    def _soft_update_target(self) -> None:
        """Polyak averaging: target = tau * critic + (1-tau) * target."""
        for target_param, param in zip(
            self.target_critic.parameters(), self.critic.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1.0 - self.tau) * target_param.data
            )

    def select_action(
        self, state: np.ndarray, deterministic: bool = False
    ) -> np.ndarray:
        """
        Select action given state.

        Args:
            state: (state_dim,) numpy array
            deterministic: Use mean action (no sampling)

        Returns:
            (action_dim,) numpy array
        """
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _ = self.actor(state_t, deterministic=deterministic)
            return action.squeeze(0).cpu().numpy()

    def update(self) -> TrainMetrics:
        """
        Perform one SAC update step.

        Returns:
            TrainMetrics with losses
        """
        if self.buffer.size < self.batch_size:
            return TrainMetrics(step=self.total_steps)

        # Sample from PER buffer
        (
            states, actions, rewards, next_states, dones,
            is_weights, tree_indices
        ) = self.buffer.sample(self.batch_size)

        # Convert to tensors
        s = torch.FloatTensor(states).to(self.device)
        a = torch.FloatTensor(actions).to(self.device)
        r = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        s_next = torch.FloatTensor(next_states).to(self.device)
        d = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        w = torch.FloatTensor(is_weights).unsqueeze(1).to(self.device)

        # ─── Critic Update ───
        with torch.no_grad():
            # Sample next action from current policy
            next_action, next_log_prob = self.actor(s_next)

            # Target Q-value
            target_q = self.target_critic.min_q(s_next, next_action)
            target_value = r + (1.0 - d) * self.gamma * (
                target_q - self.alpha * next_log_prob
            )

        # Current Q-values
        q1, q2 = self.critic(s, a)

        # TD-errors for PER priority update
        td_error1 = (q1 - target_value).detach().cpu().numpy().squeeze()
        td_error2 = (q2 - target_value).detach().cpu().numpy().squeeze()
        td_errors = (np.abs(td_error1) + np.abs(td_error2)) / 2.0

        # Weighted MSE loss (IS weights from PER)
        critic_loss = (
            w * F.mse_loss(q1, target_value, reduction="none")
            + w * F.mse_loss(q2, target_value, reduction="none")
        ).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), self.grad_clip)
        self.critic_optimizer.step()

        # Update PER priorities
        self.buffer.update_priorities(tree_indices, td_errors)

        # ─── Actor Update ───
        new_action, log_prob = self.actor(s)
        q_new = self.critic.min_q(s, new_action)
        actor_loss = (self.alpha.detach() * log_prob - q_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), self.grad_clip)
        self.actor_optimizer.step()

        # ─── Alpha (Temperature) Update ───
        alpha_loss = -(
            self.log_alpha * (log_prob + self.target_entropy).detach()
        ).mean()

        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()

        # ─── Soft update target ───
        self._soft_update_target()

        return TrainMetrics(
            step=self.total_steps,
            actor_loss=actor_loss.item(),
            critic_loss=critic_loss.item(),
            alpha_loss=alpha_loss.item(),
            alpha_value=self.alpha.item(),
            buffer_size=self.buffer.size,
        )

    def save_checkpoint(self, path: Path | str) -> None:
        """Save full training state."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "log_alpha": self.log_alpha.data,
            "alpha_optimizer": self.alpha_optimizer.state_dict(),
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
        }, path)
        logger.info("Saved checkpoint to %s (step=%d)", path, self.total_steps)

    def load_checkpoint(self, path: Path | str) -> None:
        """Load full training state."""
        path = Path(path)
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.target_critic.load_state_dict(checkpoint["target_critic"])
        self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        self.log_alpha.data = checkpoint["log_alpha"]
        self.alpha_optimizer.load_state_dict(checkpoint["alpha_optimizer"])
        self.total_steps = checkpoint["total_steps"]
        self.total_episodes = checkpoint["total_episodes"]
        logger.info("Loaded checkpoint from %s (step=%d)", path, self.total_steps)
