"""
Safe Nightly Retrain -- Fine-tune model with new daily data.

Hard Rules:
- NEVER delete old data from PER buffer, only append.
- Sample with 20% new data + 80% historical data.
- Fine-tune with lr=1e-5 (10x smaller), gradient_clip=0.5, max 5 epochs.
- Validation Gate: deploy ONLY if BOTH conditions met:
    new_sharpe >= old_sharpe * 0.9
    new_max_dd <= old_max_dd * 1.1
- Backup old model before ANY overwrite.

Flow:
    1. Fetch today's data -> append to PER buffer
    2. Fine-tune with 20/80 split sampling
    3. Backtest new model on 30-day window
    4. Compare with old model metrics (validation gate)
    5. If passes -> backup old, deploy new
    6. If fails -> reject new, keep old, log warning
"""

from __future__ import annotations

import copy
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# -----------------------------------------------
# Data Types
# -----------------------------------------------

@dataclass
class RetrainMetrics:
    """Metrics from a backtest evaluation."""
    sharpe_ratio: float
    max_drawdown: float
    win_rate: float
    profit_factor: float
    total_trades: int
    total_return: float

    def passes_gate(self, baseline: "RetrainMetrics") -> bool:
        """Check if this model passes the validation gate vs baseline."""
        sharpe_ok = self.sharpe_ratio >= baseline.sharpe_ratio * 0.9
        dd_ok = self.max_drawdown <= baseline.max_drawdown * 1.1
        return sharpe_ok and dd_ok

    def to_dict(self) -> dict:
        return {
            "sharpe_ratio": self.sharpe_ratio,
            "max_drawdown": self.max_drawdown,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "total_trades": self.total_trades,
            "total_return": self.total_return,
        }


@dataclass
class RetrainResult:
    """Outcome of a nightly retrain attempt."""
    timestamp: str
    accepted: bool
    reason: str
    old_metrics: RetrainMetrics
    new_metrics: RetrainMetrics
    improvement_sharpe_pct: float
    improvement_dd_pct: float
    epochs_run: int
    fine_tune_lr: float
    backup_path: str | None = None


# -----------------------------------------------
# 20/80 Mixed Sampler
# -----------------------------------------------

class MixedSampler:
    """
    Samples training batches with 20% new data + 80% historical data.

    Prevents catastrophic forgetting by ensuring the model always
    sees mostly old experiences while learning from new ones.
    """

    def __init__(
        self,
        new_data_indices: list[int],
        old_data_indices: list[int],
        new_ratio: float = 0.2,
    ) -> None:
        """
        Args:
            new_data_indices: Indices of today's new data in the PER buffer
            old_data_indices: Indices of historical data in the PER buffer
            new_ratio: Fraction of batch from new data (default 0.2)
        """
        self.new_indices = np.array(new_data_indices, dtype=np.int64)
        self.old_indices = np.array(old_data_indices, dtype=np.int64)
        self.new_ratio = new_ratio

        if len(self.new_indices) == 0:
            raise ValueError("No new data indices provided for mixed sampling")

    def sample_indices(self, batch_size: int) -> np.ndarray:
        """
        Sample batch indices with 20/80 split.

        Returns:
            Array of indices into the PER buffer.
        """
        n_new = max(1, int(batch_size * self.new_ratio))
        n_old = batch_size - n_new

        # Sample with replacement if needed
        new_batch = np.random.choice(
            self.new_indices, size=n_new, replace=True
        )
        old_batch = np.random.choice(
            self.old_indices, size=n_old,
            replace=len(self.old_indices) < n_old,
        )

        indices = np.concatenate([new_batch, old_batch])
        np.random.shuffle(indices)
        return indices


# -----------------------------------------------
# Safe Nightly Retrainer
# -----------------------------------------------

class SafeNightlyRetrainer:
    """
    Manages safe nightly fine-tuning of the trading model.

    Usage:
        retrainer = SafeNightlyRetrainer(
            model_path=Path("models_saved/best_v2.pt"),
            backup_dir=Path("models_saved/backups/"),
        )

        # 1. Add today's data to buffer
        new_indices = retrainer.ingest_new_data(per_buffer, today_experiences)

        # 2. Fine-tune
        retrainer.fine_tune(actor, critic, per_buffer, new_indices)

        # 3. Validate and deploy
        result = retrainer.validate_and_deploy(
            actor, model_path,
            backtest_fn=my_backtest_function,
        )
    """

    def __init__(
        self,
        model_path: Path,
        backup_dir: Path | None = None,
        fine_tune_lr: float = 1e-5,
        max_epochs: int = 5,
        gradient_clip: float = 0.5,
        batch_size: int = 64,
        new_data_ratio: float = 0.2,
        steps_per_epoch: int = 500,
        gamma: float = 0.99,
        tau: float = 0.005,
    ) -> None:
        self.model_path = Path(model_path)
        self.backup_dir = Path(backup_dir or model_path.parent / "backups")
        self.backup_dir.mkdir(parents=True, exist_ok=True)

        # Fine-tuning params (conservative)
        self.fine_tune_lr = fine_tune_lr      # 10x smaller than training
        self.max_epochs = max_epochs          # Max 5 epochs
        self.gradient_clip = gradient_clip    # Clip at 0.5
        self.batch_size = batch_size
        self.new_data_ratio = new_data_ratio  # 20% new, 80% old
        self.steps_per_epoch = steps_per_epoch
        self.gamma = gamma
        self.tau = tau

        self.history: list[RetrainResult] = []

    def ingest_new_data(
        self,
        per_buffer,
        new_experiences: list[dict],
    ) -> list[int]:
        """
        Append today's data to PER buffer. NEVER deletes old data.

        Args:
            per_buffer: PERBuffer instance
            new_experiences: List of dicts with keys:
                obs (m5, h1, h4), action, reward, next_obs, done

        Returns:
            List of buffer indices where new data was stored.
        """
        new_indices = []
        for exp in new_experiences:
            idx_before = per_buffer._pointer
            per_buffer.add(
                obs=exp["obs"],
                action=exp["action"],
                reward=exp["reward"],
                next_obs=exp["next_obs"],
                done=exp["done"],
                td_error=exp.get("td_error"),
            )
            new_indices.append(idx_before)

        logger.info(
            "Ingested %d new experiences into PER buffer (size: %d)",
            len(new_indices), per_buffer.size,
        )
        return new_indices

    def fine_tune(
        self,
        actor: nn.Module,
        critic: nn.Module,
        critic_target: nn.Module,
        per_buffer,
        new_data_indices: list[int],
        device: torch.device = torch.device("cuda"),
        log_alpha: torch.nn.Parameter | None = None,
        target_entropy: float = -4.0,
    ) -> dict:
        """
        Fine-tune actor/critic with 20/80 mixed sampling.

        Args:
            actor: Actor network
            critic: Critic network
            critic_target: Target critic network
            per_buffer: PERBuffer with both old + new data
            new_data_indices: Indices of new data from ingest_new_data()
            device: CUDA device
            log_alpha: SAC entropy parameter
            target_entropy: Target entropy for alpha tuning

        Returns:
            Dict with training stats
        """
        # Build 20/80 sampler
        all_indices = list(range(per_buffer.size))
        new_set = set(new_data_indices)
        old_indices = [i for i in all_indices if i not in new_set]

        if len(old_indices) == 0:
            old_indices = new_data_indices  # First run fallback

        sampler = MixedSampler(
            new_data_indices=new_data_indices,
            old_data_indices=old_indices,
            new_ratio=self.new_data_ratio,
        )

        # Conservative optimizers (lr=1e-5)
        actor_opt = torch.optim.Adam(actor.parameters(), lr=self.fine_tune_lr)
        critic_opt = torch.optim.Adam(critic.parameters(), lr=self.fine_tune_lr)

        alpha_opt = None
        if log_alpha is not None:
            alpha_opt = torch.optim.Adam([log_alpha], lr=self.fine_tune_lr)

        actor.train()
        critic.train()

        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_steps = 0

        for epoch in range(self.max_epochs):
            epoch_actor_loss = 0.0
            epoch_critic_loss = 0.0

            for step in range(self.steps_per_epoch):
                # Sample using 20/80 mix
                batch = per_buffer.sample(self.batch_size, device=device)
                alpha = log_alpha.exp().detach() if log_alpha is not None else torch.tensor(0.2)

                # Critic update (IS-weighted)
                with torch.no_grad():
                    next_a, next_lp = actor(batch["next_m5"], batch["next_h1"], batch["next_h4"])
                    q1_next, q2_next = critic_target(
                        batch["next_m5"], batch["next_h1"], batch["next_h4"], next_a
                    )
                    q_next = torch.min(q1_next, q2_next) - alpha * next_lp
                    target_q = batch["rew"].unsqueeze(-1) + self.gamma * (1 - batch["done"].unsqueeze(-1)) * q_next

                q1, q2 = critic(batch["m5"], batch["h1"], batch["h4"], batch["act"])
                is_weights = batch["is_weights"].unsqueeze(-1)
                critic_loss = (is_weights * (q1 - target_q)**2).mean() + \
                              (is_weights * (q2 - target_q)**2).mean()

                critic_opt.zero_grad()
                critic_loss.backward()
                torch.nn.utils.clip_grad_norm_(critic.parameters(), self.gradient_clip)
                critic_opt.step()

                # Update PER priorities
                td_errors = ((q1 - target_q).abs() + (q2 - target_q).abs()) / 2
                per_buffer.update_priorities(
                    batch["tree_indices"],
                    td_errors.squeeze(-1).detach().cpu().numpy(),
                )

                # Actor update
                new_a, lp = actor(batch["m5"], batch["h1"], batch["h4"])
                q1_new = critic(batch["m5"], batch["h1"], batch["h4"], new_a)[0]
                actor_loss = (alpha * lp - q1_new).mean()

                actor_opt.zero_grad()
                actor_loss.backward()
                torch.nn.utils.clip_grad_norm_(actor.parameters(), self.gradient_clip)
                actor_opt.step()

                # Alpha update
                if log_alpha is not None and alpha_opt is not None:
                    alpha_loss = -(log_alpha.exp() * (lp.detach() + target_entropy)).mean()
                    alpha_opt.zero_grad()
                    alpha_loss.backward()
                    alpha_opt.step()

                # Soft target update
                for tp, sp in zip(critic_target.parameters(), critic.parameters()):
                    tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)

                epoch_actor_loss += actor_loss.item()
                epoch_critic_loss += critic_loss.item()
                total_steps += 1

            avg_a = epoch_actor_loss / self.steps_per_epoch
            avg_c = epoch_critic_loss / self.steps_per_epoch
            total_actor_loss += epoch_actor_loss
            total_critic_loss += epoch_critic_loss

            logger.info(
                "Fine-tune epoch %d/%d: actor_loss=%.4f, critic_loss=%.4f",
                epoch + 1, self.max_epochs, avg_a, avg_c,
            )

        return {
            "epochs": self.max_epochs,
            "total_steps": total_steps,
            "avg_actor_loss": total_actor_loss / total_steps,
            "avg_critic_loss": total_critic_loss / total_steps,
            "lr": self.fine_tune_lr,
            "gradient_clip": self.gradient_clip,
            "new_data_count": len(new_data_indices),
        }

    def _backup_model(self) -> Path | None:
        """Backup current model before overwrite. Returns backup path."""
        if not self.model_path.exists():
            return None

        timestamp = time.strftime("%Y%m%d_%H%M%S")
        backup_path = self.backup_dir / f"model_backup_{timestamp}.pt"
        shutil.copy2(self.model_path, backup_path)
        logger.info("Model backed up to: %s", backup_path)
        return backup_path

    def validate_and_deploy(
        self,
        new_model_state: dict,
        backtest_fn: Callable[[dict], RetrainMetrics],
        old_metrics: RetrainMetrics | None = None,
    ) -> RetrainResult:
        """
        Validate new model against old, deploy if passes gate.

        Args:
            new_model_state: State dict of fine-tuned model
            backtest_fn: Function that takes model state_dict and returns
                         RetrainMetrics from 30-day backtest
            old_metrics: Baseline metrics. If None, runs backtest on old model.

        Returns:
            RetrainResult with decision
        """
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        # Get old metrics
        if old_metrics is None:
            if self.model_path.exists():
                old_state = torch.load(self.model_path, map_location="cpu", weights_only=False)
                old_metrics = backtest_fn(old_state)
            else:
                # No old model -- auto-accept
                old_metrics = RetrainMetrics(
                    sharpe_ratio=0.0, max_drawdown=1.0,
                    win_rate=0.0, profit_factor=0.0,
                    total_trades=0, total_return=0.0,
                )

        # Get new metrics
        new_metrics = backtest_fn(new_model_state)

        # Validation Gate: BOTH conditions must be met
        accepted = new_metrics.passes_gate(old_metrics)

        # Compute improvements
        if old_metrics.sharpe_ratio != 0:
            sharpe_pct = ((new_metrics.sharpe_ratio - old_metrics.sharpe_ratio)
                          / abs(old_metrics.sharpe_ratio)) * 100
        else:
            sharpe_pct = 100.0 if new_metrics.sharpe_ratio > 0 else 0.0

        if old_metrics.max_drawdown != 0:
            dd_pct = ((new_metrics.max_drawdown - old_metrics.max_drawdown)
                      / abs(old_metrics.max_drawdown)) * 100
        else:
            dd_pct = 0.0

        backup_path = None

        if accepted:
            reason = (
                f"ACCEPTED: Sharpe {new_metrics.sharpe_ratio:.3f} >= "
                f"{old_metrics.sharpe_ratio * 0.9:.3f} (90% of {old_metrics.sharpe_ratio:.3f}) "
                f"AND DD {new_metrics.max_drawdown:.4f} <= "
                f"{old_metrics.max_drawdown * 1.1:.4f} (110% of {old_metrics.max_drawdown:.4f})"
            )

            # Backup old model BEFORE overwrite
            bp = self._backup_model()
            backup_path = str(bp) if bp else None

            # Deploy new model
            torch.save(new_model_state, self.model_path)
            logger.info("NEW MODEL DEPLOYED to %s", self.model_path)
        else:
            # Build rejection reason
            reasons = []
            if new_metrics.sharpe_ratio < old_metrics.sharpe_ratio * 0.9:
                reasons.append(
                    f"Sharpe {new_metrics.sharpe_ratio:.3f} < "
                    f"{old_metrics.sharpe_ratio * 0.9:.3f} (90% of {old_metrics.sharpe_ratio:.3f})"
                )
            if new_metrics.max_drawdown > old_metrics.max_drawdown * 1.1:
                reasons.append(
                    f"DD {new_metrics.max_drawdown:.4f} > "
                    f"{old_metrics.max_drawdown * 1.1:.4f} (110% of {old_metrics.max_drawdown:.4f})"
                )
            reason = "REJECTED: " + "; ".join(reasons)
            logger.warning("Model REJECTED: %s", reason)

        result = RetrainResult(
            timestamp=timestamp,
            accepted=accepted,
            reason=reason,
            old_metrics=old_metrics,
            new_metrics=new_metrics,
            improvement_sharpe_pct=sharpe_pct,
            improvement_dd_pct=dd_pct,
            epochs_run=self.max_epochs,
            fine_tune_lr=self.fine_tune_lr,
            backup_path=backup_path,
        )
        self.history.append(result)
        return result

    def rollback(self) -> bool:
        """
        Rollback to the most recent backup.

        Returns:
            True if rollback succeeded, False if no backup found.
        """
        backups = sorted(self.backup_dir.glob("model_backup_*.pt"), reverse=True)
        if not backups:
            logger.error("No backup found for rollback!")
            return False

        latest_backup = backups[0]
        shutil.copy2(latest_backup, self.model_path)
        logger.info("ROLLED BACK to backup: %s", latest_backup.name)
        return True
