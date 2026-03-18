"""
Safe Nightly Retrain — Shadow training with validation gate before deployment.

Process:
1. Fork current best model weights
2. Train on latest data (new day's data appended)
3. Evaluate new model on holdout data
4. Compare performance metrics with current best
5. Only promote if new model is significantly better (gate threshold)
6. Register in Model Registry
7. Alert operator via Telegram

If new model fails validation → keep old model → alert operator.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RetrainResult:
    """Result of a nightly retrain attempt."""
    timestamp: str
    old_version: int
    new_version: int | None  # None if rejected
    old_metric: float
    new_metric: float
    improvement_pct: float
    accepted: bool
    reason: str
    details: dict[str, Any]


class SafeRetrainer:
    """
    Manages safe nightly retraining with validation gating.

    Only promotes a retrained model if it meets the improvement threshold
    over the current best model.
    """

    def __init__(
        self,
        config: dict,
        registry: Any = None,  # ModelRegistry
    ) -> None:
        """
        Args:
            config: From train_hyperparams.yaml nightly_retrain section
            registry: ModelRegistry instance for version management
        """
        retrain_config = config.get("nightly_retrain", config)
        self.improvement_threshold = retrain_config.get(
            "improvement_threshold", 0.05
        )  # 5% improvement required
        self.eval_episodes = retrain_config.get("eval_episodes", 50)
        self.max_retrain_steps = retrain_config.get("max_steps", 10_000)

        self.registry = registry
        self.history: list[RetrainResult] = []
        self._alert_callback: Optional[Callable[[str, str], None]] = None

    def set_alert_callback(
        self, callback: Callable[[str, str], None]
    ) -> None:
        """Set callback for sending alerts."""
        self._alert_callback = callback

    def evaluate_model(
        self,
        model_fn: Callable[[np.ndarray], np.ndarray],
        eval_data: np.ndarray,
        env_factory: Callable,
    ) -> dict[str, float]:
        """
        Evaluate a model on holdout data.

        Args:
            model_fn: Function that takes state → action
            eval_data: Evaluation dataset
            env_factory: Callable that creates a fresh environment

        Returns:
            Dict of metrics (eval_reward, sharpe, max_dd, win_rate)
        """
        rewards = []
        max_dds = []
        wins = 0
        total = 0

        for episode in range(self.eval_episodes):
            env = env_factory(eval_data)
            obs, _ = env.reset(seed=episode)
            episode_reward = 0.0
            max_dd = 0.0

            done = False
            while not done:
                action = model_fn(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                max_dd = max(max_dd, info.get("daily_dd", 0))
                done = terminated or truncated

            rewards.append(episode_reward)
            max_dds.append(max_dd)

            # Count trades
            total_trades = info.get("total_trades", 0)
            total += max(total_trades, 1)
            pnl = info.get("balance", 10000) - 10000
            if pnl > 0:
                wins += total_trades

        mean_reward = float(np.mean(rewards))
        std_reward = float(np.std(rewards)) + 1e-8
        sharpe = mean_reward / std_reward

        return {
            "eval_reward": mean_reward,
            "eval_std": std_reward,
            "sharpe": sharpe,
            "max_dd_mean": float(np.mean(max_dds)),
            "win_rate": wins / max(total, 1),
        }

    def try_retrain(
        self,
        old_metrics: dict[str, float],
        new_metrics: dict[str, float],
        new_checkpoint: dict | None = None,
        curriculum_stage: str = "",
        training_steps: int = 0,
    ) -> RetrainResult:
        """
        Compare old vs new model and decide whether to promote.

        Args:
            old_metrics: Metrics of current best model
            new_metrics: Metrics of retrained model
            new_checkpoint: PyTorch state dict of new model (for registry)
            curriculum_stage: Current curriculum stage
            training_steps: Total training steps

        Returns:
            RetrainResult with decision
        """
        now = datetime.now(timezone.utc).isoformat()

        old_metric = old_metrics.get("eval_reward", 0.0)
        new_metric = new_metrics.get("eval_reward", 0.0)

        if old_metric == 0:
            improvement = 1.0 if new_metric > 0 else 0.0
        else:
            improvement = (new_metric - old_metric) / abs(old_metric)

        old_version = self.registry.best_version if self.registry else 0

        # Decision gate
        accepted = improvement >= self.improvement_threshold
        reason = ""

        if accepted:
            reason = f"Improvement {improvement:.1%} >= threshold {self.improvement_threshold:.1%}"

            # Register new version if registry available
            new_version = None
            if self.registry and new_checkpoint:
                version = self.registry.register(
                    checkpoint_state=new_checkpoint,
                    metrics=new_metrics,
                    curriculum_stage=curriculum_stage,
                    training_steps=training_steps,
                    notes=f"Nightly retrain: {reason}",
                )
                new_version = version.version_id

            self._send_alert(
                "✅ RETRAIN ACCEPTED",
                f"New model v{new_version}: reward {new_metric:.4f} "
                f"(+{improvement:.1%} vs v{old_version})",
            )
        else:
            new_version = None
            reason = f"Improvement {improvement:.1%} < threshold {self.improvement_threshold:.1%}"

            self._send_alert(
                "❌ RETRAIN REJECTED",
                f"New model rejected: reward {new_metric:.4f} "
                f"({improvement:+.1%} vs v{old_version}). Keeping old model.",
            )

        result = RetrainResult(
            timestamp=now,
            old_version=old_version or 0,
            new_version=new_version,
            old_metric=old_metric,
            new_metric=new_metric,
            improvement_pct=improvement * 100,
            accepted=accepted,
            reason=reason,
            details={
                "old_metrics": old_metrics,
                "new_metrics": new_metrics,
            },
        )
        self.history.append(result)

        logger.info(
            "Retrain result: %s (improvement=%.1f%%)",
            "ACCEPTED" if accepted else "REJECTED",
            improvement * 100,
        )
        return result

    def _send_alert(self, title: str, details: str) -> None:
        if self._alert_callback:
            try:
                self._alert_callback(title, details)
            except Exception as e:
                logger.error("Failed to send retrain alert: %s", e)
