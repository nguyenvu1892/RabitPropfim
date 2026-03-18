"""
Curriculum Learning Manager — Progressive difficulty for DRL training.

4 Stages:
1. Kindergarten: Fixed spread, no slippage, trending data only, relaxed DD
2. Elementary: Variable spread, slippage ON, trending+ranging, moderate DD
3. High School: Full execution sim, all regimes, Prop Firm DD limits
4. University: Full complexity + news events, train until convergence

Auto-promotes to next stage when performance threshold is met.
All params from train_hyperparams.yaml.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class StageConfig:
    """Configuration for a single curriculum stage."""
    name: str
    max_steps: int
    spread_mode: str        # "fixed" or "variable"
    slippage_enabled: bool
    commission_enabled: bool
    data_filter: str        # Which data regimes to include
    max_dd_override: float | None
    promote_reward_threshold: float | None

    def to_env_overrides(self, base_config: dict) -> dict:
        """Generate environment config overrides for this stage."""
        overrides = dict(base_config)

        if self.spread_mode == "fixed":
            overrides["news_spread_multiplier"] = 1.0
            overrides["low_liquidity_multiplier"] = 1.0

        if not self.slippage_enabled:
            overrides["slippage_base_pips"] = 0.0
            overrides["slippage_lot_coefficient"] = 0.0

        if self.max_dd_override is not None:
            overrides["max_daily_drawdown"] = self.max_dd_override
            overrides["max_total_drawdown"] = min(
                self.max_dd_override * 2, 0.20
            )
            overrides["killswitch_dd_threshold"] = self.max_dd_override * 0.9

        return overrides


class CurriculumManager:
    """
    Manages progressive difficulty stages during training.

    Tracks performance and auto-promotes when threshold is met.
    """

    def __init__(self, config: dict) -> None:
        """
        Args:
            config: Dict from train_hyperparams.yaml
        """
        stage_configs = config.get("curriculum_stage_configs", {})
        self.stages: list[StageConfig] = []

        for key in sorted(stage_configs.keys()):
            sc = stage_configs[key]
            self.stages.append(StageConfig(
                name=sc["name"],
                max_steps=sc["max_steps"],
                spread_mode=sc["spread_mode"],
                slippage_enabled=sc["slippage_enabled"],
                commission_enabled=sc["commission_enabled"],
                data_filter=sc["data_filter"],
                max_dd_override=sc.get("max_dd_override"),
                promote_reward_threshold=sc.get("promote_reward_threshold"),
            ))

        self.current_stage_idx = 0
        self.stage_steps: int = 0
        self.stage_rewards: list[float] = []
        self._eval_window = 20  # Evaluate over last N episodes

        logger.info(
            "Curriculum initialized with %d stages: %s",
            len(self.stages),
            [s.name for s in self.stages],
        )

    @property
    def current_stage(self) -> StageConfig:
        return self.stages[self.current_stage_idx]

    @property
    def stage_name(self) -> str:
        return self.current_stage.name

    @property
    def is_final_stage(self) -> bool:
        return self.current_stage_idx >= len(self.stages) - 1

    @property
    def total_stages(self) -> int:
        return len(self.stages)

    def get_env_config(self, base_config: dict) -> dict:
        """Get environment config with current stage overrides."""
        return self.current_stage.to_env_overrides(base_config)

    def record_episode(self, total_reward: float) -> None:
        """Record an episode's total reward for promotion evaluation."""
        self.stage_rewards.append(total_reward)

    def step(self) -> None:
        """Increment step counter."""
        self.stage_steps += 1

    def should_promote(self) -> bool:
        """
        Check if agent should be promoted to the next stage.

        Criteria:
        1. Enough episodes evaluated (>= eval_window)
        2. Mean reward exceeds threshold
        3. Not already at final stage
        """
        if self.is_final_stage:
            return False

        threshold = self.current_stage.promote_reward_threshold
        if threshold is None:
            return False

        if len(self.stage_rewards) < self._eval_window:
            return False

        recent = self.stage_rewards[-self._eval_window:]
        mean_reward = sum(recent) / len(recent)

        return mean_reward >= threshold

    def promote(self) -> bool:
        """
        Promote to next stage if conditions are met.

        Returns:
            True if promoted, False if already at final stage
        """
        if self.is_final_stage:
            return False

        old_name = self.stage_name
        self.current_stage_idx += 1
        self.stage_steps = 0
        self.stage_rewards.clear()

        logger.info(
            "🎓 PROMOTED: %s → %s (stage %d/%d)",
            old_name, self.stage_name,
            self.current_stage_idx + 1, self.total_stages,
        )
        return True

    def check_and_promote(self) -> bool:
        """Check and promote if conditions met. Returns True if promoted."""
        if self.should_promote():
            return self.promote()
        return False

    def state_dict(self) -> dict:
        """Serialize state for checkpointing."""
        return {
            "current_stage_idx": self.current_stage_idx,
            "stage_steps": self.stage_steps,
            "stage_rewards": list(self.stage_rewards),
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore state from checkpoint."""
        self.current_stage_idx = state["current_stage_idx"]
        self.stage_steps = state["stage_steps"]
        self.stage_rewards = list(state["stage_rewards"])
