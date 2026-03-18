"""
CurriculumRunner -- 4-stage progressive difficulty for DRL training.

Mimics a trader's education from a safe sandbox to brutal real markets:

    Stage 1 (Kindergarten):
        Fixed spread, zero slippage, trending data only.
        "Learn to crawl before you walk."
        -> Most trades are easy profits in clear trends.

    Stage 2 (Elementary):
        Variable spread, ranging market introduced.
        "Learn to handle boredom and chop."
        -> Must learn to HOLD when market is indecisive.

    Stage 3 (High School):
        Real spread, real slippage, real commission.
        "Welcome to the real world."
        -> Execution costs eat into profits. Must be selective.

    Stage 4 (University / Hardcore):
        News gaps, requotes, execution delay, full volatility.
        "Trial by fire -- only the strong survive."
        -> If it can profit here, it can profit anywhere.

Auto-promote: When avg_reward of last 1000 episodes exceeds
the stage's threshold, automatically advance to next stage.

All thresholds configurable via constructor or yaml.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class StageConfig:
    """Configuration for a single curriculum stage."""
    name: str
    stage_id: int               # 1-4

    # Execution simulation
    spread_mode: str            # "fixed" or "variable"
    spread_multiplier: float    # 1.0 = normal, 0.0 = no spread
    slippage_enabled: bool
    slippage_pips: float        # Max slippage in pips
    commission_enabled: bool
    commission_per_lot: float   # $ per lot

    # Market data filter
    data_filter: str            # "trend_only", "trend+range", "all", "all+news"

    # Risk overrides (relaxed for easier stages)
    max_daily_dd: float         # Daily DD limit (fraction)
    max_total_dd: float         # Total DD limit (fraction)

    # Auto-promote threshold
    promote_threshold: float | None  # avg_reward must exceed this
    promote_window: int = 1000       # Evaluate over last N episodes

    def to_env_overrides(self) -> dict:
        """Generate environment config overrides for this stage."""
        return {
            "spread_mode": self.spread_mode,
            "spread_multiplier": self.spread_multiplier,
            "slippage_enabled": self.slippage_enabled,
            "slippage_pips": self.slippage_pips,
            "commission_enabled": self.commission_enabled,
            "commission_per_lot": self.commission_per_lot,
            "data_filter": self.data_filter,
            "max_daily_dd": self.max_daily_dd,
            "max_total_dd": self.max_total_dd,
        }


# Default 4-stage curriculum
DEFAULT_STAGES: list[StageConfig] = [
    StageConfig(
        name="Kindergarten",
        stage_id=1,
        spread_mode="fixed",
        spread_multiplier=0.0,    # No spread cost
        slippage_enabled=False,
        slippage_pips=0.0,
        commission_enabled=False,
        commission_per_lot=0.0,
        data_filter="trend_only",
        max_daily_dd=0.10,        # Relaxed: 10% daily DD allowed
        max_total_dd=0.20,        # Relaxed: 20% total DD
        promote_threshold=2.0,    # Easy target
        promote_window=1000,
    ),
    StageConfig(
        name="Elementary",
        stage_id=2,
        spread_mode="variable",
        spread_multiplier=0.5,    # Half real spread
        slippage_enabled=False,
        slippage_pips=0.0,
        commission_enabled=False,
        commission_per_lot=0.0,
        data_filter="trend+range",
        max_daily_dd=0.05,        # Moderate: 5% daily DD
        max_total_dd=0.12,        # Moderate: 12% total DD
        promote_threshold=1.5,
        promote_window=1000,
    ),
    StageConfig(
        name="High School",
        stage_id=3,
        spread_mode="variable",
        spread_multiplier=1.0,    # Full real spread
        slippage_enabled=True,
        slippage_pips=0.5,
        commission_enabled=True,
        commission_per_lot=7.0,   # Standard $7/lot
        data_filter="all",
        max_daily_dd=0.03,        # Prop firm: 3% daily DD
        max_total_dd=0.08,        # Prop firm: 8% total DD
        promote_threshold=1.0,
        promote_window=1000,
    ),
    StageConfig(
        name="University",
        stage_id=4,
        spread_mode="variable",
        spread_multiplier=1.5,    # Wider spread (news events)
        slippage_enabled=True,
        slippage_pips=2.0,        # Heavy slippage
        commission_enabled=True,
        commission_per_lot=7.0,
        data_filter="all+news",   # Include news gaps
        max_daily_dd=0.03,        # Prop firm strict
        max_total_dd=0.06,        # Even stricter
        promote_threshold=None,   # Final stage -- no promotion
        promote_window=1000,
    ),
]


class CurriculumRunner:
    """
    Manages 4-stage progressive difficulty during training.

    Tracks episode rewards in a rolling window and auto-promotes
    when the average exceeds the current stage's threshold.

    Usage:
        runner = CurriculumRunner()
        env_config = runner.get_env_overrides()

        for episode in range(total_episodes):
            # ... run episode with env_config ...
            runner.record_episode(total_reward)
            promoted = runner.check_and_promote()
            if promoted:
                env_config = runner.get_env_overrides()
    """

    def __init__(
        self,
        stages: list[StageConfig] | None = None,
        promote_window: int = 1000,
    ) -> None:
        """
        Args:
            stages: Custom stage configs. If None, uses DEFAULT_STAGES.
            promote_window: Override promote_window for all stages.
        """
        self.stages = stages or [
            StageConfig(**{**s.__dict__}) for s in DEFAULT_STAGES
        ]

        # Override promote_window if specified
        if promote_window != 1000:
            for s in self.stages:
                s.promote_window = promote_window

        self.current_stage_idx = 0
        self.total_episodes = 0
        self.stage_episode_count = 0

        # Rolling reward buffer per stage
        max_window = max(s.promote_window for s in self.stages)
        self._reward_buffer: deque[float] = deque(maxlen=max_window)

        self._promotion_log: list[dict] = []

        logger.info(
            "Curriculum initialized: %d stages [%s]",
            len(self.stages),
            " -> ".join(s.name for s in self.stages),
        )

    @property
    def current_stage(self) -> StageConfig:
        """Current active stage config."""
        return self.stages[self.current_stage_idx]

    @property
    def stage_name(self) -> str:
        """Human-readable name of current stage."""
        return self.current_stage.name

    @property
    def is_final_stage(self) -> bool:
        """True if at the last (hardest) stage."""
        return self.current_stage_idx >= len(self.stages) - 1

    @property
    def progress(self) -> str:
        """Stage progress string, e.g. '2/4 (Elementary)'."""
        return f"{self.current_stage_idx + 1}/{len(self.stages)} ({self.stage_name})"

    def get_env_overrides(self) -> dict:
        """Get environment config overrides for current stage."""
        return self.current_stage.to_env_overrides()

    def record_episode(self, total_reward: float) -> None:
        """Record an episode's total reward."""
        self._reward_buffer.append(total_reward)
        self.total_episodes += 1
        self.stage_episode_count += 1

    def _avg_reward(self) -> float:
        """Average reward over the promote window."""
        window = self.current_stage.promote_window
        if len(self._reward_buffer) < window:
            return float("-inf")
        recent = list(self._reward_buffer)[-window:]
        return sum(recent) / len(recent)

    def should_promote(self) -> bool:
        """
        Check if agent should advance to next stage.

        Criteria:
        1. Not at final stage
        2. Stage has a promote_threshold (not None)
        3. Enough episodes (>= promote_window)
        4. avg_reward of last promote_window episodes > threshold
        """
        if self.is_final_stage:
            return False

        threshold = self.current_stage.promote_threshold
        if threshold is None:
            return False

        window = self.current_stage.promote_window
        if len(self._reward_buffer) < window:
            return False

        avg = self._avg_reward()
        return avg >= threshold

    def promote(self) -> bool:
        """
        Promote to next stage.

        Returns:
            True if promoted, False if already at final stage.
        """
        if self.is_final_stage:
            return False

        old_name = self.stage_name
        old_avg = self._avg_reward()

        self.current_stage_idx += 1
        self.stage_episode_count = 0
        self._reward_buffer.clear()

        self._promotion_log.append({
            "from_stage": old_name,
            "to_stage": self.stage_name,
            "avg_reward_at_promote": old_avg,
            "total_episodes": self.total_episodes,
        })

        logger.info(
            "PROMOTED: %s -> %s (avg_reward=%.2f, episode=%d)",
            old_name, self.stage_name, old_avg, self.total_episodes,
        )
        return True

    def check_and_promote(self) -> bool:
        """Check conditions and promote if met. Returns True if promoted."""
        if self.should_promote():
            return self.promote()
        return False

    def state_dict(self) -> dict:
        """Serialize state for checkpointing."""
        return {
            "current_stage_idx": self.current_stage_idx,
            "total_episodes": self.total_episodes,
            "stage_episode_count": self.stage_episode_count,
            "reward_buffer": list(self._reward_buffer),
            "promotion_log": self._promotion_log,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore state from checkpoint."""
        self.current_stage_idx = state["current_stage_idx"]
        self.total_episodes = state["total_episodes"]
        self.stage_episode_count = state["stage_episode_count"]
        self._reward_buffer = deque(
            state["reward_buffer"],
            maxlen=self._reward_buffer.maxlen,
        )
        self._promotion_log = state["promotion_log"]
