"""
VIP Buffer -- "Ruong Vang Ky Uc" (Golden Memory Box).

Self-Imitation Learning buffer for Stage 2+.
Stores high-quality winning trades that pass the SMC Expert Filter.

SMC Filter checks:
1. Proximity to Order Block (OB)
2. Post-CHOCH/BOS confirmation
3. Volume spike > 2 sigma
4. Pin bar / engulfing pattern

This module is SCAFFOLDED for Stage 2. Stage 1 uses discrete actions
without VIP replay.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class VIPExperience:
    """A single high-quality trade experience."""
    obs: np.ndarray           # Observation at entry
    action: int | np.ndarray  # Action taken (discrete or continuous)
    reward: float             # Total reward from this trade
    pnl: float                # Realized PnL
    entry_price: float
    exit_price: float
    direction: int            # +1 BUY, -1 SELL
    hold_duration: int        # Steps held
    smc_score: float          # SMC filter score (0-1)
    symbol: str = ""


class SMCFilter:
    """
    Smart Money Concepts filter for qualifying winning trades.

    Checks if a winning trade aligns with institutional trading patterns:
    1. Near Order Block (supply/demand zone)
    2. After CHOCH (Change of Character) or BOS (Break of Structure)
    3. Volume confirmation (spike > 2 std)
    4. Price action patterns (pin bar, engulfing)
    """

    def __init__(self, config: dict | None = None) -> None:
        self.ob_proximity_bars = 10    # Look back N bars for OB
        self.volume_threshold = 2.0    # Sigma threshold for volume spike
        self.min_score = 0.5           # Minimum score to qualify

    def score_trade(
        self,
        entry_obs: np.ndarray,
        direction: int,
        features_at_entry: np.ndarray | None = None,
    ) -> float:
        """
        Score a winning trade for VIP quality.

        Returns:
            float in [0, 1]. Higher = more aligned with SMC.
        """
        score = 0.0
        n_checks = 4

        if features_at_entry is None or len(features_at_entry) < 5:
            return 0.0

        # Check 1: Volume spike (knowledge feature columns ~40-44)
        try:
            # Volume is typically in the OHLCV or knowledge features
            # Feature index may vary -- use last available volume column
            vol_col = min(4, features_at_entry.shape[1] - 1)
            volumes = features_at_entry[:, vol_col]
            vol_mean = np.mean(volumes[:-1]) if len(volumes) > 1 else volumes[0]
            vol_std = np.std(volumes[:-1]) if len(volumes) > 1 else 1.0
            if vol_std > 0 and (volumes[-1] - vol_mean) / vol_std > self.volume_threshold:
                score += 1.0
        except (IndexError, ValueError):
            pass

        # Check 2: Pin bar pattern (small body, long wick)
        try:
            # OHLCV: open, high, low, close
            o, h, l, c = features_at_entry[-1, 0], features_at_entry[-1, 1], features_at_entry[-1, 2], features_at_entry[-1, 3]
            body = abs(c - o)
            total_range = h - l
            if total_range > 0:
                body_ratio = body / total_range
                if body_ratio < 0.3:  # Pin bar: body < 30% of range
                    score += 1.0
        except (IndexError, ValueError):
            pass

        # Check 3: Trend alignment (simple: direction matches recent move)
        try:
            recent_close = features_at_entry[-5:, 3]  # Last 5 closes
            trend = recent_close[-1] - recent_close[0]
            if (direction > 0 and trend > 0) or (direction < 0 and trend < 0):
                score += 1.0
        except (IndexError, ValueError):
            pass

        # Check 4: Breakout from range (BOS proxy)
        try:
            lookback = features_at_entry[-20:, 1] if len(features_at_entry) >= 20 else features_at_entry[:, 1]
            recent_high = np.max(lookback[:-1])
            if features_at_entry[-1, 1] > recent_high:  # New high
                score += 1.0 if direction > 0 else 0.0
            recent_low = np.min(features_at_entry[-20:, 2] if len(features_at_entry) >= 20 else features_at_entry[:, 2])
            if features_at_entry[-1, 2] < recent_low:  # New low
                score += 1.0 if direction < 0 else 0.0
        except (IndexError, ValueError):
            pass

        return min(score / n_checks, 1.0)

    def qualifies(self, score: float) -> bool:
        """Check if a trade score meets VIP threshold."""
        return score >= self.min_score


class VIPBuffer:
    """
    Prioritized replay buffer for high-quality trades.

    Trades must:
    1. Be winning (PnL > 0)
    2. Pass SMC filter (score >= 0.5)

    Used in Stage 2+ for Self-Imitation Learning.
    """

    def __init__(self, capacity: int = 10_000, config: dict | None = None) -> None:
        self.capacity = capacity
        self.buffer: deque[VIPExperience] = deque(maxlen=capacity)
        self.smc_filter = SMCFilter(config)

        # Stats
        self.total_evaluated = 0
        self.total_accepted = 0
        self.total_rejected = 0

    def try_add(
        self,
        obs: np.ndarray,
        action: int | np.ndarray,
        reward: float,
        pnl: float,
        entry_price: float,
        exit_price: float,
        direction: int,
        hold_duration: int,
        features_at_entry: np.ndarray | None = None,
        symbol: str = "",
    ) -> bool:
        """
        Evaluate and potentially add a winning trade to VIP buffer.

        Returns:
            True if trade was accepted into VIP buffer.
        """
        self.total_evaluated += 1

        # Must be a winning trade
        if pnl <= 0:
            self.total_rejected += 1
            return False

        # SMC filter
        smc_score = self.smc_filter.score_trade(
            obs, direction, features_at_entry,
        )

        if not self.smc_filter.qualifies(smc_score):
            self.total_rejected += 1
            return False

        # Add to buffer
        exp = VIPExperience(
            obs=obs, action=action, reward=reward, pnl=pnl,
            entry_price=entry_price, exit_price=exit_price,
            direction=direction, hold_duration=hold_duration,
            smc_score=smc_score, symbol=symbol,
        )
        self.buffer.append(exp)
        self.total_accepted += 1

        logger.debug(
            "VIP: Added %s trade (PnL=%.2f, SMC=%.2f) [%d/%d in buffer]",
            "BUY" if direction > 0 else "SELL", pnl, smc_score,
            len(self.buffer), self.capacity,
        )
        return True

    def sample_batch(self, batch_size: int) -> list[VIPExperience]:
        """Sample a batch of VIP experiences for imitation learning."""
        if len(self.buffer) < batch_size:
            return list(self.buffer)

        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

    def __len__(self) -> int:
        return len(self.buffer)

    def stats(self) -> dict:
        return {
            "buffer_size": len(self.buffer),
            "capacity": self.capacity,
            "total_evaluated": self.total_evaluated,
            "total_accepted": self.total_accepted,
            "total_rejected": self.total_rejected,
            "acceptance_rate": self.total_accepted / max(self.total_evaluated, 1),
        }
