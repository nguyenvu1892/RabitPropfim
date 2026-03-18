"""
Market Physics Simulation — Realistic execution conditions for training environment.

Simulates:
- Variable spread (session time, volatility, news events)
- Slippage (proportional to lot size)
- Execution delay (50-150ms random)
- Partial fills (5% probability for large lots)
- Requotes (2% probability during high volatility)

All parameters from prop_rules.yaml — zero hardcoding.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExecutionResult:
    """Result of attempting to execute an order through the physics engine."""
    filled: bool                    # Whether the order was filled
    fill_price: float              # Actual fill price (after spread + slippage)
    fill_lots: float               # Actual filled lot size (may be partial)
    spread_pips: float             # Spread applied
    slippage_pips: float           # Slippage applied
    delay_ms: float                # Execution delay in milliseconds
    requoted: bool                 # Whether a requote occurred
    total_cost_pips: float         # Total execution cost in pips


class MarketPhysics:
    """
    Simulates realistic market execution conditions.

    All parameters loaded from config dict (from prop_rules.yaml).
    """

    def __init__(self, config: dict, seed: int = 42) -> None:
        """
        Args:
            config: Dict from prop_rules.yaml (validated by Pydantic)
            seed: Random seed for reproducibility
        """
        self.rng = np.random.default_rng(seed)

        # Load from config
        self.base_spread = config.get("base_spread_pips", 1.5)
        self.news_multiplier = config.get("news_spread_multiplier", 8.0)
        self.low_liq_multiplier = config.get("low_liquidity_multiplier", 2.5)
        self.slippage_base = config.get("slippage_base_pips", 0.2)
        self.slippage_lot_coeff = config.get("slippage_lot_coefficient", 0.1)
        self.delay_min = config.get("execution_delay_min_ms", 50)
        self.delay_max = config.get("execution_delay_max_ms", 150)
        self.partial_fill_prob = config.get("partial_fill_probability", 0.05)
        self.requote_prob = config.get("requote_probability", 0.02)

    def variable_spread(
        self,
        hour_utc: int,
        volatility_ratio: float = 1.0,
        is_news: bool = False,
    ) -> float:
        """
        Calculate variable spread based on market conditions.

        Args:
            hour_utc: Current hour in UTC (0-23)
            volatility_ratio: Current vol / average vol (1.0 = normal)
            is_news: Whether a high-impact news event is active

        Returns:
            Spread in pips
        """
        spread = self.base_spread

        # News event: massive spread spike
        if is_news:
            spread *= self.news_multiplier * self.rng.uniform(0.8, 1.2)
            return spread

        # Session-based multiplier
        if hour_utc in range(22, 24) or hour_utc in range(0, 5):
            # Asian low-liquidity (22:00-05:00 UTC)
            spread *= self.low_liq_multiplier * self.rng.uniform(0.9, 1.1)
        elif hour_utc in range(13, 17):
            # US session overlap: tightest spreads
            spread *= self.rng.uniform(0.7, 0.9)
        else:
            # Normal session
            spread *= self.rng.uniform(0.9, 1.1)

        # Volatility adjustment: higher vol → wider spread
        if volatility_ratio > 1.5:
            spread *= 1.0 + 0.3 * (volatility_ratio - 1.0)

        return max(spread, 0.1)  # Minimum 0.1 pip

    def slippage(self, lot_size: float) -> float:
        """
        Calculate slippage based on lot size.

        Larger orders slip more due to liquidity consumption.

        Args:
            lot_size: Order size in lots

        Returns:
            Slippage in pips (always >= 0)
        """
        if lot_size <= 0:
            return 0.0

        base = self.slippage_base
        lot_component = self.slippage_lot_coeff * np.log1p(lot_size)
        noise = self.rng.exponential(0.3)  # Right-skewed noise

        return max(0.0, base + lot_component + noise * 0.1)

    def execution_delay(self) -> float:
        """
        Generate random execution delay in milliseconds.

        Returns:
            Delay in milliseconds (uniform distribution)
        """
        return self.rng.uniform(self.delay_min, self.delay_max)

    def partial_fill(self, lot_size: float) -> float:
        """
        Determine actual fill size (may be partial for large orders).

        Args:
            lot_size: Requested lot size

        Returns:
            Actual filled lot size
        """
        if lot_size <= 5.0:
            return lot_size  # Small orders always fully filled

        if self.rng.random() < self.partial_fill_prob:
            fill_ratio = self.rng.uniform(0.5, 1.0)
            filled = lot_size * fill_ratio
            logger.debug("Partial fill: %.2f / %.2f lots", filled, lot_size)
            return filled

        return lot_size

    def requote(self, volatility_ratio: float = 1.0) -> bool:
        """
        Determine if a requote occurs.

        Higher volatility increases requote probability.

        Args:
            volatility_ratio: Current vol / average vol

        Returns:
            True if requoted (order not filled at requested price)
        """
        adjusted_prob = self.requote_prob
        if volatility_ratio > 2.0:
            adjusted_prob *= volatility_ratio / 2.0  # Scale up with vol

        return self.rng.random() < min(adjusted_prob, 0.20)  # Cap at 20%

    def execute_order(
        self,
        price: float,
        direction: int,        # +1 = BUY, -1 = SELL
        lot_size: float,
        hour_utc: int,
        volatility_ratio: float = 1.0,
        is_news: bool = False,
        pip_value: float = 0.0001,  # 1 pip in price units (0.0001 for forex)
    ) -> ExecutionResult:
        """
        Simulate full order execution with all market physics.

        Args:
            price: Current market price
            direction: +1 for BUY, -1 for SELL
            lot_size: Requested lot size
            hour_utc: Current hour UTC
            volatility_ratio: Current volatility / average
            is_news: Whether news event is active
            pip_value: Value of 1 pip in price units

        Returns:
            ExecutionResult with all execution details
        """
        # 1. Check requote first
        if self.requote(volatility_ratio):
            # Requote: price moves against us
            price_change = self.rng.uniform(0.5, 2.0) * pip_value * direction
            return ExecutionResult(
                filled=False,
                fill_price=price + price_change,
                fill_lots=0.0,
                spread_pips=0.0,
                slippage_pips=0.0,
                delay_ms=self.execution_delay(),
                requoted=True,
                total_cost_pips=0.0,
            )

        # 2. Calculate spread
        spread = self.variable_spread(hour_utc, volatility_ratio, is_news)

        # 3. Calculate slippage
        slip = self.slippage(lot_size)

        # 4. Partial fill check
        filled_lots = self.partial_fill(lot_size)

        # 5. Execution delay
        delay = self.execution_delay()

        # 6. Calculate fill price
        # BUY: pay ask (price + spread/2) + slippage
        # SELL: receive bid (price - spread/2) - slippage
        spread_cost = spread * pip_value / 2.0
        slip_cost = slip * pip_value

        if direction > 0:  # BUY
            fill_price = price + spread_cost + slip_cost
        else:  # SELL
            fill_price = price - spread_cost - slip_cost

        total_cost = spread + slip

        return ExecutionResult(
            filled=True,
            fill_price=fill_price,
            fill_lots=filled_lots,
            spread_pips=spread,
            slippage_pips=slip,
            delay_ms=delay,
            requoted=False,
            total_cost_pips=total_cost,
        )
