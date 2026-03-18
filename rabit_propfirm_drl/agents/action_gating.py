"""
ActionGating — Safety layer between SAC output and trade execution.

Purpose:
    Converts raw SAC actions into trade decisions, FORCING the bot to
    sit flat (HOLD) when it's not confident enough. This prevents:
    - Overtrading on noise
    - Taking low-conviction bets that erode the account
    - Trading during uncertain regime transitions

How it works:
    SAC outputs: [confidence, risk_frac, sl_mult, tp_mult] ∈ [-1, 1]^4

    ┌──────────────────────────────────────────────────────────┐
    │  confidence > +threshold  →  BUY                        │
    │     risk = scale(|confidence|) × raw_risk               │
    │                                                          │
    │  confidence < -threshold  →  SELL                       │
    │     risk = scale(|confidence|) × raw_risk               │
    │                                                          │
    │  |confidence| < threshold →  HOLD (mandatory)           │
    │     risk = 0, sl = 0, tp = 0 (no trade)                │
    └──────────────────────────────────────────────────────────┘

    Scale function: (|c| - threshold) / (1 - threshold)
    Maps [threshold, 1] → [0, 1] linearly.
    So c=0.3 (at threshold) → risk_scale=0 (barely confident)
       c=1.0 (max)         → risk_scale=1 (fully confident)

    This creates a "dead zone" near zero where the bot MUST hold.
    A trader who trades without conviction is a losing trader.

Config: confidence_threshold from prop_rules.yaml (default: 0.3)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import NamedTuple

import torch


class TradeSignal(IntEnum):
    """Trade direction enum."""
    HOLD = 0
    BUY = 1
    SELL = -1


@dataclass
class GatedAction:
    """
    Processed action after gating.

    Attributes:
        signal: TradeSignal (HOLD/BUY/SELL)
        risk_fraction: Scaled risk fraction (0 if HOLD)
        sl_multiplier: Stop-loss multiplier (0 if HOLD)
        tp_multiplier: Take-profit multiplier (0 if HOLD)
        raw_confidence: Original confidence from SAC [-1, 1]
        confidence_scale: How far above threshold (0-1)
    """
    signal: TradeSignal
    risk_fraction: float
    sl_multiplier: float
    tp_multiplier: float
    raw_confidence: float
    confidence_scale: float


class ActionGating:
    """
    Action gating layer — converts raw SAC actions to trade decisions.

    Forces HOLD when |confidence| < threshold.
    Scales risk proportionally to confidence above threshold.

    Args:
        confidence_threshold: Minimum |confidence| to trigger a trade.
            Read from prop_rules.yaml. Default: 0.3
        min_risk_fraction: Minimum risk fraction when trading (floor).
            Prevents micro-sized trades. Default: 0.0
        max_risk_fraction: Maximum risk fraction cap. Default: 1.0
    """

    def __init__(
        self,
        confidence_threshold: float = 0.3,
        min_risk_fraction: float = 0.0,
        max_risk_fraction: float = 1.0,
    ) -> None:
        assert 0.0 < confidence_threshold < 1.0, (
            f"threshold must be in (0, 1), got {confidence_threshold}"
        )
        self.confidence_threshold = confidence_threshold
        self.min_risk_fraction = min_risk_fraction
        self.max_risk_fraction = max_risk_fraction

    def gate(self, action: torch.Tensor) -> list[GatedAction]:
        """
        Apply gating to a batch of SAC actions.

        Args:
            action: (batch, 4) — [confidence, risk_frac, sl_mult, tp_mult]
                    All values in [-1, 1] (tanh output from SAC)

        Returns:
            List of GatedAction objects, one per sample.
        """
        actions = action.detach().cpu()
        batch_size = actions.shape[0]
        results: list[GatedAction] = []

        for i in range(batch_size):
            confidence = float(actions[i, 0])
            raw_risk = float(actions[i, 1])
            raw_sl = float(actions[i, 2])
            raw_tp = float(actions[i, 3])

            abs_conf = abs(confidence)

            if abs_conf < self.confidence_threshold:
                # ── HOLD: Not confident enough ──
                results.append(GatedAction(
                    signal=TradeSignal.HOLD,
                    risk_fraction=0.0,
                    sl_multiplier=0.0,
                    tp_multiplier=0.0,
                    raw_confidence=confidence,
                    confidence_scale=0.0,
                ))
            else:
                # ── BUY or SELL ──
                # Scale: maps [threshold, 1] → [0, 1]
                confidence_scale = (abs_conf - self.confidence_threshold) / (
                    1.0 - self.confidence_threshold
                )

                # Scale risk by confidence
                # Higher confidence → more risk allowed
                # raw_risk ∈ [-1, 1], we need [0, 1] → use (raw_risk + 1) / 2
                risk_01 = (raw_risk + 1.0) / 2.0
                scaled_risk = confidence_scale * risk_01
                scaled_risk = max(self.min_risk_fraction,
                                  min(self.max_risk_fraction, scaled_risk))

                # SL/TP multipliers: map [-1, 1] → [0.5, 2.0] range
                sl_mult = 0.5 + (raw_sl + 1.0) / 2.0 * 1.5  # [0.5, 2.0]
                tp_mult = 0.5 + (raw_tp + 1.0) / 2.0 * 1.5  # [0.5, 2.0]

                signal = TradeSignal.BUY if confidence > 0 else TradeSignal.SELL

                results.append(GatedAction(
                    signal=signal,
                    risk_fraction=scaled_risk,
                    sl_multiplier=sl_mult,
                    tp_multiplier=tp_mult,
                    raw_confidence=confidence,
                    confidence_scale=confidence_scale,
                ))

        return results

    def gate_single(self, action: torch.Tensor) -> GatedAction:
        """Gate a single action vector (1, 4) → GatedAction."""
        if action.dim() == 1:
            action = action.unsqueeze(0)
        return self.gate(action)[0]
