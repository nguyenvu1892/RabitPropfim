"""
EpisodicMemory — Case-Based Experience Memory for Trading.

Implements a rolling memory bank of past trade setups. Before each new
trade, the agent can query similar setups (via cosine similarity on the
knowledge vector) to get a confidence bonus/penalty based on historical
win rates of similar situations.

Key design decisions:
    1. AUXILIARY signal only — memory_bonus ∈ [-0.3, +0.3]
       Never overrides the neural network's decision, only adjusts confidence.
    2. Rolling window of 500 entries — keeps memory fresh and relevant.
    3. Cold start protection — returns 0 bonus when < 50 entries.
    4. k-NN with cosine similarity — fast (<1ms) on 500 × 22-dim vectors.
    5. Only stores clean trades (with SL/TP hit, not timeout/force-close).

Usage:
    memory = EpisodicMemory(capacity=500, k=5)
    memory.add(knowledge_vec, result_pnl, is_win, rr, hold_time, ...)
    bonus = memory.query(current_knowledge_vec)
    final_confidence = agent_confidence * (1 + bonus)
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ─── Constants ──────────────────────────────────────────────────────
DEFAULT_CAPACITY = 500
DEFAULT_K = 5
MIN_ENTRIES_FOR_QUERY = 50
BONUS_CLAMP_MIN = -0.3
BONUS_CLAMP_MAX = 0.3


@dataclass
class MemoryEntry:
    """
    A single episode in the trading memory.

    Stores the setup's fingerprint (knowledge_vector) and the outcome
    (win/loss, PnL, hold time), enabling similarity-based retrieval.
    """

    # Fingerprint — for similarity search
    knowledge_vector: np.ndarray   # (22,) from KnowledgeExtractor
    regime: int                    # 0-3 from RegimeDetector

    # Outcome
    pnl: float                     # Realized PnL (positive or negative)
    is_win: bool                   # Trade was profitable
    rr_achieved: float             # Actual risk-reward ratio
    hold_time_minutes: int         # How long the position was held

    # Metadata
    timestamp: str                 # ISO format timestamp
    symbol: str                    # Trading instrument
    direction: int                 # +1 BUY, -1 SELL
    close_reason: str              # "SL_HIT", "TP_HIT", "SIGNAL_EXIT"

    def to_dict(self) -> dict:
        """Serialize to dict (for JSON persistence)."""
        d = {
            "knowledge_vector": self.knowledge_vector.tolist(),
            "regime": self.regime,
            "pnl": self.pnl,
            "is_win": self.is_win,
            "rr_achieved": self.rr_achieved,
            "hold_time_minutes": self.hold_time_minutes,
            "timestamp": self.timestamp,
            "symbol": self.symbol,
            "direction": self.direction,
            "close_reason": self.close_reason,
        }
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "MemoryEntry":
        """Deserialize from dict."""
        return cls(
            knowledge_vector=np.array(d["knowledge_vector"], dtype=np.float32),
            regime=d["regime"],
            pnl=d["pnl"],
            is_win=d["is_win"],
            rr_achieved=d["rr_achieved"],
            hold_time_minutes=d["hold_time_minutes"],
            timestamp=d["timestamp"],
            symbol=d["symbol"],
            direction=d["direction"],
            close_reason=d["close_reason"],
        )


class EpisodicMemory:
    """
    Rolling memory bank of past trade setups with k-NN retrieval.

    Stores up to `capacity` recent trade entries. When queried with a
    new knowledge vector, finds the `k` most similar past setups and
    returns a confidence bonus based on their aggregate win rate.

    The bonus is AUXILIARY — it adjusts the agent's confidence by ±30%
    at most. The neural network always has the final say.

    Args:
        capacity: Maximum entries in memory (rolling window). Default: 500
        k: Number of nearest neighbors for retrieval. Default: 5
        min_entries: Minimum entries required before querying. Default: 50
        bonus_scale: Base scaling for win-rate-to-bonus conversion. Default: 0.3
        direction_weight: Extra weight for same-direction similarity. Default: 0.1
    """

    def __init__(
        self,
        capacity: int = DEFAULT_CAPACITY,
        k: int = DEFAULT_K,
        min_entries: int = MIN_ENTRIES_FOR_QUERY,
        bonus_scale: float = 0.3,
        direction_weight: float = 0.1,
    ) -> None:
        self.capacity = capacity
        self.k = k
        self.min_entries = min_entries
        self.bonus_scale = bonus_scale
        self.direction_weight = direction_weight

        # Storage
        self._entries: list[MemoryEntry] = []

        # Pre-computed matrix for fast k-NN (updated on add)
        self._vectors: Optional[np.ndarray] = None   # (N, 22) float32
        self._wins: Optional[np.ndarray] = None       # (N,) bool
        self._directions: Optional[np.ndarray] = None  # (N,) int
        self._rrs: Optional[np.ndarray] = None         # (N,) float

        # Stats
        self._total_added = 0
        self._total_queries = 0

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PUBLIC API
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def add(
        self,
        knowledge_vector: np.ndarray,
        pnl: float,
        is_win: bool,
        rr_achieved: float = 0.0,
        hold_time_minutes: int = 0,
        symbol: str = "",
        direction: int = 0,
        regime: int = 0,
        close_reason: str = "UNKNOWN",
    ) -> None:
        """
        Add a completed trade to memory.

        Only clean trades should be added — trades with clear SL/TP outcomes.
        Do NOT add timeout/force-close trades as they pollute similarity.

        Args:
            knowledge_vector: (22,) from KnowledgeExtractor at entry time
            pnl: Realized profit/loss
            is_win: Whether trade was profitable
            rr_achieved: Actual risk-reward achieved
            hold_time_minutes: How long position was held
            symbol: Trading instrument name
            direction: +1 BUY, -1 SELL
            regime: Market regime at time of entry (0-3)
            close_reason: Why the trade was closed
        """
        # Filter out non-clean closes
        clean_reasons = {"SL_HIT", "TP_HIT", "SIGNAL_EXIT"}
        if close_reason not in clean_reasons:
            logger.debug(
                "Skipping non-clean trade for memory: %s", close_reason
            )
            return

        entry = MemoryEntry(
            knowledge_vector=np.asarray(knowledge_vector, dtype=np.float32),
            regime=regime,
            pnl=pnl,
            is_win=is_win,
            rr_achieved=rr_achieved,
            hold_time_minutes=hold_time_minutes,
            timestamp=datetime.now(timezone.utc).isoformat(),
            symbol=symbol,
            direction=direction,
            close_reason=close_reason,
        )

        self._entries.append(entry)
        self._total_added += 1

        # Evict oldest if over capacity
        if len(self._entries) > self.capacity:
            self._entries.pop(0)

        # Rebuild lookup arrays
        self._rebuild_arrays()

        logger.debug(
            "Memory: added %s %s (PnL=%.2f, win=%s). Size=%d/%d",
            symbol, "BUY" if direction > 0 else "SELL",
            pnl, is_win, len(self._entries), self.capacity,
        )

    def query(
        self,
        knowledge_vector: np.ndarray,
        direction: int = 0,
    ) -> float:
        """
        Query memory for confidence bonus based on similar past setups.

        Finds k nearest neighbors by cosine similarity, computes their
        aggregate win rate, and returns a bonus in [-0.3, +0.3].

        Args:
            knowledge_vector: (22,) current setup's knowledge vector
            direction: +1 BUY, -1 SELL (for direction-aware matching)

        Returns:
            float: memory_bonus ∈ [-0.3, +0.3]
                   Positive = similar setups were profitable
                   Negative = similar setups lost money
                   0.0 = not enough data or neutral history
        """
        self._total_queries += 1

        # Cold start protection
        if len(self._entries) < self.min_entries:
            return 0.0

        if self._vectors is None:
            return 0.0

        query_vec = np.asarray(knowledge_vector, dtype=np.float32)

        # ── Cosine similarity ──
        similarities = self._cosine_similarity(query_vec, self._vectors)

        # ── Direction bonus ──
        # Give slight preference to same-direction setups
        if direction != 0 and self._directions is not None:
            dir_match = (self._directions == direction).astype(np.float32)
            similarities = similarities + dir_match * self.direction_weight

        # ── Find top-k ──
        k = min(self.k, len(self._entries))
        top_k_indices = np.argsort(similarities)[-k:]

        # ── Compute weighted win rate ──
        top_k_sims = similarities[top_k_indices]
        top_k_wins = self._wins[top_k_indices].astype(np.float32)

        # Similarity-weighted win rate
        sim_weights = np.maximum(top_k_sims, 0.0)  # Only positive similarities
        total_weight = sim_weights.sum()

        if total_weight < 1e-8:
            return 0.0

        weighted_wr = (top_k_wins * sim_weights).sum() / total_weight

        # ── Convert to bonus ──
        # WR = 0.5 → bonus = 0 (neutral)
        # WR = 1.0 → bonus = +bonus_scale (+0.3)
        # WR = 0.0 → bonus = -bonus_scale (-0.3)
        bonus = (weighted_wr - 0.5) * 2.0 * self.bonus_scale

        # Clamp to safety bounds
        bonus = float(np.clip(bonus, BONUS_CLAMP_MIN, BONUS_CLAMP_MAX))

        logger.debug(
            "Memory query: k=%d, avg_sim=%.3f, weighted_WR=%.1f%%, bonus=%.3f",
            k, float(top_k_sims.mean()), weighted_wr * 100, bonus,
        )

        return bonus

    @property
    def size(self) -> int:
        """Current number of entries in memory."""
        return len(self._entries)

    @property
    def is_warm(self) -> bool:
        """Whether memory has enough entries for querying."""
        return len(self._entries) >= self.min_entries

    def get_stats(self) -> dict:
        """Return memory statistics."""
        if not self._entries:
            return {
                "size": 0, "capacity": self.capacity,
                "total_added": self._total_added,
                "total_queries": self._total_queries,
                "is_warm": False,
                "win_rate": 0.0,
            }

        wins = sum(1 for e in self._entries if e.is_win)
        return {
            "size": len(self._entries),
            "capacity": self.capacity,
            "total_added": self._total_added,
            "total_queries": self._total_queries,
            "is_warm": self.is_warm,
            "win_rate": wins / len(self._entries),
            "avg_pnl": sum(e.pnl for e in self._entries) / len(self._entries),
            "avg_rr": sum(e.rr_achieved for e in self._entries) / len(self._entries),
            "avg_hold_min": sum(e.hold_time_minutes for e in self._entries) / len(self._entries),
            "symbols": list(set(e.symbol for e in self._entries)),
        }

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PERSISTENCE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def save(self, path: Path | str) -> None:
        """Save memory to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "capacity": self.capacity,
            "k": self.k,
            "min_entries": self.min_entries,
            "bonus_scale": self.bonus_scale,
            "total_added": self._total_added,
            "total_queries": self._total_queries,
            "entries": [e.to_dict() for e in self._entries],
        }

        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info("EpisodicMemory saved: %d entries → %s", len(self._entries), path)

    @classmethod
    def load(cls, path: Path | str) -> "EpisodicMemory":
        """Load memory from JSON file."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        memory = cls(
            capacity=data.get("capacity", DEFAULT_CAPACITY),
            k=data.get("k", DEFAULT_K),
            min_entries=data.get("min_entries", MIN_ENTRIES_FOR_QUERY),
            bonus_scale=data.get("bonus_scale", 0.3),
        )
        memory._total_added = data.get("total_added", 0)
        memory._total_queries = data.get("total_queries", 0)
        memory._entries = [
            MemoryEntry.from_dict(d) for d in data.get("entries", [])
        ]
        memory._rebuild_arrays()

        logger.info("EpisodicMemory loaded: %d entries ← %s", len(memory._entries), path)
        return memory

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PRIVATE
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

    def _rebuild_arrays(self) -> None:
        """Rebuild NumPy arrays for fast k-NN after add/remove."""
        if not self._entries:
            self._vectors = None
            self._wins = None
            self._directions = None
            self._rrs = None
            return

        self._vectors = np.stack(
            [e.knowledge_vector for e in self._entries], axis=0,
        )  # (N, 22)
        self._wins = np.array(
            [e.is_win for e in self._entries], dtype=bool,
        )  # (N,)
        self._directions = np.array(
            [e.direction for e in self._entries], dtype=np.int32,
        )  # (N,)
        self._rrs = np.array(
            [e.rr_achieved for e in self._entries], dtype=np.float32,
        )  # (N,)

    @staticmethod
    def _cosine_similarity(
        query: np.ndarray, matrix: np.ndarray,
    ) -> np.ndarray:
        """
        Compute cosine similarity between query (22,) and matrix (N, 22).

        Returns: (N,) similarity scores in [-1, 1].

        Vectorized — runs in <1ms for N=500, dim=22.
        """
        # Normalize query
        query_norm = np.linalg.norm(query) + 1e-10
        query_unit = query / query_norm

        # Normalize each row of matrix
        matrix_norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-10
        matrix_unit = matrix / matrix_norms

        # Dot product = cosine similarity (both are unit vectors)
        similarities = matrix_unit @ query_unit  # (N,)

        return similarities
