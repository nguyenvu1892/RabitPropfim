"""
Prioritized Experience Replay (PER) Buffer — Proportional variant.

Key features:
- SumTree data structure for O(log n) priority sampling
- Importance sampling weights with beta annealing
- Efficient batch sampling
- Compatible with SAC's TD-error-based priorities

Reference: Schaul et al. (2015) "Prioritized Experience Replay"
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# SumTree for O(log n) priority sampling
# ─────────────────────────────────────────────

class SumTree:
    """Binary tree where each leaf stores a priority, internal nodes store sums."""

    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_pointer = 0
        self.n_entries = 0

    def _propagate(self, idx: int, change: float) -> None:
        """Propagate priority change up the tree."""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx: int, value: float) -> int:
        """Find leaf index for a given cumulative sum value."""
        left = 2 * idx + 1
        right = left + 1

        if left >= len(self.tree):
            return idx

        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(right, value - self.tree[left])

    @property
    def total(self) -> float:
        """Total priority sum (root of tree)."""
        return float(self.tree[0])

    @property
    def max_priority(self) -> float:
        """Maximum priority across all leaves."""
        leaf_start = self.capacity - 1
        leaf_end = leaf_start + self.n_entries
        if self.n_entries == 0:
            return 1.0
        return float(np.max(self.tree[leaf_start:leaf_end]))

    @property
    def min_priority(self) -> float:
        """Minimum non-zero priority across all leaves."""
        leaf_start = self.capacity - 1
        leaf_end = leaf_start + self.n_entries
        if self.n_entries == 0:
            return 1.0
        leaves = self.tree[leaf_start:leaf_end]
        nonzero = leaves[leaves > 0]
        if len(nonzero) == 0:
            return 1.0
        return float(np.min(nonzero))

    def add(self, priority: float, data_idx: int) -> None:
        """Add/update a leaf with given priority."""
        tree_idx = data_idx + self.capacity - 1
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def get(self, value: float) -> tuple[int, float, int]:
        """
        Sample a leaf by cumulative sum value.

        Returns:
            (tree_index, priority, data_index)
        """
        tree_idx = self._retrieve(0, value)
        data_idx = tree_idx - self.capacity + 1
        return tree_idx, float(self.tree[tree_idx]), data_idx


# ─────────────────────────────────────────────
# PER Buffer
# ─────────────────────────────────────────────

@dataclass
class Transition:
    """Single experience transition."""
    state: np.ndarray
    action: np.ndarray
    reward: float
    next_state: np.ndarray
    done: bool


class PERBuffer:
    """
    Prioritized Experience Replay buffer.

    Uses proportional prioritization: P(i) = p_i^α / Σ p_k^α
    With importance sampling weights: w_i = (N · P(i))^(-β) / max(w)
    """

    def __init__(
        self,
        capacity: int = 1_000_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 500_000,
        eps: float = 1e-6,
        state_dim: int = 14,
        action_dim: int = 4,
    ) -> None:
        """
        Args:
            capacity: Max buffer size
            alpha: Priority exponent (0=uniform, 1=full priority)
            beta_start: Initial importance sampling exponent
            beta_frames: Steps to anneal beta from start to 1.0
            eps: Small constant added to TD-error for non-zero priority
            state_dim: Dimension of state features
            action_dim: Dimension of actions
        """
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.eps = eps

        # SumTree for priority sampling
        self.tree = SumTree(capacity)

        # Pre-allocated storage arrays
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self._pointer = 0
        self._size = 0
        self._frame = 0  # For beta annealing

    @property
    def size(self) -> int:
        return self._size

    @property
    def beta(self) -> float:
        """Current beta value (anneals from beta_start to 1.0)."""
        fraction = min(self._frame / max(self.beta_frames, 1), 1.0)
        return self.beta_start + fraction * (1.0 - self.beta_start)

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add a transition with maximum priority (will be corrected on first sample)."""
        max_p = self.tree.max_priority
        if max_p == 0:
            max_p = 1.0

        # Store data
        idx = self._pointer
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = float(done)

        # Add to tree with max priority
        self.tree.add(max_p, idx)

        self._pointer = (self._pointer + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self, batch_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[int]]:
        """
        Sample a batch with PER.

        Returns:
            (states, actions, rewards, next_states, dones, is_weights, tree_indices)
        """
        self._frame += 1

        indices: list[int] = []
        priorities: list[float] = []
        tree_indices: list[int] = []

        segment = self.tree.total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            value = np.random.uniform(a, b)
            tree_idx, priority, data_idx = self.tree.get(value)

            # Clamp data_idx to valid range
            data_idx = max(0, min(data_idx, self._size - 1))

            tree_indices.append(tree_idx)
            indices.append(data_idx)
            priorities.append(max(priority, self.eps))

        # Calculate importance sampling weights
        priorities_arr = np.array(priorities, dtype=np.float64)
        probs = priorities_arr / max(self.tree.total, self.eps)

        beta = self.beta
        is_weights = (self._size * probs) ** (-beta)
        is_weights /= max(is_weights.max(), self.eps)  # Normalize
        is_weights = is_weights.astype(np.float32)

        idx_arr = np.array(indices)

        return (
            self.states[idx_arr],
            self.actions[idx_arr],
            self.rewards[idx_arr],
            self.next_states[idx_arr],
            self.dones[idx_arr],
            is_weights,
            tree_indices,
        )

    def update_priorities(
        self, tree_indices: list[int], td_errors: np.ndarray
    ) -> None:
        """
        Update priorities based on new TD-errors.

        Args:
            tree_indices: Tree indices from sample()
            td_errors: New TD-errors (absolute values)
        """
        for tree_idx, td_error in zip(tree_indices, td_errors):
            priority = (abs(float(td_error)) + self.eps) ** self.alpha
            self.tree.tree[tree_idx] = priority
            # Propagate up
            idx = tree_idx
            while idx != 0:
                idx = (idx - 1) // 2
                left = 2 * idx + 1
                right = left + 1
                self.tree.tree[idx] = self.tree.tree[left] + self.tree.tree[right]
