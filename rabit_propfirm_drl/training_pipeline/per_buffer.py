"""
Prioritized Experience Replay (PER) Buffer -- Multi-TF variant.

Key features:
- SumTree data structure for O(log n) priority sampling
- Importance sampling (IS) weights with beta annealing
- Supports structured multi-TF observations (M5 + H1 + H4)
- Efficient batch sampling with GPU-ready tensor output

Reference: Schaul et al. (2015) "Prioritized Experience Replay"

Architecture:
    SumTree (binary tree)
    [root = sum of all priorities]
        /           \\
    [sum L]       [sum R]
      / \\           / \\
    [p1] [p2]   [p3] [p4]    <-- leaf priorities

    Add: O(log n) -- update leaf + propagate up
    Sample: O(log n) -- segment sampling + tree traversal
    Update: O(log n) -- change priority + propagate
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

logger = logging.getLogger(__name__)


# -----------------------------------------------
# SumTree for O(log n) priority sampling
# -----------------------------------------------

class SumTree:
    """
    Binary tree where each leaf stores a priority value and
    internal nodes store the sum of their children.

    This allows O(log n) proportional sampling:
    - Pick a random value in [0, total_priority)
    - Traverse tree to find the leaf

    Capacity MUST be a power of 2 for correct indexing.
    (We round up to next power of 2 internally.)
    """

    def __init__(self, capacity: int) -> None:
        # Round up to next power of 2
        self.capacity = 1
        while self.capacity < capacity:
            self.capacity *= 2

        # Tree array: internal nodes + leaves
        # Internal nodes: [0 .. capacity-2]
        # Leaves: [capacity-1 .. 2*capacity-2]
        self.tree = np.zeros(2 * self.capacity - 1, dtype=np.float64)
        self.n_entries = 0

    def _propagate_up(self, idx: int) -> None:
        """Propagate priority change from leaf to root. O(log n)."""
        parent = (idx - 1) // 2
        # Recompute parent from children
        left = 2 * parent + 1
        right = left + 1
        self.tree[parent] = self.tree[left] + self.tree[right]
        if parent > 0:
            self._propagate_up(parent)

    def _retrieve(self, idx: int, value: float) -> int:
        """Find leaf index for a given cumulative sum value. O(log n)."""
        left = 2 * idx + 1

        # Reached a leaf
        if left >= len(self.tree):
            return idx

        if value <= self.tree[left]:
            return self._retrieve(left, value)
        else:
            return self._retrieve(left + 1, value - self.tree[left])

    @property
    def total(self) -> float:
        """Total priority sum (root node)."""
        return float(self.tree[0])

    @property
    def max_priority(self) -> float:
        """Maximum priority across all active leaves."""
        if self.n_entries == 0:
            return 1.0
        leaf_start = self.capacity - 1
        leaf_end = leaf_start + self.n_entries
        return float(np.max(self.tree[leaf_start:leaf_end]))

    @property
    def min_priority(self) -> float:
        """Minimum non-zero priority across all active leaves."""
        if self.n_entries == 0:
            return 1.0
        leaf_start = self.capacity - 1
        leaf_end = leaf_start + self.n_entries
        leaves = self.tree[leaf_start:leaf_end]
        nonzero = leaves[leaves > 0]
        return float(np.min(nonzero)) if len(nonzero) > 0 else 1.0

    def update(self, data_idx: int, priority: float) -> None:
        """Set priority for data index and propagate. O(log n)."""
        tree_idx = data_idx + self.capacity - 1
        self.tree[tree_idx] = priority
        self._propagate_up(tree_idx)

    def add(self, data_idx: int, priority: float) -> None:
        """Add or update a leaf. Tracks n_entries."""
        self.update(data_idx, priority)
        if data_idx >= self.n_entries:
            self.n_entries = data_idx + 1

    def get(self, value: float) -> tuple[int, float, int]:
        """
        Sample a leaf by cumulative sum.

        Args:
            value: Random value in [0, total_priority)

        Returns:
            (tree_index, priority, data_index)
        """
        tree_idx = self._retrieve(0, value)
        data_idx = tree_idx - self.capacity + 1
        return tree_idx, float(self.tree[tree_idx]), data_idx


# -----------------------------------------------
# Multi-TF PER Buffer
# -----------------------------------------------

class PERBuffer:
    """
    Prioritized Experience Replay buffer for multi-TF SAC.

    Stores structured observations (M5, H1, H4 sequences) instead of
    flat state vectors. Uses proportional prioritization:

        P(i) = p_i^alpha / sum(p_k^alpha)

    With importance sampling weights:

        w_i = (N * P(i))^(-beta) / max(w)

    Beta anneals from beta_start to 1.0 over beta_frames steps,
    which corrects the bias introduced by non-uniform sampling.

    Args:
        capacity: Maximum number of transitions to store
        alpha: Priority exponent (0=uniform, 1=full priority). Default 0.6
        beta_start: Initial IS exponent. Default 0.4
        beta_frames: Steps to anneal beta to 1.0. Default 500_000
        eps: Small constant for non-zero priority. Default 1e-6
        seq_m5: M5 sequence length. Default 64
        seq_h1: H1 sequence length. Default 24
        seq_h4: H4 sequence length. Default 30
        n_features: Feature dimension per bar. Default 28
        action_dim: Action dimension. Default 4
    """

    def __init__(
        self,
        capacity: int = 100_000,
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 500_000,
        eps: float = 1e-6,
        seq_m5: int = 64,
        seq_h1: int = 24,
        seq_h4: int = 30,
        n_features: int = 28,
        action_dim: int = 4,
    ) -> None:
        self.capacity = capacity
        self.alpha = alpha
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.eps = eps

        # SumTree for priority sampling
        self.tree = SumTree(capacity)

        # Pre-allocated storage: multi-TF observations
        self.m5 = np.zeros((capacity, seq_m5, n_features), dtype=np.float32)
        self.h1 = np.zeros((capacity, seq_h1, n_features), dtype=np.float32)
        self.h4 = np.zeros((capacity, seq_h4, n_features), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_m5 = np.zeros((capacity, seq_m5, n_features), dtype=np.float32)
        self.next_h1 = np.zeros((capacity, seq_h1, n_features), dtype=np.float32)
        self.next_h4 = np.zeros((capacity, seq_h4, n_features), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)

        self._pointer = 0
        self._size = 0
        self._frame = 0  # For beta annealing

    @property
    def size(self) -> int:
        """Current number of stored transitions."""
        return self._size

    @property
    def beta(self) -> float:
        """Current beta value (anneals from beta_start to 1.0)."""
        fraction = min(self._frame / max(self.beta_frames, 1), 1.0)
        return self.beta_start + fraction * (1.0 - self.beta_start)

    def add(
        self,
        obs: tuple[np.ndarray, np.ndarray, np.ndarray],
        action: np.ndarray,
        reward: float,
        next_obs: tuple[np.ndarray, np.ndarray, np.ndarray],
        done: bool,
        td_error: float | None = None,
    ) -> None:
        """
        Add a transition with priority based on |td_error|.

        If td_error is None, uses max existing priority (new experiences
        get highest priority so they're sampled at least once).

        Args:
            obs: (m5, h1, h4) observation tuple
            action: Action array
            reward: Scalar reward
            next_obs: (next_m5, next_h1, next_h4) tuple
            done: Episode terminated
            td_error: Optional TD-error for initial priority
        """
        # Compute priority
        if td_error is not None:
            priority = (abs(td_error) + self.eps) ** self.alpha
        else:
            max_p = self.tree.max_priority
            priority = max(max_p, 1.0)

        # Store data
        idx = self._pointer
        self.m5[idx] = obs[0]
        self.h1[idx] = obs[1]
        self.h4[idx] = obs[2]
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_m5[idx] = next_obs[0]
        self.next_h1[idx] = next_obs[1]
        self.next_h4[idx] = next_obs[2]
        self.dones[idx] = float(done)

        # Add to SumTree
        self.tree.add(idx, priority)

        self._pointer = (self._pointer + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def sample(
        self,
        batch_size: int,
        device: torch.device = torch.device("cpu"),
    ) -> dict:
        """
        Sample a prioritized batch.

        Uses stratified sampling: divide total priority into equal
        segments, sample one point per segment. This reduces variance.

        Returns:
            dict with keys: m5, h1, h4, act, rew, next_m5, next_h1,
                           next_h4, done, is_weights, tree_indices
        """
        self._frame += 1

        indices = []
        priorities = []
        tree_indices = []

        # Stratified sampling: divide [0, total) into batch_size segments
        total = max(self.tree.total, self.eps)
        segment = total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            value = np.random.uniform(a, b)
            tree_idx, priority, data_idx = self.tree.get(value)

            # Clamp to valid range
            data_idx = max(0, min(data_idx, self._size - 1))

            tree_indices.append(tree_idx)
            indices.append(data_idx)
            priorities.append(max(priority, self.eps))

        # Importance sampling weights: w_i = (N * P(i))^(-beta) / max(w)
        priorities_arr = np.array(priorities, dtype=np.float64)
        probs = priorities_arr / total
        beta = self.beta
        is_weights = (self._size * probs) ** (-beta)
        is_weights /= max(is_weights.max(), self.eps)  # Normalize to max=1
        is_weights = is_weights.astype(np.float32)

        idx_arr = np.array(indices)

        return {
            "m5": torch.FloatTensor(self.m5[idx_arr]).to(device),
            "h1": torch.FloatTensor(self.h1[idx_arr]).to(device),
            "h4": torch.FloatTensor(self.h4[idx_arr]).to(device),
            "act": torch.FloatTensor(self.actions[idx_arr]).to(device),
            "rew": torch.FloatTensor(self.rewards[idx_arr]).to(device),
            "next_m5": torch.FloatTensor(self.next_m5[idx_arr]).to(device),
            "next_h1": torch.FloatTensor(self.next_h1[idx_arr]).to(device),
            "next_h4": torch.FloatTensor(self.next_h4[idx_arr]).to(device),
            "done": torch.FloatTensor(self.dones[idx_arr]).to(device),
            "is_weights": torch.FloatTensor(is_weights).to(device),
            "tree_indices": tree_indices,
        }

    def update_priorities(
        self, tree_indices: list[int], td_errors: np.ndarray
    ) -> None:
        """
        Update priorities after computing new TD-errors.

        Called after each training step to refine which experiences
        the agent should focus on. High TD-error = model predicts
        poorly = needs more training.

        Args:
            tree_indices: Tree indices from sample()
            td_errors: New TD-errors (absolute values recommended)
        """
        for tree_idx, td_error in zip(tree_indices, td_errors):
            priority = (abs(float(td_error)) + self.eps) ** self.alpha
            # Update leaf and propagate up
            data_idx = tree_idx - self.tree.capacity + 1
            self.tree.update(data_idx, priority)

    def state_dict(self) -> dict:
        """Serialize buffer state for checkpointing."""
        return {
            "pointer": self._pointer,
            "size": self._size,
            "frame": self._frame,
            "tree": self.tree.tree.copy(),
            "tree_n_entries": self.tree.n_entries,
        }

    def load_state_dict(self, state: dict) -> None:
        """Restore buffer state from checkpoint."""
        self._pointer = state["pointer"]
        self._size = state["size"]
        self._frame = state["frame"]
        self.tree.tree = state["tree"].copy()
        self.tree.n_entries = state["tree_n_entries"]
