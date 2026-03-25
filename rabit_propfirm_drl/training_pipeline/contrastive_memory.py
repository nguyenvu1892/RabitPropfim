"""
V3.7 Contrastive Memory -- Hard Negative Mining + WIN/LOSS pair storage.

Instead of copying expert trades (Imitation Learning), the bot learns to DISTINGUISH
what makes a trade WIN vs LOSS by comparing embeddings of trades in similar contexts.
V3.7: Hard Negative Mining — prioritizes most "painful" losses (high confidence + similar context).
"""
from __future__ import annotations
import json
import logging
import random
from collections import deque
from pathlib import Path

import numpy as np
import torch

logger = logging.getLogger("contrastive_memory")

SYMBOLS = ["XAUUSD", "BTCUSD", "ETHUSD", "US30_cash", "US100_cash"]


class ContrastiveMemory:
    """
    Stores WIN and LOSS trades per symbol for contrastive pair sampling.
    
    Key insight: pairs MUST be from the same symbol (and ideally same regime)
    to teach the bot meaningful differences.
    """

    def __init__(self, max_per_symbol: int = 500):
        self.max_per_symbol = max_per_symbol
        self.wins = {s: deque(maxlen=max_per_symbol) for s in SYMBOLS}
        self.losses = {s: deque(maxlen=max_per_symbol) for s in SYMBOLS}

    def add_trade(
        self,
        obs: np.ndarray,
        action: int,
        pnl: float,
        symbol: str,
        regime: str = "unknown",
        confidence: float = 0.0,
    ):
        """Add a completed trade to the appropriate buffer."""
        entry = {
            "obs": obs.astype(np.float32),
            "action": action,
            "pnl": float(pnl),
            "regime": regime,
            "confidence": float(confidence),
        }
        if pnl > 0:
            self.wins[symbol].append(entry)
        else:
            self.losses[symbol].append(entry)

    def can_sample(self, min_per_symbol: int = 5) -> bool:
        """Check if we have enough pairs to sample from."""
        for sym in SYMBOLS:
            if len(self.wins[sym]) >= min_per_symbol and len(self.losses[sym]) >= min_per_symbol:
                return True
        return False

    def sample_contrastive_pairs(
        self,
        batch_size: int = 64,
        device: torch.device = None,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """
        Sample (win_obs, loss_obs) pairs from the same symbol.
        
        Returns:
            win_obs: (B, 400) tensor of winning trade observations
            loss_obs: (B, 400) tensor of losing trade observations
        """
        if device is None:
            device = torch.device("cpu")

        # Find symbols with enough data
        valid_syms = [
            s for s in SYMBOLS
            if len(self.wins[s]) >= 3 and len(self.losses[s]) >= 3
        ]
        if not valid_syms:
            return None

        win_obs_list = []
        loss_obs_list = []

        for _ in range(batch_size):
            sym = random.choice(valid_syms)
            
            # Pick a random WIN and LOSS from same symbol
            win_entry = random.choice(self.wins[sym])
            loss_entry = random.choice(self.losses[sym])

            # Prefer same regime if possible
            same_regime_losses = [
                e for e in self.losses[sym] if e["regime"] == win_entry["regime"]
            ]
            if same_regime_losses:
                loss_entry = random.choice(same_regime_losses)

            win_obs_list.append(win_entry["obs"])
            loss_obs_list.append(loss_entry["obs"])

        win_obs = torch.from_numpy(np.array(win_obs_list)).float().to(device)
        loss_obs = torch.from_numpy(np.array(loss_obs_list)).float().to(device)

        return win_obs, loss_obs

    def sample_hard_negative_pairs(
        self,
        model,
        batch_size: int = 64,
        device: torch.device = None,
        top_k_pct: float = 0.20,
        cosine_threshold: float = 0.85,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """
        V3.7 Hard Negative Mining: sample the most "painful" LOSS trades.

        1. Filter top-20% most confident LOSS trades (bot was sure but lost)
        2. For each, find the WIN trade with highest cosine similarity (most similar context)
        3. Only keep pairs where cosine sim > threshold

        Falls back to regular sampling if not enough hard negatives.
        """
        if device is None:
            device = torch.device("cpu")

        valid_syms = [s for s in SYMBOLS if len(self.wins[s]) >= 3 and len(self.losses[s]) >= 3]
        if not valid_syms:
            return None

        hard_win_obs = []
        hard_loss_obs = []

        for sym in valid_syms:
            losses = list(self.losses[sym])
            wins = list(self.wins[sym])
            if len(losses) < 5 or len(wins) < 5:
                continue

            # Step 1: Sort losses by confidence (descending), take top-K%
            losses_sorted = sorted(losses, key=lambda e: e.get("confidence", 0), reverse=True)
            top_k = max(3, int(len(losses_sorted) * top_k_pct))
            hard_losses = losses_sorted[:top_k]

            # Step 2: Build WIN obs matrix for cosine similarity
            win_obs_np = np.array([w["obs"] for w in wins])  # (N_win, obs_dim)
            win_norms = np.linalg.norm(win_obs_np, axis=1, keepdims=True) + 1e-8
            win_normalized = win_obs_np / win_norms

            # Step 3: For each hard loss, find most similar WIN
            for loss_entry in hard_losses:
                loss_obs = loss_entry["obs"]
                loss_norm = np.linalg.norm(loss_obs) + 1e-8
                loss_normalized = loss_obs / loss_norm

                # Cosine similarity with all wins
                similarities = win_normalized @ loss_normalized  # (N_win,)
                best_idx = int(np.argmax(similarities))
                best_sim = float(similarities[best_idx])

                if best_sim >= cosine_threshold:
                    hard_win_obs.append(wins[best_idx]["obs"])
                    hard_loss_obs.append(loss_obs)

        if len(hard_win_obs) < 3:
            # Fallback to regular sampling
            logger.debug("Not enough hard negatives (%d), falling back to regular", len(hard_win_obs))
            return self.sample_contrastive_pairs(batch_size, device)

        # Sample batch_size pairs from hard negatives
        indices = [random.randint(0, len(hard_win_obs) - 1) for _ in range(batch_size)]
        win_batch = torch.from_numpy(np.array([hard_win_obs[i] for i in indices])).float().to(device)
        loss_batch = torch.from_numpy(np.array([hard_loss_obs[i] for i in indices])).float().to(device)

        return win_batch, loss_batch

    def stats(self) -> dict:
        """Return current memory statistics."""
        return {
            sym: {"wins": len(self.wins[sym]), "losses": len(self.losses[sym])}
            for sym in SYMBOLS
        }

    def total_entries(self) -> int:
        return sum(
            len(self.wins[s]) + len(self.losses[s]) for s in SYMBOLS
        )

    def save(self, path: Path):
        """Save memory to disk."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for sym in SYMBOLS:
            safe = sym.replace(".", "_")
            wins = [{"obs": e["obs"].tolist(), "action": e["action"], "pnl": e["pnl"], "regime": e["regime"], "confidence": e.get("confidence", 0)} for e in self.wins[sym]]
            losses = [{"obs": e["obs"].tolist(), "action": e["action"], "pnl": e["pnl"], "regime": e["regime"], "confidence": e.get("confidence", 0)} for e in self.losses[sym]]
            with open(path / f"{safe}_wins.json", "w") as f:
                json.dump(wins, f)
            with open(path / f"{safe}_losses.json", "w") as f:
                json.dump(losses, f)
        logger.info("ContrastiveMemory saved to %s (%d entries)", path, self.total_entries())

    @classmethod
    def load(cls, path: Path, max_per_symbol: int = 500) -> ContrastiveMemory:
        """Load memory from disk."""
        mem = cls(max_per_symbol=max_per_symbol)
        path = Path(path)
        if not path.exists():
            return mem
        for sym in SYMBOLS:
            safe = sym.replace(".", "_")
            wins_path = path / f"{safe}_wins.json"
            losses_path = path / f"{safe}_losses.json"
            if wins_path.exists():
                with open(wins_path) as f:
                    for e in json.load(f):
                        mem.wins[sym].append({"obs": np.array(e["obs"], dtype=np.float32), "action": e["action"], "pnl": e["pnl"], "regime": e.get("regime", "unknown"), "confidence": e.get("confidence", 0)})
            if losses_path.exists():
                with open(losses_path) as f:
                    for e in json.load(f):
                        mem.losses[sym].append({"obs": np.array(e["obs"], dtype=np.float32), "action": e["action"], "pnl": e["pnl"], "regime": e.get("regime", "unknown"), "confidence": e.get("confidence", 0)})
        logger.info("ContrastiveMemory loaded from %s (%d entries)", path, mem.total_entries())
        return mem


def contrastive_loss(
    model,
    win_obs: torch.Tensor,
    loss_obs: torch.Tensor,
    margin: float = 1.0,
) -> torch.Tensor:
    """
    Margin-based contrastive loss.
    
    WIN embeddings should cluster together (small pairwise distance),
    LOSS embeddings should be far from WIN embeddings (distance > margin).
    
    Loss = mean_win_pairs(d²) + mean_cross_pairs(max(0, margin - d)²)
    
    Args:
        model: AttentionPPO with get_embedding() method
        win_obs: (B, 400) WIN observations
        loss_obs: (B, 400) LOSS observations  
        margin: minimum distance between WIN and LOSS clusters
        
    Returns:
        scalar loss (bounded, numerically stable)
    """
    embed_win = model.get_embedding(win_obs)    # (B, 128) L2-norm
    embed_loss = model.get_embedding(loss_obs)  # (B, 128) L2-norm

    # 1. Pull WIN embeddings together (positive pairs)
    #    Distance between consecutive win pairs
    win_dist = torch.norm(embed_win[:-1] - embed_win[1:], dim=1)  # (B-1,)
    pull_loss = (win_dist ** 2).mean()

    # 2. Push WIN away from LOSS (negative pairs) 
    #    Paired distance: win_i vs loss_i
    cross_dist = torch.norm(embed_win - embed_loss, dim=1)  # (B,)
    push_loss = (torch.clamp(margin - cross_dist, min=0) ** 2).mean()

    return pull_loss + push_loss

