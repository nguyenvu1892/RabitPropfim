"""
V3.6 Contrastive Memory -- Stores WIN/LOSS trade pairs for contrastive learning.

Instead of copying expert trades (Imitation Learning), the bot learns to DISTINGUISH
what makes a trade WIN vs LOSS by comparing embeddings of trades in similar contexts.
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
    ):
        """Add a completed trade to the appropriate buffer."""
        entry = {
            "obs": obs.astype(np.float32),
            "action": action,
            "pnl": float(pnl),
            "regime": regime,
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
            wins = [{"obs": e["obs"].tolist(), "action": e["action"], "pnl": e["pnl"], "regime": e["regime"]} for e in self.wins[sym]]
            losses = [{"obs": e["obs"].tolist(), "action": e["action"], "pnl": e["pnl"], "regime": e["regime"]} for e in self.losses[sym]]
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
                        mem.wins[sym].append({"obs": np.array(e["obs"], dtype=np.float32), "action": e["action"], "pnl": e["pnl"], "regime": e.get("regime", "unknown")})
            if losses_path.exists():
                with open(losses_path) as f:
                    for e in json.load(f):
                        mem.losses[sym].append({"obs": np.array(e["obs"], dtype=np.float32), "action": e["action"], "pnl": e["pnl"], "regime": e.get("regime", "unknown")})
        logger.info("ContrastiveMemory loaded from %s (%d entries)", path, mem.total_entries())
        return mem


def contrastive_loss(
    model,
    win_obs: torch.Tensor,
    loss_obs: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE contrastive loss.
    
    Pushes WIN embeddings apart from LOSS embeddings in the learned space.
    Win-win pairs should be close, win-loss pairs should be far.
    
    Args:
        model: AttentionPPO with get_embedding() method
        win_obs: (B, 400) WIN observations
        loss_obs: (B, 400) LOSS observations  
        temperature: scaling factor for cosine similarity
        
    Returns:
        scalar loss
    """
    embed_win = model.get_embedding(win_obs)    # (B, 128) L2-norm
    embed_loss = model.get_embedding(loss_obs)  # (B, 128) L2-norm

    B = embed_win.shape[0]

    # Similarity matrix: win_i vs loss_j
    # Positive pair: (win_i, loss_i) should have different embeddings
    # We want: win embeddings cluster together, loss embeddings apart from wins
    
    # Build similarity matrix of all pairs
    all_embeds = torch.cat([embed_win, embed_loss], dim=0)  # (2B, 128)
    sim = all_embeds @ all_embeds.T / temperature            # (2B, 2B)

    # Labels: each win_i is positive with other wins, negative with losses
    labels = torch.cat([
        torch.zeros(B, dtype=torch.long),  # wins = class 0
        torch.ones(B, dtype=torch.long),   # losses = class 1
    ]).to(win_obs.device)

    # Mask out self-similarity
    mask = torch.eye(2 * B, dtype=torch.bool, device=win_obs.device)
    sim.masked_fill_(mask, -1e9)

    # InfoNCE: for each win, pull toward other wins, push away from losses
    loss = torch.nn.functional.cross_entropy(sim[:B], labels[:B])

    return loss
