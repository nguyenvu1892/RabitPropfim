"""
V3.8 Contrastive Memory -- Fake Setup Mining + WIN/LOSS pair storage.

Instead of copying expert trades (Imitation Learning), the bot learns to DISTINGUISH
what makes a trade WIN vs LOSS by comparing embeddings of trades in similar contexts.
V3.8: Fake Setup Mining — prioritizes LOSS trades where Micro form (M5/M1) looks beautiful 
(volume spike or OB proximity) but macro context failed it.
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

    def __init__(self):
        self.wins = {s: [] for s in SYMBOLS}
        self.losses = {s: [] for s in SYMBOLS}

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

    def sample_fake_setup_pairs(
        self,
        batch_size: int = 64,
        device: torch.device = None,
        cosine_threshold: float = 0.85,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """
        V3.8 Fake Setup Mining: sample LOSS trades where Micro form is 'beautiful'
        but the trade still lost (likely due to bad Macro context).
        Pairs them with similar WIN trades to teach the Cross-Attention layer.
        """
        if device is None:
            device = torch.device("cpu")

        valid_syms = [s for s in SYMBOLS if len(self.wins[s]) >= 3 and len(self.losses[s]) >= 3]
        if not valid_syms:
            return None

        # Helper to identify "beautiful" micro forms (Fake Setups)
        # obs dims: H1(56) + M15(56) + M5(56) + M1_1(56)... M1_5(56) = 448
        # M5 ob_prox = index 112 (M5 start) + 50 = 162
        # M1_b5 vol_spike = index 392 (M1_b5 start) + 51 = 443
        def is_fake_setup(obs: np.ndarray) -> bool:
            if len(obs) != 448:
                return False
            m5_ob_prox = obs[162]
            m1_last_vol = obs[443]
            # Beautiful if very close to M5 OB or huge M1 volume spike
            return m1_last_vol > 0.5 or m5_ob_prox < 0.2

        hard_win_obs = []
        hard_loss_obs = []

        for sym in valid_syms:
            losses = list(self.losses[sym])
            wins = list(self.wins[sym])
            if len(losses) < 5 or len(wins) < 5:
                continue

            # Step 1: Filter LOSS trades to find Fake Setups
            fake_losses = [e for e in losses if is_fake_setup(e["obs"])]
            
            # If no fake setups, fallback to top-confidence losses
            if len(fake_losses) == 0:
                losses_sorted = sorted(losses, key=lambda e: e.get("confidence", 0), reverse=True)
                fake_losses = losses_sorted[:max(3, int(len(losses) * 0.1))]

            # Step 2: Build WIN obs matrix for cosine similarity
            win_obs_np = np.array([w["obs"] for w in wins])  # (N_win, obs_dim)
            win_norms = np.linalg.norm(win_obs_np, axis=1, keepdims=True) + 1e-8
            win_normalized = win_obs_np / win_norms

            # Step 3: For each fake setup loss, find most similar WIN
            for loss_entry in fake_losses:
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
            logger.debug("Not enough fake setups (%d), falling back to regular", len(hard_win_obs))
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

    def save(self, path: Path, append: bool = True):
        """Save Master Vault to disk in JSONL format."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        
        wins_file = path / "master_vault_wins.jsonl"
        losses_file = path / "master_vault_losses.jsonl"
        
        with open(wins_file, mode) as fw:
            for sym in SYMBOLS:
                for e in self.wins[sym]:
                    e_copy = dict(e)
                    e_copy["symbol"] = sym
                    e_copy["obs"] = e["obs"].tolist()
                    fw.write(json.dumps(e_copy) + "\n")
                    
        with open(losses_file, mode) as fl:
            for sym in SYMBOLS:
                for e in self.losses[sym]:
                    e_copy = dict(e)
                    e_copy["symbol"] = sym
                    e_copy["obs"] = e["obs"].tolist()
                    fl.write(json.dumps(e_copy) + "\n")
                    
        logger.info("Master Vault saved to %s (%d new entries, append=%s)", path, self.total_entries(), append)

    @classmethod
    def load(cls, path: Path) -> ContrastiveMemory:
        """Load Master Vault memory from disk."""
        mem = cls()
        path = Path(path)
        if not path.exists():
            return mem
            
        wins_path = path / "master_vault_wins.jsonl"
        losses_path = path / "master_vault_losses.jsonl"
        
        if wins_path.exists():
            with open(wins_path) as f:
                for line in f:
                    if not line.strip(): continue
                    e = json.loads(line)
                    sym = e.get("symbol")
                    if sym in mem.wins:
                        e["obs"] = np.array(e["obs"], dtype=np.float32)
                        mem.wins[sym].append(e)
                        
        if losses_path.exists():
            with open(losses_path) as f:
                for line in f:
                    if not line.strip(): continue
                    e = json.loads(line)
                    sym = e.get("symbol")
                    if sym in mem.losses:
                        e["obs"] = np.array(e["obs"], dtype=np.float32)
                        mem.losses[sym].append(e)
                        
        logger.info("Master Vault loaded from %s (%d entries)", path, mem.total_entries())
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

