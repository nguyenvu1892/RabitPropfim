"""
Model Registry — Version control for trained models.

Features:
- Saves model snapshots with metadata (metrics, curriculum stage, timestamp)
- Tracks best model by evaluation metric
- Supports rollback to any previous version
- JSON manifest for human-readable registry
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch

logger = logging.getLogger(__name__)


@dataclass
class ModelVersion:
    """Metadata for a registered model version."""
    version_id: int
    timestamp: str
    checkpoint_path: str
    metrics: dict[str, float]
    curriculum_stage: str = ""
    training_steps: int = 0
    is_best: bool = False
    notes: str = ""


class ModelRegistry:
    """
    Registry for managing model versions with rollback capability.

    Models are saved in:
        registry_dir/
        ├── manifest.json          # Human-readable version list
        ├── v001/checkpoint.pt     # Model checkpoint
        ├── v002/checkpoint.pt
        └── best/checkpoint.pt     # Symlink/copy to best
    """

    def __init__(self, registry_dir: Path | str) -> None:
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        self.versions: list[ModelVersion] = []
        self._best_metric_key = "eval_reward"
        self._best_metric_value = float("-inf")
        self._best_version_id: int | None = None

        # Load existing manifest
        self._load_manifest()

    def _manifest_path(self) -> Path:
        return self.registry_dir / "manifest.json"

    def _version_dir(self, version_id: int) -> Path:
        return self.registry_dir / f"v{version_id:03d}"

    def _load_manifest(self) -> None:
        """Load manifest from disk if exists."""
        path = self._manifest_path()
        if path.exists():
            with open(path, "r") as f:
                data = json.load(f)
                self.versions = [
                    ModelVersion(**v) for v in data.get("versions", [])
                ]
                self._best_version_id = data.get("best_version_id")
                self._best_metric_value = data.get("best_metric_value", float("-inf"))

    def _save_manifest(self) -> None:
        """Save manifest to disk."""
        data = {
            "best_version_id": self._best_version_id,
            "best_metric_value": self._best_metric_value,
            "versions": [asdict(v) for v in self.versions],
        }
        with open(self._manifest_path(), "w") as f:
            json.dump(data, f, indent=2)

    @property
    def latest_version(self) -> int:
        """Latest registered version ID (0 if none)."""
        return self.versions[-1].version_id if self.versions else 0

    @property
    def best_version(self) -> int | None:
        """Best performing version ID."""
        return self._best_version_id

    def register(
        self,
        checkpoint_state: dict,
        metrics: dict[str, float],
        curriculum_stage: str = "",
        training_steps: int = 0,
        notes: str = "",
    ) -> ModelVersion:
        """
        Register a new model version.

        Args:
            checkpoint_state: PyTorch state dict to save
            metrics: Evaluation metrics (must include `eval_reward`)
            curriculum_stage: Current curriculum stage name
            training_steps: Total training steps at registration
            notes: Optional human-readable notes

        Returns:
            ModelVersion of the newly registered model
        """
        version_id = self.latest_version + 1
        version_dir = self._version_dir(version_id)
        version_dir.mkdir(parents=True, exist_ok=True)

        # Save checkpoint
        checkpoint_path = version_dir / "checkpoint.pt"
        torch.save(checkpoint_state, checkpoint_path)

        # Check if best
        eval_metric = metrics.get(self._best_metric_key, float("-inf"))
        is_best = eval_metric > self._best_metric_value

        if is_best:
            self._best_metric_value = eval_metric
            self._best_version_id = version_id

            # Copy to best/ directory
            best_dir = self.registry_dir / "best"
            best_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(checkpoint_path, best_dir / "checkpoint.pt")

            logger.info(
                "🏆 New best model: v%03d (metric=%.4f)",
                version_id, eval_metric,
            )

        # Create version entry
        version = ModelVersion(
            version_id=version_id,
            timestamp=datetime.now(timezone.utc).isoformat(),
            checkpoint_path=str(checkpoint_path),
            metrics=metrics,
            curriculum_stage=curriculum_stage,
            training_steps=training_steps,
            is_best=is_best,
            notes=notes,
        )
        self.versions.append(version)
        self._save_manifest()

        logger.info(
            "Registered model v%03d (steps=%d, stage=%s)",
            version_id, training_steps, curriculum_stage,
        )
        return version

    def load_version(
        self, version_id: int, device: str = "cpu"
    ) -> dict:
        """Load a specific model version checkpoint."""
        version_dir = self._version_dir(version_id)
        checkpoint_path = version_dir / "checkpoint.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        return torch.load(checkpoint_path, map_location=device, weights_only=False)

    def load_best(self, device: str = "cpu") -> dict:
        """Load the best model checkpoint."""
        best_path = self.registry_dir / "best" / "checkpoint.pt"
        if not best_path.exists():
            raise FileNotFoundError("No best model registered yet")
        return torch.load(best_path, map_location=device, weights_only=False)

    def rollback(self, version_id: int) -> ModelVersion:
        """
        Rollback to a specific version (marks it as the new best).

        Returns:
            ModelVersion of the rollback target
        """
        target = None
        for v in self.versions:
            if v.version_id == version_id:
                target = v
                break

        if target is None:
            raise ValueError(f"Version {version_id} not found in registry")

        # Copy to best/
        src = self._version_dir(version_id) / "checkpoint.pt"
        dst_dir = self.registry_dir / "best"
        dst_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst_dir / "checkpoint.pt")

        self._best_version_id = version_id
        self._best_metric_value = target.metrics.get(
            self._best_metric_key, self._best_metric_value
        )
        self._save_manifest()

        logger.warning("🔄 ROLLBACK to model v%03d", version_id)
        return target

    def list_versions(self) -> list[dict]:
        """Return summary of all registered versions."""
        return [asdict(v) for v in self.versions]
