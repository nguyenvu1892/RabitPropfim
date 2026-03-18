"""
Tests for Running Normalizer (T1.4.2).

Validates:
- Normalized output ≈ N(0, 1) for large sample
- Save/load state produces identical results
- Clip at ±5σ works correctly
- Batch update matches row-by-row update
- Edge cases (single sample, zero variance)
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from data_engine.normalizer import RunningNormalizer


class TestBasicNormalization:

    def test_mean_approximately_zero(self) -> None:
        """After normalizing 10K samples, mean should be ≈ 0."""
        rng = np.random.default_rng(42)
        data = rng.normal(loc=100.0, scale=15.0, size=(10_000, 5))

        norm = RunningNormalizer(n_features=5)
        norm.update_batch(data)

        normalized = norm.normalize(data)
        assert normalized.shape == data.shape
        col_means = normalized.mean(axis=0)
        assert np.allclose(col_means, 0.0, atol=0.05), f"Means: {col_means}"

    def test_std_approximately_one(self) -> None:
        """After normalizing 10K samples, std should be ≈ 1."""
        rng = np.random.default_rng(42)
        data = rng.normal(loc=50.0, scale=10.0, size=(10_000, 3))

        norm = RunningNormalizer(n_features=3)
        norm.update_batch(data)

        normalized = norm.normalize(data)
        col_stds = normalized.std(axis=0)
        assert np.allclose(col_stds, 1.0, atol=0.05), f"Stds: {col_stds}"


class TestClipping:

    def test_clip_at_5_sigma(self) -> None:
        """Values beyond ±5σ should be clipped."""
        norm = RunningNormalizer(n_features=2, clip_sigma=5.0)

        # Train on standard normal data
        data = np.random.default_rng(42).normal(0, 1, size=(10_000, 2))
        norm.update_batch(data)

        # Test with extreme values
        extreme = np.array([[100.0, -100.0]])  # Way beyond 5σ
        result = norm.normalize(extreme)

        assert np.all(result <= 5.0), f"Got values > 5σ: {result}"
        assert np.all(result >= -5.0), f"Got values < -5σ: {result}"

    def test_custom_clip_sigma(self) -> None:
        """Test clip_sigma=3.0."""
        norm = RunningNormalizer(n_features=1, clip_sigma=3.0)
        data = np.random.default_rng(42).normal(0, 1, size=(10_000, 1))
        norm.update_batch(data)

        extreme = np.array([[50.0]])
        result = norm.normalize(extreme)
        assert np.all(result <= 3.0)


class TestSerialization:

    def test_save_load_produces_same_output(self, tmp_path: Path) -> None:
        """Save/load state should produce identical normalization."""
        rng = np.random.default_rng(42)
        data = rng.normal(loc=5.0, scale=2.0, size=(5000, 4))

        # Create and train normalizer
        norm1 = RunningNormalizer(n_features=4)
        norm1.update_batch(data)

        # Save
        save_path = tmp_path / "normalizer.json"
        norm1.save(save_path)

        # Load
        norm2 = RunningNormalizer.load(save_path)

        # Compare outputs
        test_data = rng.normal(loc=5.0, scale=2.0, size=(100, 4))
        out1 = norm1.normalize(test_data)
        out2 = norm2.normalize(test_data)

        assert np.allclose(out1, out2, atol=1e-10), "Save/load changed normalization output"

    def test_save_load_preserves_count(self, tmp_path: Path) -> None:
        norm = RunningNormalizer(n_features=2)
        data = np.random.default_rng(42).normal(0, 1, size=(1234, 2))
        norm.update_batch(data)

        save_path = tmp_path / "norm.json"
        norm.save(save_path)
        loaded = RunningNormalizer.load(save_path)

        assert loaded.count == 1234


class TestBatchVsRowUpdate:

    def test_batch_equals_row_by_row(self) -> None:
        """Batch update should produce same stats as row-by-row."""
        rng = np.random.default_rng(42)
        data = rng.normal(3.0, 1.5, size=(500, 3))

        # Row-by-row
        norm_row = RunningNormalizer(n_features=3)
        norm_row.update(data)  # This does row-by-row internally

        # Batch
        norm_batch = RunningNormalizer(n_features=3)
        norm_batch.update_batch(data)

        assert np.allclose(norm_row.mean, norm_batch.mean, atol=1e-8)
        assert np.allclose(norm_row.var, norm_batch.var, atol=1e-6)


class TestEdgeCases:

    def test_single_sample(self) -> None:
        """Normalizer should handle a single sample without error."""
        norm = RunningNormalizer(n_features=2)
        norm.update(np.array([1.0, 2.0]))
        result = norm.normalize(np.array([1.0, 2.0]))
        assert result.shape == (2,)
        assert np.all(np.isfinite(result))

    def test_wrong_feature_count_raises(self) -> None:
        """Mismatched feature count should raise."""
        norm = RunningNormalizer(n_features=3)
        with pytest.raises(ValueError):
            norm.update(np.array([1.0, 2.0]))  # 2 features, expected 3

    def test_denormalize_roundtrip(self) -> None:
        """normalize → denormalize should recover original values."""
        rng = np.random.default_rng(42)
        data = rng.normal(10.0, 3.0, size=(5000, 2))

        norm = RunningNormalizer(n_features=2)
        norm.update_batch(data)

        test = rng.normal(10.0, 3.0, size=(50, 2))
        normalized = norm.normalize(test)
        recovered = norm.denormalize(normalized)

        # Should be close to original (exact if no clipping)
        assert np.allclose(recovered, test, atol=0.1)

    def test_repr(self) -> None:
        norm = RunningNormalizer(n_features=5)
        repr_str = repr(norm)
        assert "n_features=5" in repr_str
        assert "count=0" in repr_str
