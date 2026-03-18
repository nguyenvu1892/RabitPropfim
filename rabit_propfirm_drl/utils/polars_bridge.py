"""
Polars ↔ PyTorch Bridge — Seamless conversion between Polars DataFrames and Tensors.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import polars as pl
import torch


def polars_to_tensor(
    df: pl.DataFrame,
    columns: list[str] | None = None,
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """
    Convert Polars DataFrame (or selected columns) to PyTorch Tensor.

    Args:
        df: Input Polars DataFrame
        columns: Columns to include. None = all numeric columns.
        dtype: PyTorch data type
        device: Target device

    Returns:
        Tensor of shape (n_rows, n_columns)
    """
    if columns is None:
        # Auto-select numeric columns
        columns = [
            col for col, dt in zip(df.columns, df.dtypes)
            if dt in (pl.Float32, pl.Float64, pl.Int32, pl.Int64, pl.UInt32, pl.UInt64)
        ]

    np_array = df.select(columns).to_numpy().astype(np.float32)
    tensor = torch.from_numpy(np_array).to(dtype=dtype, device=device)
    return tensor


def tensor_to_polars(
    tensor: torch.Tensor,
    columns: list[str],
) -> pl.DataFrame:
    """
    Convert PyTorch Tensor back to Polars DataFrame.

    Args:
        tensor: Input tensor of shape (n_rows, n_columns)
        columns: Column names (must match tensor.shape[1])

    Returns:
        Polars DataFrame
    """
    if tensor.ndim != 2:
        raise ValueError(f"Expected 2D tensor, got {tensor.ndim}D")
    if tensor.shape[1] != len(columns):
        raise ValueError(
            f"Tensor has {tensor.shape[1]} columns but {len(columns)} names given"
        )

    np_array = tensor.detach().cpu().numpy()
    return pl.DataFrame(
        {col: np_array[:, i] for i, col in enumerate(columns)}
    )


def polars_to_sequences(
    df: pl.DataFrame,
    columns: list[str],
    seq_length: int,
    stride: int = 1,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Convert DataFrame to sliding window sequences for Transformer input.

    Args:
        df: Input DataFrame
        columns: Feature columns to extract
        seq_length: Number of timesteps per sequence
        stride: Step between consecutive sequences
        dtype: PyTorch data type

    Returns:
        Tensor of shape (n_sequences, seq_length, n_features)
    """
    np_array = df.select(columns).to_numpy().astype(np.float32)
    n_rows = len(np_array)

    if n_rows < seq_length:
        raise ValueError(
            f"DataFrame has {n_rows} rows but seq_length is {seq_length}"
        )

    # Create sliding windows
    sequences = []
    for i in range(0, n_rows - seq_length + 1, stride):
        sequences.append(np_array[i : i + seq_length])

    stacked = np.stack(sequences, axis=0)
    return torch.from_numpy(stacked).to(dtype=dtype)
