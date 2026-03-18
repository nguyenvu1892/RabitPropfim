"""
MT5 Data Fetcher — Pulls tick/OHLCV data from MetaTrader 5 and stores as Parquet.

Supports:
- Historical M1 data fetch (5 years)
- Incremental fetch (only new data since last pull)
- Multiple symbols
- Compressed Parquet output via Polars

Requires: MetaTrader5 terminal running on Windows.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import polars as pl

logger = logging.getLogger(__name__)

# Graceful import — MT5 only works on Windows with terminal installed
try:
    import MetaTrader5 as mt5

    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False
    mt5 = None  # type: ignore[assignment]


# ─────────────────────────────────────────────
# MT5 Connection Manager
# ─────────────────────────────────────────────

class MT5Connection:
    """Context manager for MT5 terminal connection."""

    def __init__(self, path: Optional[str] = None) -> None:
        self._path = path
        self._connected = False

    def __enter__(self) -> "MT5Connection":
        if not MT5_AVAILABLE:
            raise RuntimeError(
                "MetaTrader5 package not installed. "
                "Install with: pip install MetaTrader5"
            )
        kwargs: dict = {}
        if self._path:
            kwargs["path"] = self._path
        if not mt5.initialize(**kwargs):
            error = mt5.last_error()
            raise ConnectionError(
                f"MT5 initialize failed: {error}. "
                "Ensure MT5 terminal is running."
            )
        self._connected = True
        logger.info("MT5 connected: %s", mt5.terminal_info().name)
        return self

    def __exit__(self, *args: object) -> None:
        if self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("MT5 disconnected")


# ─────────────────────────────────────────────
# Data Fetcher
# ─────────────────────────────────────────────

# MT5 timeframe mapping
TIMEFRAME_MAP = {
    "M1": mt5.TIMEFRAME_M1 if MT5_AVAILABLE else 1,
    "M5": mt5.TIMEFRAME_M5 if MT5_AVAILABLE else 5,
    "M15": mt5.TIMEFRAME_M15 if MT5_AVAILABLE else 15,
    "M30": mt5.TIMEFRAME_M30 if MT5_AVAILABLE else 30,
    "H1": mt5.TIMEFRAME_H1 if MT5_AVAILABLE else 60,
    "H4": mt5.TIMEFRAME_H4 if MT5_AVAILABLE else 240,
    "D1": mt5.TIMEFRAME_D1 if MT5_AVAILABLE else 1440,
}


def fetch_ohlcv(
    symbol: str,
    timeframe: str = "M1",
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    n_bars: Optional[int] = None,
) -> pl.DataFrame:
    """
    Fetch OHLCV data from MT5 for a single symbol/timeframe.

    Args:
        symbol: MT5 symbol name (e.g., "EURUSD", "XAUUSDm")
        timeframe: One of M1, M5, M15, M30, H1, H4, D1
        start: Start datetime (UTC). Defaults to 5 years ago.
        end: End datetime (UTC). Defaults to now.
        n_bars: If set, fetch last N bars instead of date range.

    Returns:
        Polars DataFrame with columns: [time, open, high, low, close, volume, spread]
    """
    if not MT5_AVAILABLE:
        raise RuntimeError("MetaTrader5 not available")

    tf = TIMEFRAME_MAP.get(timeframe.upper())
    if tf is None:
        raise ValueError(f"Unknown timeframe: {timeframe}. Use one of {list(TIMEFRAME_MAP)}")

    if n_bars is not None:
        rates = mt5.copy_rates_from_pos(symbol, tf, 0, n_bars)
    else:
        if end is None:
            end = datetime.now(tz=timezone.utc)
        if start is None:
            start = end - timedelta(days=365 * 5)  # 5 years
        rates = mt5.copy_rates_range(symbol, tf, start, end)

    if rates is None or len(rates) == 0:
        error = mt5.last_error()
        raise ValueError(
            f"No data returned for {symbol}/{timeframe}: {error}. "
            f"Check symbol name and MT5 market watch."
        )

    # Convert structured numpy array → Polars DataFrame
    df = pl.DataFrame({
        "time": pl.Series(rates["time"]).cast(pl.Datetime("ms", time_zone="UTC")),
        "open": pl.Series(rates["real_volume"] if "real_volume" in rates.dtype.names else rates["open"]).cast(pl.Float64),
        "high": pl.Series(rates["high"]).cast(pl.Float64),
        "low": pl.Series(rates["low"]).cast(pl.Float64),
        "close": pl.Series(rates["close"]).cast(pl.Float64),
        "volume": pl.Series(rates["tick_volume"]).cast(pl.Float64),
        "spread": pl.Series(rates["spread"]).cast(pl.Float64),
    })

    # Fix: actually use open column
    df = df.with_columns(
        pl.Series("open", rates["open"]).cast(pl.Float64),
    )

    logger.info(
        "Fetched %d bars for %s/%s (%s → %s)",
        len(df), symbol, timeframe,
        df["time"].min(), df["time"].max(),
    )
    return df


def fetch_ticks(
    symbol: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    n_ticks: int = 100_000,
) -> pl.DataFrame:
    """
    Fetch tick data from MT5.

    Args:
        symbol: MT5 symbol name
        start: Start datetime (UTC)
        end: End datetime (UTC). Defaults to now.
        n_ticks: Max number of ticks to fetch

    Returns:
        Polars DataFrame with columns: [time, bid, ask, volume, flags]
    """
    if not MT5_AVAILABLE:
        raise RuntimeError("MetaTrader5 not available")

    if end is None:
        end = datetime.now(tz=timezone.utc)
    if start is None:
        start = end - timedelta(hours=1)

    ticks = mt5.copy_ticks_range(symbol, start, end, mt5.COPY_TICKS_ALL)

    if ticks is None or len(ticks) == 0:
        error = mt5.last_error()
        raise ValueError(f"No ticks returned for {symbol}: {error}")

    df = pl.DataFrame({
        "time": pl.Series(ticks["time"]).cast(pl.Datetime("ms", time_zone="UTC")),
        "bid": pl.Series(ticks["bid"]).cast(pl.Float64),
        "ask": pl.Series(ticks["ask"]).cast(pl.Float64),
        "volume": pl.Series(ticks["volume"]).cast(pl.Float64),
        "flags": pl.Series(ticks["flags"]).cast(pl.Int32),
    })

    logger.info("Fetched %d ticks for %s", len(df), symbol)
    return df


# ─────────────────────────────────────────────
# Save / Load / Incremental
# ─────────────────────────────────────────────

def save_parquet(df: pl.DataFrame, path: Path | str) -> Path:
    """Save DataFrame to compressed Parquet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(path, compression="zstd")
    size_mb = path.stat().st_size / (1024 * 1024)
    logger.info("Saved %d rows to %s (%.2f MB)", len(df), path, size_mb)
    return path


def load_parquet(path: Path | str) -> pl.DataFrame:
    """Load DataFrame from Parquet."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")
    df = pl.read_parquet(path)
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def fetch_and_save(
    symbol: str,
    timeframe: str = "M1",
    output_dir: Path | str = Path("data/raw"),
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> Path:
    """
    Fetch data from MT5 and save to Parquet. Supports incremental fetch.

    If a parquet file already exists for this symbol/timeframe, only fetches
    data newer than the last timestamp in the file, then appends.

    Returns:
        Path to the saved parquet file.
    """
    output_dir = Path(output_dir)
    filename = f"{symbol}_{timeframe}.parquet"
    filepath = output_dir / filename

    # Incremental: if file exists, only fetch new data
    if filepath.exists():
        existing = load_parquet(filepath)
        last_time = existing["time"].max()
        logger.info("Incremental fetch for %s/%s from %s", symbol, timeframe, last_time)
        # Fetch from last timestamp + 1 minute
        new_start = last_time + timedelta(minutes=1)
        try:
            new_data = fetch_ohlcv(symbol, timeframe, start=new_start, end=end)
            combined = pl.concat([existing, new_data]).unique(subset=["time"]).sort("time")
            return save_parquet(combined, filepath)
        except ValueError:
            logger.info("No new data available for %s/%s", symbol, timeframe)
            return filepath
    else:
        df = fetch_ohlcv(symbol, timeframe, start=start, end=end)
        return save_parquet(df, filepath)


def fetch_multiple_symbols(
    symbols: list[str],
    timeframe: str = "M1",
    output_dir: Path | str = Path("data/raw"),
    mt5_path: Optional[str] = None,
) -> dict[str, Path]:
    """
    Fetch data for multiple symbols. Manages MT5 connection lifecycle.

    Returns:
        Dict mapping symbol → saved parquet path.
    """
    results: dict[str, Path] = {}
    with MT5Connection(path=mt5_path):
        for symbol in symbols:
            try:
                path = fetch_and_save(symbol, timeframe, output_dir)
                results[symbol] = path
            except Exception as e:
                logger.error("Failed to fetch %s: %s", symbol, e)
    return results
