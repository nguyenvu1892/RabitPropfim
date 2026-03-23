"""
Generate _ohlcv.npy files from existing _50dim.npy data.

Since MT5 is not available, this script reconstructs synthetic OHLCV
from log_return (col 27) in the 50-dim features. The env will detect
these files and use them instead of the fallback (identical result,
but no warning message).

For XAUUSD: base price ~2600 (current gold price range)
For BTCUSD: base price ~65000
For ETHUSD: base price ~3500  
For US30_cash: base price ~39000
For US100_cash: base price ~17500

Usage: python scripts/generate_ohlcv.py
       Then push _ohlcv.npy files to server.
"""
import numpy as np
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Symbol base prices (approximate real-world reference)
SYMBOL_PRICES = {
    "XAUUSD": 2600.0,
    "BTCUSD": 65000.0,
    "ETHUSD": 3500.0,
    "US30_cash": 39000.0,
    "US100_cash": 17500.0,
}

LOG_RETURN_COL = 27  # log_return column in 50-dim features

def generate_ohlcv(safe_name: str, tf_name: str, base_price: float):
    """Generate OHLCV from log_return in 50-dim features."""
    npy_path = DATA_DIR / f"{safe_name}_{tf_name}_50dim.npy"
    if not npy_path.exists():
        print(f"  SKIP {npy_path.name} (not found)")
        return False
    
    features = np.load(npy_path)
    n_bars = len(features)
    
    # Extract log_return and reconstruct prices
    log_returns = features[:, LOG_RETURN_COL].astype(np.float64)
    log_returns = np.nan_to_num(log_returns, nan=0.0)
    
    # Cumulative sum with clamp to prevent overflow
    cum_log_ret = np.cumsum(log_returns)
    cum_log_ret = np.clip(cum_log_ret, -20.0, 20.0)
    
    # Reconstruct close prices from base price
    close = base_price * np.exp(cum_log_ret)
    close = np.nan_to_num(close, nan=base_price, posinf=base_price*10, neginf=base_price*0.1)
    
    # Generate realistic OHLCV from close
    # Typical intraday range: ~0.05-0.2% per bar
    volatility = np.abs(log_returns) + 0.0005  # minimum volatility
    high = close * (1.0 + volatility * 0.6)
    low = close * (1.0 - volatility * 0.6)
    opn = np.roll(close, 1)  # open ≈ previous close
    opn[0] = close[0]
    volume = np.ones(n_bars, dtype=np.float64) * 1000  # dummy volume
    
    # Stack OHLCV: (N, 5)
    ohlcv = np.column_stack([opn, high, low, close, volume]).astype(np.float32)
    
    # Save
    ohlcv_path = DATA_DIR / f"{safe_name}_{tf_name}_ohlcv.npy"
    np.save(ohlcv_path, ohlcv)
    print(f"  {safe_name}_{tf_name}_ohlcv.npy: ({n_bars:,} x 5) | "
          f"price range: {close.min():.2f} - {close.max():.2f}")
    return True

def main():
    print("=" * 70)
    print("  Generating OHLCV files from 50-dim features")
    print("=" * 70)
    
    total = 0
    for sym, base_price in SYMBOL_PRICES.items():
        print(f"\n{sym} (base={base_price:.0f}):")
        for tf in ["M1", "M5", "M15", "H1"]:
            if generate_ohlcv(sym, tf, base_price):
                total += 1
    
    print(f"\n{'=' * 70}")
    print(f"  Done! Generated {total} OHLCV files in {DATA_DIR}")
    
    # List generated files
    ohlcv_files = sorted(DATA_DIR.glob("*_ohlcv.npy"))
    total_mb = sum(f.stat().st_size for f in ohlcv_files) / (1024*1024)
    print(f"  Total: {len(ohlcv_files)} files, {total_mb:.1f} MB")
    print(f"{'=' * 70}")

if __name__ == "__main__":
    main()
