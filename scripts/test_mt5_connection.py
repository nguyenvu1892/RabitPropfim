"""
MT5 Connection Test — Verifies FTMO Demo account + all target symbols.

Usage: py -3.11 scripts/test_mt5_connection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))


def load_env() -> dict[str, str]:
    env_path = project_root / ".env"
    if not env_path.exists():
        print("[X] .env file not found! Copy .env.example to .env and fill in credentials.")
        sys.exit(1)
    env = {}
    with open(env_path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                env[key.strip()] = value.strip()
    return env


def main() -> None:
    print("=" * 60)
    print("  RABIT-PROPFIRM -- MT5 Connection Test (Multi-Symbol)")
    print("=" * 60)

    # Check MT5 package
    print("\n[1/5] Checking MetaTrader5 package...")
    try:
        import MetaTrader5 as mt5
        print(f"  [OK] MetaTrader5 version: {mt5.__version__}")
    except ImportError:
        print("  [X] MetaTrader5 not installed! Run: pip install MetaTrader5")
        sys.exit(1)

    # Load credentials
    print("\n[2/5] Loading credentials from .env...")
    env = load_env()
    login = int(env.get("MT5_LOGIN", "0"))
    password = env.get("MT5_PASSWORD", "")
    server = env.get("MT5_SERVER", "")
    symbols_str = env.get("MT5_SYMBOLS", "XAUUSD,US100.cash,US30.cash,ETHUSD,BTCUSD")
    symbols = [s.strip() for s in symbols_str.split(",")]
    print(f"  Login: {login}")
    print(f"  Server: {server}")
    print(f"  Target symbols: {symbols}")

    # Initialize MT5
    print("\n[3/5] Connecting to MT5 terminal...")
    if not mt5.initialize():
        print(f"  [X] MT5 initialize failed: {mt5.last_error()}")
        sys.exit(1)
    print(f"  [OK] Terminal: {mt5.terminal_info().name}")

    # Login
    print("\n[4/5] Logging into FTMO account...")
    if not mt5.login(login, password=password, server=server):
        print(f"  [X] Login failed: {mt5.last_error()}")
        mt5.shutdown()
        sys.exit(1)

    account = mt5.account_info()
    print(f"  [OK] Logged in!")
    print(f"  Account: {account.login}")
    print(f"  Name: {account.name}")
    print(f"  Balance: ${account.balance:,.2f}")
    print(f"  Equity: ${account.equity:,.2f}")
    print(f"  Leverage: 1:{account.leverage}")
    print(f"  Mode: {'Demo' if account.trade_mode == 0 else 'Real'}")

    # Check all target symbols
    print(f"\n[5/5] Checking {len(symbols)} target symbols...")
    print("-" * 60)
    print(f"  {'Symbol':<15} {'Bid':>12} {'Ask':>12} {'Spread':>8} {'MinLot':>8} {'Status':>8}")
    print("-" * 60)

    all_ok = True
    for sym in symbols:
        info = mt5.symbol_info(sym)
        if info is None:
            print(f"  {sym:<15} {'':>12} {'':>12} {'':>8} {'':>8} {'[X]':>8}")
            all_ok = False
            continue

        mt5.symbol_select(sym, True)
        tick = mt5.symbol_info_tick(sym)

        bid = tick.bid if tick else 0.0
        ask = tick.ask if tick else 0.0
        spread = info.spread
        min_lot = info.volume_min

        status = "[OK]" if info else "[X]"
        print(f"  {sym:<15} {bid:>12.5f} {ask:>12.5f} {spread:>8} {min_lot:>8.2f} {status:>8}")

    print("-" * 60)

    if all_ok:
        print("\n  ALL SYMBOLS AVAILABLE!")
    else:
        print("\n  [!] Some symbols not found. Check FTMO symbol names.")
        # Show available similar symbols
        all_symbols = mt5.symbols_get()
        if all_symbols:
            names = [s.name for s in all_symbols]
            for sym in symbols:
                if mt5.symbol_info(sym) is None:
                    similar = [n for n in names if sym.split(".")[0].lower() in n.lower()][:5]
                    if similar:
                        print(f"  Suggestions for {sym}: {similar}")

    print("\n" + "=" * 60)
    mt5.shutdown()


if __name__ == "__main__":
    main()
