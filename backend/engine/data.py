"""Market data loading via yfinance, with a tiny in-process cache."""
from __future__ import annotations

import time
from typing import Dict, Tuple

import pandas as pd
import yfinance as yf

# Cache: key -> (timestamp, DataFrame). Avoids hammering yfinance during
# parameter sweeps where the same series is fetched repeatedly.
_CACHE: Dict[Tuple[str, str, str], Tuple[float, pd.DataFrame]] = {}
_CACHE_TTL = 60 * 30  # 30 minutes


def load_prices(symbol: str, start: str, end: str) -> pd.DataFrame:
    """Return a DataFrame indexed by date with a single 'close' column.

    Raises ValueError if the symbol returns no data.
    """
    symbol = symbol.strip().upper()
    key = (symbol, start, end)
    now = time.time()

    cached = _CACHE.get(key)
    if cached and now - cached[0] < _CACHE_TTL:
        return cached[1].copy()

    raw = yf.download(
        symbol,
        start=start,
        end=end,
        progress=False,
        auto_adjust=True,
    )

    if raw is None or raw.empty:
        raise ValueError(
            f"No price data for '{symbol}' between {start} and {end}. "
            "Check the ticker (e.g. AAPL, BTC-USD) and date range."
        )

    # yfinance returns a MultiIndex column frame for single tickers in recent
    # versions; flatten to grab the close series robustly.
    if isinstance(raw.columns, pd.MultiIndex):
        close = raw["Close"][symbol] if symbol in raw["Close"].columns else raw["Close"].iloc[:, 0]
    else:
        close = raw["Close"]

    df = pd.DataFrame({"close": close.astype(float)})
    df = df.dropna()
    df.index = pd.to_datetime(df.index)

    _CACHE[key] = (now, df)
    return df.copy()
