"""Trading strategies.

Each strategy takes a price DataFrame (with a 'close' column) plus parameters
and returns a pandas Series of target positions: 1.0 = fully long, 0.0 = flat.
The backtest engine handles execution, costs, and metrics — strategies only
decide *when* to be in the market.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import pandas as pd


@dataclass
class ParamSpec:
    name: str
    label: str
    default: float
    min: float
    max: float
    step: float


@dataclass
class StrategyDef:
    key: str
    name: str
    description: str
    params: List[ParamSpec]
    fn: Callable[[pd.DataFrame, dict], pd.Series]


def _ma_crossover(df: pd.DataFrame, p: dict) -> pd.Series:
    fast = int(p["fast"])
    slow = int(p["slow"])
    close = df["close"]
    fast_ma = close.rolling(fast).mean()
    slow_ma = close.rolling(slow).mean()
    # Long when fast MA above slow MA.
    pos = (fast_ma > slow_ma).astype(float)
    pos[fast_ma.isna() | slow_ma.isna()] = 0.0
    return pos


def _rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    return 100 - (100 / (1 + rs))


def _rsi_reversion(df: pd.DataFrame, p: dict) -> pd.Series:
    period = int(p["period"])
    lower = float(p["lower"])
    upper = float(p["upper"])
    rsi = _rsi(df["close"], period)
    # Enter long when oversold, exit when overbought; hold in between.
    pos = pd.Series(np.nan, index=df.index)
    pos[rsi < lower] = 1.0
    pos[rsi > upper] = 0.0
    pos = pos.ffill().fillna(0.0)
    return pos


def _bollinger_breakout(df: pd.DataFrame, p: dict) -> pd.Series:
    window = int(p["window"])
    mult = float(p["mult"])
    close = df["close"]
    mid = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = mid + mult * std
    # Breakout: go long when price closes above upper band, exit below mid.
    pos = pd.Series(np.nan, index=df.index)
    pos[close > upper] = 1.0
    pos[close < mid] = 0.0
    pos = pos.ffill().fillna(0.0)
    pos[mid.isna()] = 0.0
    return pos


STRATEGIES: Dict[str, StrategyDef] = {
    "ma_crossover": StrategyDef(
        key="ma_crossover",
        name="Moving Average Crossover",
        description="Go long when the fast moving average crosses above the slow one.",
        params=[
            ParamSpec("fast", "Fast MA window", 20, 5, 100, 1),
            ParamSpec("slow", "Slow MA window", 50, 10, 250, 1),
        ],
        fn=_ma_crossover,
    ),
    "rsi_reversion": StrategyDef(
        key="rsi_reversion",
        name="RSI Mean Reversion",
        description="Buy when RSI is oversold, sell when overbought.",
        params=[
            ParamSpec("period", "RSI period", 14, 2, 50, 1),
            ParamSpec("lower", "Oversold level", 30, 5, 45, 1),
            ParamSpec("upper", "Overbought level", 70, 55, 95, 1),
        ],
        fn=_rsi_reversion,
    ),
    "bollinger_breakout": StrategyDef(
        key="bollinger_breakout",
        name="Bollinger Band Breakout",
        description="Go long on a breakout above the upper band, exit at the mean.",
        params=[
            ParamSpec("window", "Band window", 20, 5, 100, 1),
            ParamSpec("mult", "Std-dev multiplier", 2.0, 0.5, 4.0, 0.1),
        ],
        fn=_bollinger_breakout,
    ),
}


def get_signals(strategy_key: str, df: pd.DataFrame, params: dict) -> pd.Series:
    if strategy_key not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy_key}'.")
    return STRATEGIES[strategy_key].fn(df, params)
