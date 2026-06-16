"""Vectorized backtest engine and performance metrics.

Execution model
---------------
- Positions are decided on the close of bar t (the signal), and we assume the
  trade fills at the close of bar t+1. We implement this by shifting the target
  position forward one bar before applying returns. This avoids look-ahead bias.
- Transaction costs are charged on the *turnover* (change in position) each bar.
- The strategy is long/flat only (positions in [0, 1]).
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def run_backtest(
    df: pd.DataFrame,
    signals: pd.Series,
    initial_capital: float = 10_000.0,
    fee_bps: float = 10.0,
) -> dict:
    """Run the backtest and return equity curves, trades, and metrics.

    fee_bps: round-trip cost expressed in basis points of traded notional,
    charged proportional to position change each bar.
    """
    close = df["close"]
    asset_ret = close.pct_change().fillna(0.0)

    # Shift to avoid look-ahead: act on the bar *after* the signal.
    position = signals.shift(1).fillna(0.0).clip(0.0, 1.0)

    # Costs proportional to turnover.
    turnover = position.diff().abs().fillna(position.abs())
    fee_rate = fee_bps / 10_000.0
    cost = turnover * fee_rate

    strat_ret = position * asset_ret - cost

    equity = (1.0 + strat_ret).cumprod() * initial_capital
    benchmark = (1.0 + asset_ret).cumprod() * initial_capital

    metrics = _compute_metrics(strat_ret, equity, asset_ret, benchmark, position)
    trades = _extract_trades(position, close, initial_capital, strat_ret)

    # Win rate from completed round trips.
    closed = [t for t in trades if not t.get("open")]
    wins = sum(1 for t in closed if t["return_pct"] > 0)
    metrics["win_rate_pct"] = round(100.0 * wins / len(closed), 2) if closed else 0.0
    metrics["avg_trade_pct"] = (
        round(float(np.mean([t["return_pct"] for t in closed])), 2) if closed else 0.0
    )

    return {
        "dates": [d.strftime("%Y-%m-%d") for d in equity.index],
        "equity": [round(float(v), 2) for v in equity.values],
        "benchmark": [round(float(v), 2) for v in benchmark.values],
        "drawdown": _drawdown_series(equity),
        "price": [round(float(v), 4) for v in close.values],
        "position": [float(v) for v in position.values],
        "metrics": metrics,
        "trades": trades,
    }


def core_performance(strat_ret: pd.Series, asset_ret: pd.Series,
                     initial_capital: float = 10_000.0) -> dict:
    """Compute headline metrics + equity/benchmark/drawdown from return series.

    Shared by walk-forward and portfolio analysis where there is no single
    position series to reason about (so no exposure / trade stats).
    """
    strat_ret = strat_ret.fillna(0.0)
    asset_ret = asset_ret.fillna(0.0)
    equity = (1.0 + strat_ret).cumprod() * initial_capital
    benchmark = (1.0 + asset_ret).cumprod() * initial_capital

    n = len(strat_ret)
    years = max(n / TRADING_DAYS, 1e-9)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0 if n else 0.0
    bench_return = benchmark.iloc[-1] / benchmark.iloc[0] - 1.0 if n else 0.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0 if n else 0.0
    vol = strat_ret.std() * np.sqrt(TRADING_DAYS)
    mean_ret = strat_ret.mean() * TRADING_DAYS
    sharpe = mean_ret / vol if vol > 0 else 0.0
    downside = strat_ret[strat_ret < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = mean_ret / downside if downside > 0 else 0.0
    max_dd = ((equity / equity.cummax() - 1.0).min()) * 100.0 if n else 0.0

    return {
        "metrics": {
            "total_return_pct": round(float(total_return * 100), 2),
            "benchmark_return_pct": round(float(bench_return * 100), 2),
            "cagr_pct": round(float(cagr * 100), 2),
            "sharpe": round(float(sharpe), 2),
            "sortino": round(float(sortino), 2),
            "max_drawdown_pct": round(float(max_dd), 2),
            "volatility_pct": round(float(vol * 100), 2),
        },
        "equity": [round(float(v), 2) for v in equity.values],
        "benchmark": [round(float(v), 2) for v in benchmark.values],
        "drawdown": _drawdown_series(equity),
    }


def strategy_returns(df: pd.DataFrame, signals: pd.Series, fee_bps: float = 10.0) -> pd.Series:
    """Return the cost-adjusted strategy return series (no look-ahead)."""
    asset_ret = df["close"].pct_change().fillna(0.0)
    position = signals.shift(1).fillna(0.0).clip(0.0, 1.0)
    turnover = position.diff().abs().fillna(position.abs())
    cost = turnover * (fee_bps / 10_000.0)
    return position * asset_ret - cost


def _drawdown_series(equity: pd.Series) -> List[float]:
    running_max = equity.cummax()
    dd = (equity / running_max - 1.0) * 100.0
    return [round(float(v), 2) for v in dd.values]


def _compute_metrics(
    strat_ret: pd.Series,
    equity: pd.Series,
    asset_ret: pd.Series,
    benchmark: pd.Series,
    position: pd.Series,
) -> Dict[str, float]:
    n = len(strat_ret)
    years = max(n / TRADING_DAYS, 1e-9)

    total_return = equity.iloc[-1] / equity.iloc[0] - 1.0
    bench_return = benchmark.iloc[-1] / benchmark.iloc[0] - 1.0
    cagr = (equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0

    vol = strat_ret.std() * np.sqrt(TRADING_DAYS)
    mean_ret = strat_ret.mean() * TRADING_DAYS
    sharpe = mean_ret / vol if vol > 0 else 0.0

    downside = strat_ret[strat_ret < 0].std() * np.sqrt(TRADING_DAYS)
    sortino = mean_ret / downside if downside > 0 else 0.0

    running_max = equity.cummax()
    max_dd = ((equity / running_max - 1.0).min()) * 100.0

    # Exposure: fraction of bars in the market.
    exposure = (position > 0).mean() * 100.0

    # Per-trade win rate based on completed round trips.
    entries = (position > 0) & (position.shift(1).fillna(0.0) == 0)
    num_trades = int(entries.sum())

    return {
        "total_return_pct": round(float(total_return * 100), 2),
        "benchmark_return_pct": round(float(bench_return * 100), 2),
        "cagr_pct": round(float(cagr * 100), 2),
        "sharpe": round(float(sharpe), 2),
        "sortino": round(float(sortino), 2),
        "max_drawdown_pct": round(float(max_dd), 2),
        "volatility_pct": round(float(vol * 100), 2),
        "exposure_pct": round(float(exposure), 2),
        "num_trades": num_trades,
    }


def _extract_trades(
    position: pd.Series,
    close: pd.Series,
    initial_capital: float,
    strat_ret: pd.Series,
) -> List[dict]:
    """Reconstruct round-trip trades (long entry -> exit) with P&L."""
    trades: List[dict] = []
    in_trade = False
    entry_date = None
    entry_price = None

    prev = 0.0
    for date, pos in position.items():
        if pos > 0 and prev == 0:
            in_trade = True
            entry_date = date
            entry_price = close.loc[date]
        elif pos == 0 and prev > 0 and in_trade:
            exit_price = close.loc[date]
            ret = exit_price / entry_price - 1.0
            trades.append({
                "entry_date": entry_date.strftime("%Y-%m-%d"),
                "exit_date": date.strftime("%Y-%m-%d"),
                "entry_price": round(float(entry_price), 4),
                "exit_price": round(float(exit_price), 4),
                "return_pct": round(float(ret * 100), 2),
                "bars": int((date - entry_date).days),
            })
            in_trade = False
        prev = pos

    # Close any open trade at the last bar.
    if in_trade and entry_date is not None:
        last_date = position.index[-1]
        exit_price = close.iloc[-1]
        ret = exit_price / entry_price - 1.0
        trades.append({
            "entry_date": entry_date.strftime("%Y-%m-%d"),
            "exit_date": last_date.strftime("%Y-%m-%d"),
            "entry_price": round(float(entry_price), 4),
            "exit_price": round(float(exit_price), 4),
            "return_pct": round(float(ret * 100), 2),
            "bars": int((last_date - entry_date).days),
            "open": True,
        })

    return trades
