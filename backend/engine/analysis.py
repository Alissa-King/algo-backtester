"""Higher-level analyses built on the backtest engine:

- walk_forward: anchored walk-forward optimization (out-of-sample validation)
- portfolio_backtest: run a strategy across multiple weighted assets
"""
from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd

from . import backtest, data, strategies


def _sweep_values(spec: strategies.ParamSpec, n: int = 20) -> List[float]:
    """Generate ~n candidate values across a parameter's range."""
    raw = np.arange(spec.min, spec.max + spec.step, spec.step)
    if len(raw) <= n:
        vals = raw
    else:
        idx = np.linspace(0, len(raw) - 1, n).round().astype(int)
        vals = raw[idx]
    return [round(float(v), 4) for v in vals]


def walk_forward(
    symbol: str, start: str, end: str, strategy_key: str,
    base_params: Dict[str, float], sweep_param: str,
    n_splits: int = 4, train_ratio: float = 0.5,
    initial_capital: float = 10_000.0, fee_bps: float = 10.0,
) -> dict:
    """Anchored walk-forward: for each out-of-sample fold, re-optimize the
    sweep parameter on all prior (in-sample) data, then trade the fold with the
    best params. Stitches OOS returns into one equity curve. This exposes
    strategies that only look good when fit to the whole history.
    """
    spec = strategies.STRATEGIES[strategy_key]
    df = data.load_prices(symbol, start, end)
    if len(df) < n_splits * 30:
        raise ValueError("Not enough data for the requested number of folds.")

    sweep_spec = next((p for p in spec.params if p.name == sweep_param), spec.params[0])
    candidates = _sweep_values(sweep_spec)

    n = len(df)
    train_end = int(n * train_ratio)
    test_idx = np.array_split(np.arange(train_end, n), n_splits)

    oos_ret = pd.Series(0.0, index=df.index)
    folds = []
    for i, idx in enumerate(test_idx):
        if len(idx) == 0:
            continue
        train_df = df.iloc[:idx[0]]
        test_df = df.iloc[idx[0]: idx[-1] + 1]

        # Optimize on in-sample window.
        best_val, best_sharpe = candidates[0], -np.inf
        for val in candidates:
            params = {**base_params, sweep_param: val}
            sig = strategies.get_signals(strategy_key, train_df, params)
            ret = backtest.strategy_returns(train_df, sig, fee_bps)
            perf = backtest.core_performance(ret, train_df["close"].pct_change(), initial_capital)
            if perf["metrics"]["sharpe"] > best_sharpe:
                best_sharpe, best_val = perf["metrics"]["sharpe"], val

        # Apply to out-of-sample window. Use full df for signal context, slice to fold.
        params = {**base_params, sweep_param: best_val}
        full_sig = strategies.get_signals(strategy_key, df, params)
        fold_sig = full_sig.loc[test_df.index]
        fold_ret = backtest.strategy_returns(test_df, fold_sig, fee_bps)
        oos_ret.loc[test_df.index] = fold_ret

        fold_perf = backtest.core_performance(
            fold_ret, test_df["close"].pct_change(), initial_capital)
        folds.append({
            "fold": i + 1,
            "train_start": train_df.index[0].strftime("%Y-%m-%d"),
            "test_start": test_df.index[0].strftime("%Y-%m-%d"),
            "test_end": test_df.index[-1].strftime("%Y-%m-%d"),
            "best_param": best_val,
            "is_sharpe": round(float(best_sharpe), 2),
            "oos_return_pct": fold_perf["metrics"]["total_return_pct"],
            "oos_sharpe": fold_perf["metrics"]["sharpe"],
        })

    # Stitched OOS curve over the full OOS span.
    oos_span = df.iloc[train_end:]
    oos_ret_span = oos_ret.loc[oos_span.index]
    perf = backtest.core_performance(
        oos_ret_span, oos_span["close"].pct_change(), initial_capital)

    return {
        "symbol": symbol.upper(),
        "strategy_name": spec.name,
        "sweep_param": sweep_param,
        "dates": [d.strftime("%Y-%m-%d") for d in oos_span.index],
        "equity": perf["equity"],
        "benchmark": perf["benchmark"],
        "drawdown": perf["drawdown"],
        "metrics": perf["metrics"],
        "folds": folds,
    }


def portfolio_backtest(
    holdings: List[dict], start: str, end: str, strategy_key: str,
    params: Dict[str, float], initial_capital: float = 10_000.0,
    fee_bps: float = 10.0,
) -> dict:
    """Run a strategy on each holding and combine into a daily-rebalanced,
    weight-blended portfolio. holdings: [{symbol, weight}, ...].
    """
    spec = strategies.STRATEGIES[strategy_key]
    weights = np.array([h["weight"] for h in holdings], dtype=float)
    if weights.sum() <= 0:
        raise ValueError("Portfolio weights must sum to a positive number.")
    weights = weights / weights.sum()  # normalize

    strat_rets, asset_rets, per_asset = {}, {}, []
    for h in holdings:
        df = data.load_prices(h["symbol"], start, end)
        sig = strategies.get_signals(strategy_key, df, params)
        sret = backtest.strategy_returns(df, sig, fee_bps)
        aret = df["close"].pct_change().fillna(0.0)
        strat_rets[h["symbol"].upper()] = sret
        asset_rets[h["symbol"].upper()] = aret

    # Align all series on common trading dates.
    strat_df = pd.DataFrame(strat_rets).dropna(how="all").fillna(0.0)
    asset_df = pd.DataFrame(asset_rets).reindex(strat_df.index).fillna(0.0)

    port_ret = (strat_df * weights).sum(axis=1)
    bench_ret = (asset_df * weights).sum(axis=1)
    perf = backtest.core_performance(port_ret, bench_ret, initial_capital)

    # Per-asset standalone contribution (its own strategy equity).
    for i, h in enumerate(holdings):
        sym = h["symbol"].upper()
        ap = backtest.core_performance(strat_df[sym], asset_df[sym], initial_capital)
        per_asset.append({
            "symbol": sym,
            "weight_pct": round(float(weights[i] * 100), 1),
            "return_pct": ap["metrics"]["total_return_pct"],
            "sharpe": ap["metrics"]["sharpe"],
            "max_drawdown_pct": ap["metrics"]["max_drawdown_pct"],
        })

    return {
        "strategy_name": spec.name,
        "dates": [d.strftime("%Y-%m-%d") for d in strat_df.index],
        "equity": perf["equity"],
        "benchmark": perf["benchmark"],
        "drawdown": perf["drawdown"],
        "metrics": perf["metrics"],
        "holdings": per_asset,
    }
