"""FastAPI app: serves the backtester API and the single-page frontend."""
from __future__ import annotations

import itertools
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from engine import backtest, data, strategies

app = FastAPI(title="Algo Backtester", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


class BacktestRequest(BaseModel):
    symbol: str = Field(..., examples=["AAPL"])
    start: str = Field(..., examples=["2018-01-01"])
    end: str = Field(..., examples=["2024-01-01"])
    strategy: str = Field(..., examples=["ma_crossover"])
    params: Dict[str, float] = Field(default_factory=dict)
    initial_capital: float = 10_000.0
    fee_bps: float = 10.0


class OptimizeRequest(BacktestRequest):
    # Which param to sweep and over what range.
    sweep_param: str
    sweep_values: List[float]


@app.get("/api/strategies")
def list_strategies():
    out = []
    for s in strategies.STRATEGIES.values():
        out.append({
            "key": s.key,
            "name": s.name,
            "description": s.description,
            "params": [
                {
                    "name": p.name, "label": p.label, "default": p.default,
                    "min": p.min, "max": p.max, "step": p.step,
                }
                for p in s.params
            ],
        })
    return {"strategies": out}


def _resolve_params(strategy_key: str, params: Dict[str, float]) -> dict:
    spec = strategies.STRATEGIES[strategy_key]
    resolved = {p.name: p.default for p in spec.params}
    resolved.update({k: v for k, v in params.items() if k in resolved})
    return resolved


@app.post("/api/backtest")
def run(req: BacktestRequest):
    if req.strategy not in strategies.STRATEGIES:
        raise HTTPException(400, f"Unknown strategy '{req.strategy}'.")
    try:
        df = data.load_prices(req.symbol, req.start, req.end)
        params = _resolve_params(req.strategy, req.params)
        signals = strategies.get_signals(req.strategy, df, params)
        result = backtest.run_backtest(
            df, signals,
            initial_capital=req.initial_capital,
            fee_bps=req.fee_bps,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    result["params"] = params
    result["symbol"] = req.symbol.upper()
    result["strategy_name"] = strategies.STRATEGIES[req.strategy].name
    return result


@app.post("/api/optimize")
def optimize(req: OptimizeRequest):
    """Sweep one parameter and return the metric for each value."""
    if req.strategy not in strategies.STRATEGIES:
        raise HTTPException(400, f"Unknown strategy '{req.strategy}'.")
    try:
        df = data.load_prices(req.symbol, req.start, req.end)
    except ValueError as e:
        raise HTTPException(400, str(e))

    results = []
    for val in req.sweep_values:
        params = _resolve_params(req.strategy, req.params)
        params[req.sweep_param] = val
        signals = strategies.get_signals(req.strategy, df, params)
        r = backtest.run_backtest(
            df, signals,
            initial_capital=req.initial_capital,
            fee_bps=req.fee_bps,
        )
        results.append({
            "value": val,
            "total_return_pct": r["metrics"]["total_return_pct"],
            "sharpe": r["metrics"]["sharpe"],
            "max_drawdown_pct": r["metrics"]["max_drawdown_pct"],
        })
    best = max(results, key=lambda x: x["sharpe"]) if results else None
    return {"sweep_param": req.sweep_param, "results": results, "best": best}


# ── Serve frontend ──────────────────────────────────────────────────────────
@app.get("/")
def index():
    return FileResponse(FRONTEND_DIR / "index.html")


if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")
