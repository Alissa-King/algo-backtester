# 📈 Algo Backtester

A full-stack strategy backtesting platform. Test classic trading strategies
against real market history using a **custom vectorized backtest engine** — no
black-box backtesting library, so the logic is transparent and auditable.

![stack](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![plotly](https://img.shields.io/badge/Plotly.js-3F4F75?logo=plotly&logoColor=white)

## Features

- **Three strategies** — Moving Average Crossover, RSI Mean Reversion, Bollinger Band Breakout
- **Custom engine** — vectorized in pandas/numpy with a realistic execution model:
  - No look-ahead bias (signals fill on the *next* bar)
  - Transaction costs charged on turnover (configurable bps)
  - Long/flat positioning
- **Real metrics** — Total Return, CAGR, Sharpe, Sortino, Max Drawdown, Volatility,
  Win Rate, Exposure, and a full round-trip trade log
- **Benchmark comparison** — every run is measured against buy-and-hold
- **Parameter optimization** — sweep a parameter and find the Sharpe-maximizing value
- **Polished UI** — responsive single-page dashboard with interactive Plotly charts

## Architecture

```
algo-backtester/
├── backend/
│   ├── main.py              # FastAPI app + API routes, serves the frontend
│   ├── engine/
│   │   ├── data.py          # yfinance loader with in-process cache
│   │   ├── strategies.py    # signal generators + parameter specs
│   │   └── backtest.py      # vectorized engine + performance metrics
│   └── requirements.txt
└── frontend/
    └── index.html           # SPA (Tailwind + Plotly via CDN, no build step)
```

## Running locally

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate          # Windows  (use: source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then open **http://localhost:8000**.

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/strategies` | GET | List strategies and their tunable parameters |
| `/api/backtest`   | POST | Run a single backtest, returns metrics + curves + trades |
| `/api/optimize`   | POST | Sweep one parameter, returns metric per value + best |

## Notes

Data is sourced from Yahoo Finance via `yfinance` (no API key required). Supports
equities (`AAPL`), crypto (`BTC-USD`), ETFs, and most Yahoo-listed symbols.

*Educational project. Past performance does not guarantee future results.*
