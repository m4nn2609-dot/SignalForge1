# SignalForge

**Quantitative Research & Experiment Management Terminal**

SignalForge is a full-stack research platform for testing trading signals honestly. You pick a market, the platform builds features from historical price data, trains a model with chronological validation, backtests it with transaction costs, and returns an **APPROVED / REJECTED** verdict with reasons. Every experiment is saved with its exact configuration so it can be reproduced and compared later.

> **Core idea:** the AI *proposes* settings as JSON; deterministic Python code *executes and judges* them. The AI never computes a metric or decides a verdict.

---

## Features

| Area | What it does |
|---|---|
| Dataset management | Upload OHLCV CSVs; validates required columns, timestamps, numeric values and minimum length |
| Data cleaning | Removes duplicates, forward-fills gaps, drops invalid prices, optional outlier winsorizing |
| Feature engineering | Log returns, rolling z-scores, RSI, SMA, EMA, volume z-score |
| Stationarity checks | Augmented Dickey-Fuller test on every feature; non-stationary features are excluded |
| Chronological validation | Walk-forward folds (train on past, test on future) |
| Final vault | Last 20% of the data is held out and used once, to expose overfitting |
| Backtesting | Long/short positions, configurable transaction costs in basis points |
| Benchmarks | Buy-and-hold and naive majority-class accuracy |
| Risk audit | Rule-based verdict: significance, benchmark, vault, overfitting gap, drawdown |
| Dashboard | Verdict, Sharpe, drawdown, equity curve vs buy-and-hold, train vs unseen Sharpe chart |
| Experiment history | Every run saved to a database with its full configuration |
| Background jobs | Experiments run in a worker thread; the UI polls for progress |
| Optional local LLM | Llama 3.2 via Ollama proposes configs; falls back to defaults if unavailable |

---

## Architecture

```
Streamlit UI (frontend/app.py)
        │  HTTP
        ▼
FastAPI backend (backend/main.py) ──► SQLite via SQLAlchemy
        │  BackgroundTask (worker thread)
        ▼
Orchestrator (backend/core/orchestrator.py)
  data_architect → quant_analyst → modeler → risk_auditor
        │               │             │            │
     data.py      quant_engine.py  backtest.py   audit rules
```

The pipeline is a plain sequential Python runner. The flow is strictly linear, so a graph framework (LangGraph/LangChain) added overhead without benefit and was removed.

### Pipeline steps

1. **Data Architect**: gets a proposal (AI or defaults), loads the data (uploaded CSV or synthetic) and cleans it.
2. **Quant Analyst**: validates the proposed feature JSON with Pydantic, computes features with the deterministic engine, runs the ADF test and keeps only stationary features.
3. **Modeler**: runs walk-forward validation with a logistic regression, then evaluates on the untouched vault. Costs are charged on every position change.
4. **Risk Auditor**: applies rules to the metrics and returns the verdict and reasoning.

---

## Preventing look-ahead bias and data leakage

- The prediction target is the *next* bar's return (`shift(-1)`), so features at time *t* only predict *t → t+1*.
- All rolling features use backward-looking windows only.
- The feature scaler is fitted on each training fold only (scikit-learn `Pipeline`).
- Training data always precedes test data; there is no shuffling.
- A final vault split (default 20%) is never used during walk-forward validation.

---

## Tech stack

- **Backend:** FastAPI, Uvicorn, Pydantic, SQLAlchemy, SQLite
- **Compute:** pandas, NumPy, scikit-learn, statsmodels
- **Frontend:** Streamlit
- **Optional LLM:** Ollama (Llama 3.2)
- **Tests:** pytest

---

## Project structure

```
SignalForge/
├── backend/
│   ├── main.py                 # FastAPI app and endpoints
│   ├── core/
│   │   ├── orchestrator.py     # Sequential pipeline + auditor rules
│   │   ├── data.py             # CSV validation, synthetic data, cleaning
│   │   ├── llm.py              # Optional Ollama wrapper
│   │   ├── db.py               # Database manager
│   │   └── state.py            # Experiment state definition
│   ├── engine/
│   │   ├── quant_engine.py     # Deterministic operators + ADF test
│   │   └── backtest.py         # Walk-forward, vault, costs, metrics
│   ├── models/database.py      # SQLAlchemy tables
│   └── schemas/config.py       # Pydantic config schemas
├── frontend/app.py             # Streamlit dashboard
├── tests/test_engine.py        # Unit and end-to-end tests
├── data/                       # Datasets (<SYMBOL>.csv)
├── fetch.py                    # Optional: download real data from Yahoo Finance
├── requirements.txt
└── pytest.ini
```

---

## Getting started

Requires Python 3.10 to 3.12 (very new Python versions may lack prebuilt wheels for some packages).

### 1. Install

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

macOS/Linux: `source .venv/bin/activate`

### 2. Start the backend (terminal 1)

```powershell
$env:SIGNALFORGE_USE_LLM="0"      # optional: skip the LLM for a fast, deterministic run
uvicorn backend.main:app --port 8000
```

Check it at <http://localhost:8000/health>. Interactive API docs are at <http://localhost:8000/docs>.

### 3. Start the dashboard (terminal 2)

```powershell
streamlit run frontend/app.py
```

Open the URL Streamlit prints (usually <http://localhost:8501>), enter a symbol and click **Launch Experiment**.

---

## Data

The only input is **OHLCV price data**:

```
timestamp,open,high,low,close,volume
2024-01-01,100.0,100.5,99.9,100.0,9158
```

- `Date`, `Datetime` and `Time` are accepted in place of `timestamp`.
- At least 300 rows are required; 1,000 or more is recommended.
- If no file exists for the chosen symbol, the platform generates **deterministic synthetic data** (the same symbol always yields the same series). Synthetic data is for demonstration only and is not real market history.

### Using your own or real data

**Upload in the UI:** use the sidebar uploader. The file is saved as `data/<SYMBOL>.csv` for the symbol in the text box.

**Download real data from Yahoo Finance** (needs internet):

```powershell
pip install yfinance
python fetch.py                              # BTC-USD, ETH-USD, AAPL, daily 2020-2024
python fetch.py TSLA RELIANCE.NS             # any Yahoo tickers
python fetch.py BTC-USD --start 2019-01-01 --end 2024-12-31
```

**Generate synthetic data offline:**

```powershell
foreach ($s in "BTCUSDT","ETHUSDT","AAPL") { python -c "import sys; from backend.core.data import synthetic_ohlcv as g; s=sys.argv[1]; g(s,n=1500,timeframe='1d').to_csv('data/'+s+'.csv',index=False)" $s }
```

Match the **Timeframe** in the UI to your data (use `1d` for daily files).

---

## API reference

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Server status and whether the LLM is enabled |
| POST | `/datasets/upload?symbol=X` | Upload and validate an OHLCV CSV |
| POST | `/research/start` | Start an experiment (`symbol`, `timeframe`, `cost_bps`) |
| GET | `/research/status/{id}` | Status, metrics and saved configuration |
| GET | `/research/history` | Recent experiments |

Example:

```powershell
curl -X POST http://localhost:8000/research/start -H "Content-Type: application/json" -d "{\"symbol\":\"BTCUSDT\",\"timeframe\":\"1d\",\"cost_bps\":5}"
```

---

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `SIGNALFORGE_USE_LLM` | `1` | Set to `0` to disable the LLM and use deterministic defaults |
| `SIGNALFORGE_LLM_TIMEOUT` | `40` | Seconds to wait for Ollama |
| `SIGNALFORGE_DATA_DIR` | `data` | Folder for datasets |

To use the LLM: install [Ollama](https://ollama.com), run `ollama pull llama3.2`, and leave `SIGNALFORGE_USE_LLM` unset. If Ollama is unreachable or returns invalid JSON, the pipeline falls back to defaults automatically.

---

## How the verdict is decided

An experiment is **REJECTED** if any of the following holds:

- the out-of-sample t-statistic is below 2 (not statistically significant)
- the strategy's Sharpe ratio does not beat buy-and-hold
- the Sharpe ratio on the unseen vault is zero or negative
- in-sample Sharpe exceeds out-of-sample Sharpe by more than 1.5 (overfitting)
- maximum drawdown is worse than 35%

Otherwise it is **APPROVED**. The reasoning text lists every rule that failed, along with the number of folds, the cost assumption and model accuracy against the naive baseline.

---

## Tests

```powershell
python -m pytest
```

The tests cover the SMA operator, the ADF stationarity test, and a full end-to-end pipeline run (with the LLM disabled).

---
## Roadmap

- Additional models (random forest, gradient boosting) and model comparison
- Cross-asset generalization and market-regime comparison
- Portfolio construction with allocation constraints
- Downloadable research reports and a robustness leaderboard
- Durable job queue (Celery/Redis) and PostgreSQL
- Live progress streaming (SSE/WebSockets)
- Dataset hashing for stronger reproducibility
