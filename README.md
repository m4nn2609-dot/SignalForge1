# SignalForge
Quant research platform: AI proposes JSON configs, a deterministic engine executes them.
Pipeline (plain Python, no LangGraph): data_architect → quant_analyst → modeler → risk_auditor.

## Run
```
pip install -r requirements.txt
uvicorn backend.main:app --reload --port 8000      # terminal 1
streamlit run frontend/app.py                      # terminal 2
```
Optional local LLM: `ollama pull llama3.2` (falls back to deterministic defaults if unavailable).
Disable LLM: `SIGNALFORGE_USE_LLM=0`. Tests: `python -m pytest`.
Without an uploaded CSV, deterministic synthetic OHLCV is used.
