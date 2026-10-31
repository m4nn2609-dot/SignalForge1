import os
import uuid
from typing import Dict, Any, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File
from pydantic import BaseModel

from backend.core.orchestrator import SignalForgeOrchestrator, sanitize
from backend.core.state import ResearchState
from backend.core.db import DatabaseManager
from backend.core.data import validate_csv, DATA_DIR

app = FastAPI(title="SignalForge API", description="Professional-grade Quant Research Platform")
orchestrator = SignalForgeOrchestrator()
db = DatabaseManager()


class ExperimentRequest(BaseModel):
    symbol: str
    timeframe: str = "1h"
    cost_bps: float = 5.0
    metadata: Optional[Dict[str, Any]] = None


class ExperimentResponse(BaseModel):
    experiment_id: str
    status: str
    message: str


@app.get("/health")
async def health_check():
    return {"status": "online", "engine": "deterministic", "orchestrator": "sequential-python",
            "llm_enabled": orchestrator.llm.enabled}


@app.post("/datasets/upload")
async def upload_dataset(symbol: str, file: UploadFile = File(...)):
    """Upload OHLCV CSV (timestamp,open,high,low,close,volume). Used by experiments for that symbol."""
    try:
        df = validate_csv(pd.read_csv(file.file))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid dataset: {e}")
    os.makedirs(DATA_DIR, exist_ok=True)
    df.to_csv(os.path.join(DATA_DIR, f"{symbol.upper()}.csv"), index=False)
    return {"symbol": symbol.upper(), "rows": len(df), "start": str(df.timestamp.min()), "end": str(df.timestamp.max()),
            "missing_values": int(df.isna().sum().sum())}


@app.post("/research/start", response_model=ExperimentResponse)
async def start_research(request: ExperimentRequest, background_tasks: BackgroundTasks):
    experiment_id = str(uuid.uuid4())
    db.save_experiment({"id": experiment_id, "symbol": request.symbol.upper(),
                        "timeframe": request.timeframe, "status": "RUNNING"})
    initial_state: ResearchState = {
        "symbol": request.symbol.upper(), "timeframe": request.timeframe, "experiment_id": experiment_id,
        "cost_bps": request.cost_bps, "raw_data": None, "errors": [], "iteration_count": 0,
    }
    background_tasks.add_task(run_pipeline_and_save, initial_state)
    return {"experiment_id": experiment_id, "status": "started",
            "message": f"Research pipeline triggered for {request.symbol}"}


@app.get("/research/status/{experiment_id}")
async def get_status(experiment_id: str):
    exp = db.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(status_code=404, detail="Experiment not found")
    return {
        "experiment_id": exp.id, "symbol": exp.symbol, "timeframe": exp.timeframe, "status": exp.status,
        "verdict": exp.audit_verdict, "sharpe": exp.sharpe_ratio, "drawdown": exp.max_drawdown,
        "reasoning": exp.audit_reasoning, "config": exp.feature_config,
        "validation_config": exp.validation_config, "cleaning_config": exp.cleaning_config,
        "metrics": exp.metrics,
    }


@app.get("/research/history")
async def history(limit: int = 50):
    return [{"experiment_id": e.id, "symbol": e.symbol, "timeframe": e.timeframe, "status": e.status,
             "verdict": e.audit_verdict, "sharpe": e.sharpe_ratio, "drawdown": e.max_drawdown,
             "created_at": str(e.created_at)} for e in db.list_experiments(limit)]


def run_pipeline_and_save(state: ResearchState):
    """Plain `def` => FastAPI runs it in a worker thread, so the API stays responsive."""
    try:
        final = orchestrator.run_experiment(state)
        m = final.get("performance_metrics") or {}
        db.save_experiment(sanitize({
            "id": final["experiment_id"], "symbol": final["symbol"], "timeframe": final["timeframe"],
            "feature_config": final.get("feature_config"),
            "validation_config": final.get("validation_config"),
            "cleaning_config": final.get("cleaning_config"),
            "sharpe_ratio": m.get("sharpe"), "max_drawdown": m.get("max_dd"), "pnl": m.get("total_return"),
            "metrics": {**m, "proposal_source": final.get("proposal_source"),
                        "stationarity": final.get("stationarity_report"), "errors": final.get("errors")},
            "audit_verdict": final.get("audit_verdict"), "audit_reasoning": final.get("audit_reasoning"),
            "status": "ERROR" if final.get("audit_verdict") == "ERROR" else "DONE",
        }))
    except Exception as e:
        print(f"Pipeline Error for {state['experiment_id']}: {e}")
        db.save_experiment({"id": state["experiment_id"], "symbol": state["symbol"],
                            "timeframe": state["timeframe"], "status": "ERROR",
                            "audit_verdict": "ERROR", "audit_reasoning": str(e)})
