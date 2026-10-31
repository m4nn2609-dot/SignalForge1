from typing import TypedDict, Dict, Any, List, Optional
import pandas as pd

class ResearchState(TypedDict, total=False):
    """Global state of one experiment, passed step to step by the orchestrator."""
    symbol: str
    timeframe: str
    experiment_id: str
    cost_bps: float

    raw_data: Optional[pd.DataFrame]
    cleaned_data: Optional[pd.DataFrame]
    processed_features: Optional[pd.DataFrame]

    proposal: Optional[Dict[str, Any]]
    proposal_source: Optional[str]
    cleaning_config: Optional[Dict[str, Any]]
    feature_config: Optional[Dict[str, Any]]
    validation_config: Optional[Dict[str, Any]]

    stationarity_report: Optional[Dict[str, Any]]
    performance_metrics: Optional[Dict[str, Any]]
    audit_verdict: Optional[str]
    audit_reasoning: Optional[str]

    errors: List[str]
    iteration_count: int
