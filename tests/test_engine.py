import pandas as pd
import numpy as np
import pytest
from backend.engine.quant_engine import QuantEngine
from backend.schemas.config import OperatorConfig, OperatorType

def test_quant_engine_sma():
    engine = QuantEngine()
    # Create a simple series: 1, 2, 3, 4, 5
    df = pd.DataFrame({'close': [1.0, 2.0, 3.0, 4.0, 5.0]})
    
    config = OperatorConfig(
        operator=OperatorType.SMA,
        params={'window': 2},
        input_column='close',
        output_column='sma_2'
    )
    
    result = engine.execute_pipeline(df, [config])
    # SMA of [1, 2] is 1.5; [2, 3] is 2.5... [4, 5] is 4.5
    expected = [np.nan, 1.5, 2.5, 3.5, 4.5]
    np.testing.assert_array_almost_equal(result['sma_2'].values, expected)

def test_quant_engine_adf():
    engine = QuantEngine()
    # Random walk (non-stationary)
    np.random.seed(42)
    df = pd.DataFrame({'close': np.cumsum(np.random.randn(100))})
    
    res = engine.run_stationarity_test(df['close'])
    assert "is_stationary" in res
    # Random walk should usually be non-stationary (is_stationary=False)
    assert res["is_stationary"] == False

def test_quant_engine_stationary():
    engine = QuantEngine()
    # Stationary series (noise)
    np.random.seed(42)
    df = pd.DataFrame({'close': np.random.randn(100)})
    
    res = engine.run_stationarity_test(df['close'])
    assert res["is_stationary"] == True


def test_full_pipeline_runs_without_llm(monkeypatch):
    monkeypatch.setenv("SIGNALFORGE_USE_LLM", "0")
    from backend.core.orchestrator import SignalForgeOrchestrator
    final = SignalForgeOrchestrator().run_experiment(
        {"symbol": "TEST", "timeframe": "1h", "experiment_id": "t", "cost_bps": 5.0, "errors": []})
    assert final["audit_verdict"] in ("APPROVED", "REJECTED"), final.get("errors")
    m = final["performance_metrics"]
    assert m["n_folds"] > 0 and "vault_sharpe" in m
