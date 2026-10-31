import inspect
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
from typing import Dict, Any, List
from backend.schemas.config import OperatorConfig, OperatorType

class QuantEngine:
    """Deterministic execution engine. Takes a JSON config + DataFrame, produces features. No LLM logic."""
    def __init__(self):
        self._operators = {
            OperatorType.SMA: self._compute_sma,
            OperatorType.EMA: self._compute_ema,
            OperatorType.RSI: self._compute_rsi,
            OperatorType.ZSCORE: self._compute_zscore,
            OperatorType.LOG_RETURN: self._compute_log_return,
        }

    @staticmethod
    def output_name(config: OperatorConfig) -> str:
        return config.output_column or f"{config.input_column}_{config.operator.value}"

    def execute_pipeline(self, df: pd.DataFrame, configs: List[OperatorConfig]) -> pd.DataFrame:
        working_df = df.copy()
        for config in configs:
            if config.operator not in self._operators:
                raise NotImplementedError(f"Operator {config.operator} not implemented in Engine.")
            if config.input_column not in working_df.columns:
                raise KeyError(f"Unknown input column '{config.input_column}'")
            op_func = self._operators[config.operator]
            allowed = inspect.signature(op_func).parameters   # ignore unexpected AI-proposed params
            params = {k: int(v) for k, v in config.params.items()
                      if k in allowed and isinstance(v, (int, float)) and v >= 2}
            working_df[self.output_name(config)] = op_func(working_df[config.input_column], **params)
        return working_df

    def _compute_sma(self, series: pd.Series, window: int = 20) -> pd.Series:
        return series.rolling(window=window).mean()

    def _compute_ema(self, series: pd.Series, window: int = 20) -> pd.Series:
        return series.ewm(span=window, adjust=False).mean()

    def _compute_rsi(self, series: pd.Series, window: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.where(delta > 0, 0).rolling(window=window).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
        return 100 - (100 / (1 + gain / loss))

    def _compute_zscore(self, series: pd.Series, window: int = 20) -> pd.Series:
        return (series - series.rolling(window).mean()) / series.rolling(window).std()

    def _compute_log_return(self, series: pd.Series) -> pd.Series:
        return np.log(series / series.shift(1))

    def run_stationarity_test(self, series: pd.Series) -> Dict[str, Any]:
        """ADF test; guards against constant / degenerate series."""
        clean_series = series.replace([np.inf, -np.inf], np.nan).dropna()
        if len(clean_series) < 20 or clean_series.nunique() < 3:
            return {"error": "Insufficient data for ADF test", "is_stationary": False}
        try:
            result = adfuller(clean_series)
        except Exception as e:
            return {"error": str(e), "is_stationary": False}
        return {"adf_statistic": float(result[0]), "p_value": float(result[1]),
                "is_stationary": bool(result[1] < 0.05)}
