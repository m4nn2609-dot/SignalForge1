from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from enum import Enum

class OperatorType(str, Enum):
    SMA = "sma"
    EMA = "ema"
    RSI = "rsi"
    ZSCORE = "zscore"
    LOG_RETURN = "log_return"
    ADF_TEST = "adf_test"
    CORRELATION_FILTER = "correlation_filter"

class OperatorConfig(BaseModel):
    """
    Defines a single deterministic mathematical operation.
    The AI proposes this; the Engine executes it.
    """
    operator: OperatorType
    params: Dict[str, Any] = Field(default_factory=dict)
    input_column: str
    output_column: Optional[str] = None

class FeatureSuiteConfig(BaseModel):
    """
    A collection of operators that define the feature engineering pipeline.
    """
    features: List[OperatorConfig]
    target_column: str

class ValidationConfig(BaseModel):
    """
    Configuration for Walk-Forward Validation to prevent overfitting.
    """
    train_window: int  # Days/Periods
    val_window: int    # Days/Periods
    step_size: int     # How much the window slides
    vault_split: float = 0.2  # % of data held for the Final Vault

class ExperimentConfig(BaseModel):
    """
    The complete, immutable configuration for a quantitative experiment.
    This is the 'Source of Truth' logged in the DB.
    """
    experiment_id: str
    symbol: str
    timeframe: str
    feature_suite: FeatureSuiteConfig
    validation: ValidationConfig
    metadata: Dict[str, Any] = Field(default_factory=dict)
