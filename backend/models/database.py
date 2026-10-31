from sqlalchemy import Column, Integer, String, JSON, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.sql import func
from datetime import datetime

Base = declarative_base()

class Experiment(Base):
    """
    The Immutable Experiment Record.
    Once created, an experiment should only be updated with results,
    never with a change in configuration.
    """
    __tablename__ = "experiments"

    id = Column(String, primary_key=True) # UUID
    symbol = Column(String, nullable=False)
    timeframe = Column(String, nullable=False)

    # The "Contract": Save the exact JSON proposed by the AI
    feature_config = Column(JSON, nullable=True)
    validation_config = Column(JSON, nullable=True)
    cleaning_config = Column(JSON, nullable=True)

    # Results
    sharpe_ratio = Column(Float, nullable=True)
    max_drawdown = Column(Float, nullable=True)
    pnl = Column(Float, nullable=True)
    metrics = Column(JSON, nullable=True)  # full result set incl. equity curve
    status = Column(String, nullable=True)  # RUNNING / DONE / ERROR

    # Rigor
    audit_verdict = Column(String, nullable=True) # APPROVED / REJECTED
    audit_reasoning = Column(String, nullable=True)

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class DatasetVersion(Base):
    """
    Tracks which version of the data was used to ensure reproducibility.
    """
    __tablename__ = "dataset_versions"

    id = Column(Integer, primary_key=True)
    symbol = Column(String, nullable=False)
    version_hash = Column(String, nullable=False) # SHA-256 of the CSV file
    created_at = Column(DateTime(timezone=True), server_default=func.now())
