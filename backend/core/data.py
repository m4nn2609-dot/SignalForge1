import os
import zlib
import numpy as np
import pandas as pd
from typing import Dict, Any

DATA_DIR = os.getenv("SIGNALFORGE_DATA_DIR", "data")
REQUIRED = ["timestamp", "open", "high", "low", "close", "volume"]

def validate_csv(df: pd.DataFrame) -> pd.DataFrame:
    """Validate required columns / timestamps / values. Raises ValueError with a clear message."""
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for c in REQUIRED[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["timestamp", "close"])
    if len(df) < 300:
        raise ValueError("Need at least 300 valid rows")
    return df.sort_values("timestamp").reset_index(drop=True)

def synthetic_ohlcv(symbol: str, n: int = 3000, timeframe: str = "1h") -> pd.DataFrame:
    """Deterministic (per symbol) synthetic OHLCV with regimes + weak momentum, for demos."""
    rng = np.random.default_rng(zlib.crc32(symbol.encode()))
    vol = np.where(np.sin(np.arange(n) / 300.0) > 0, 0.004, 0.009)
    eps = rng.standard_normal(n) * vol
    ret = np.zeros(n)
    for t in range(1, n):
        ret[t] = 0.08 * ret[t - 1] + eps[t]
    close = 100 * np.exp(np.cumsum(ret))
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) * (1 + np.abs(rng.standard_normal(n)) * vol / 3)
    low = np.minimum(open_, close) * (1 - np.abs(rng.standard_normal(n)) * vol / 3)
    freq = {"1m": "1min", "5m": "5min", "15m": "15min", "1h": "1h", "1d": "1D"}.get(timeframe, "1h")
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq=freq),
        "open": open_, "high": high, "low": low, "close": close,
        "volume": rng.lognormal(10, 0.5, n),
    })

def load_ohlcv(symbol: str, timeframe: str) -> pd.DataFrame:
    """Uploaded CSV (data/<SYMBOL>.csv) if present, otherwise synthetic data."""
    path = os.path.join(DATA_DIR, f"{symbol.upper()}.csv")
    if os.path.exists(path):
        return validate_csv(pd.read_csv(path))
    return synthetic_ohlcv(symbol, timeframe=timeframe)

def clean_data(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    """Deterministic cleaning. Only whitelisted options from the (possibly AI-proposed) config are used."""
    df = df.copy()
    if cfg.get("drop_duplicates", True):
        df = df.drop_duplicates(subset="timestamp")
    df = df[(df["close"] > 0)]
    if cfg.get("fill_method", "ffill") == "ffill":
        df = df.ffill()
    df = df.dropna().reset_index(drop=True)
    try:
        w = float(cfg.get("winsorize_pct", 0.0) or 0.0)
    except (TypeError, ValueError):
        w = 0.0
    if 0 < w < 0.2:
        r = np.log(df["close"]).diff()
        r = r.clip(r.quantile(w), r.quantile(1 - w)).fillna(0)
        df["close"] = df["close"].iloc[0] * np.exp(r.cumsum())
    return df
