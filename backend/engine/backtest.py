import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

PERIODS_PER_YEAR = {"1m": 525600, "5m": 105120, "15m": 35040, "1h": 8760, "1d": 252}

def _sharpe(r, ppy: int) -> float:
    r = np.asarray(r, float)
    s = r.std(ddof=1) if len(r) > 1 else 0.0
    return float(r.mean() / s * np.sqrt(ppy)) if s > 1e-12 else 0.0

def _max_dd(r) -> float:
    eq = np.exp(np.cumsum(np.asarray(r, float)))
    peak = np.maximum.accumulate(eq)
    return float(((eq - peak) / peak).min())

def _strategy_returns(pos: pd.Series, fwd: pd.Series, cost_bps: float):
    turnover = pos.diff().abs()
    turnover.iloc[0] = abs(pos.iloc[0])
    return pos * fwd - turnover * cost_bps / 1e4, turnover

def _fit_predict(X_tr, y_tr, X_te) -> np.ndarray:
    if y_tr.nunique() < 2:
        return np.zeros(len(X_te))
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=300)).fit(X_tr, y_tr)
    return np.where(m.predict_proba(X_te)[:, 1] > 0.5, 1.0, -1.0)

def run_backtest(df: pd.DataFrame, feature_cols: List[str], val_cfg: Dict[str, Any],
                 cost_bps: float, timeframe: str) -> Dict[str, Any]:
    """
    Chronological evaluation, no look-ahead:
      * features at t predict the return t -> t+1 (target built with shift(-1))
      * walk-forward folds on the first (1 - vault_split) of the data
      * the last vault_split is an untouched 'vault' used once
      * transaction costs are charged on every position change
    """
    ppy = PERIODS_PER_YEAR.get(timeframe, 252)
    data = df[feature_cols + ["close"]].replace([np.inf, -np.inf], np.nan).copy()
    data["fwd"] = np.log(data["close"]).diff().shift(-1)
    data = data.dropna().reset_index(drop=True)
    y = (data["fwd"] > 0).astype(int)
    X = data[feature_cols]
    fwd = data["fwd"]

    cut = int(len(data) * (1 - val_cfg.get("vault_split", 0.2)))
    tr_w, va_w = val_cfg["train_window"], val_cfg["val_window"]
    step = max(val_cfg["step_size"], 1)

    preds, start, folds = [], 0, 0
    while start + tr_w + va_w <= cut:
        tr = slice(start, start + tr_w)
        va = np.arange(start + tr_w, start + tr_w + va_w)
        preds.append(pd.Series(_fit_predict(X.iloc[tr], y.iloc[tr], X.iloc[va]), index=va))
        start += step
        folds += 1
    if not preds:
        raise ValueError("Not enough data for walk-forward windows")
    oos_pos = pd.concat(preds)
    oos_pos = oos_pos[~oos_pos.index.duplicated(keep="last")].sort_index()

    oos_ret, turnover = _strategy_returns(oos_pos, fwd.loc[oos_pos.index], cost_bps)
    bench_ret = fwd.loc[oos_pos.index]  # buy & hold benchmark over the same period

    dev = np.arange(cut)
    vault = np.arange(cut, len(data))
    in_pos = pd.Series(_fit_predict(X.iloc[dev], y.iloc[dev], X.iloc[dev]), index=dev)
    in_ret, _ = _strategy_returns(in_pos, fwd.iloc[dev], cost_bps)
    v_pos = pd.Series(_fit_predict(X.iloc[dev], y.iloc[dev], X.iloc[vault]), index=vault)
    v_ret, _ = _strategy_returns(v_pos, fwd.iloc[vault], cost_bps)

    eq, eq_b = np.exp(oos_ret.cumsum().values), np.exp(bench_ret.cumsum().values)
    stride = max(1, len(eq) // 200)
    n = len(oos_ret)
    sharpe = _sharpe(oos_ret, ppy)
    y_oos = y.loc[oos_pos.index]
    return {
        "sharpe": sharpe,
        "max_dd": _max_dd(oos_ret),
        "total_return": float(eq[-1] - 1),
        "t_stat": float(sharpe / np.sqrt(ppy) * np.sqrt(n)),
        "accuracy": float(((oos_pos > 0).astype(int) == y_oos).mean()),
        "benchmark_sharpe": _sharpe(bench_ret, ppy),
        "benchmark_return": float(eq_b[-1] - 1),
        "benchmark_accuracy": float(max(y_oos.mean(), 1 - y_oos.mean())),
        "in_sample_sharpe": _sharpe(in_ret, ppy),
        "vault_sharpe": _sharpe(v_ret, ppy),
        "vault_max_dd": _max_dd(v_ret),
        "avg_turnover": float(turnover.mean()),
        "n_folds": folds,
        "n_oos_periods": n,
        "cost_bps": cost_bps,
        "equity_curve": [float(v) for v in eq[::stride]],
        "benchmark_curve": [float(v) for v in eq_b[::stride]],
    }
