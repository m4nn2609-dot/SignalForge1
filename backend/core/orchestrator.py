import copy
import math
from typing import Dict, Any, Callable, List, Tuple

import numpy as np

from backend.core.state import ResearchState
from backend.core.llm import LocalLLM
from backend.core.data import load_ohlcv, clean_data
from backend.engine.quant_engine import QuantEngine
from backend.engine.backtest import run_backtest
from backend.schemas.config import FeatureSuiteConfig, ValidationConfig

DEFAULT_PROPOSAL: Dict[str, Any] = {
    "cleaning": {"drop_duplicates": True, "fill_method": "ffill", "winsorize_pct": 0.001},
    "features": {
        "target_column": "close",
        "features": [
            {"operator": "log_return", "input_column": "close", "output_column": "ret_1"},
            {"operator": "zscore", "input_column": "close", "params": {"window": 20}, "output_column": "z_20"},
            {"operator": "zscore", "input_column": "close", "params": {"window": 50}, "output_column": "z_50"},
            {"operator": "rsi", "input_column": "close", "params": {"window": 14}, "output_column": "rsi_14"},
            {"operator": "zscore", "input_column": "volume", "params": {"window": 20}, "output_column": "vol_z_20"},
        ],
    },
    "validation": {"train_window": 500, "val_window": 100, "step_size": 100, "vault_split": 0.2},
}

PROPOSER_SYSTEM = (
    "You are a quant research proposer. Reply with ONE JSON object with keys "
    "'cleaning', 'features', 'validation'. Allowed operators: sma, ema, rsi, zscore, log_return. "
    "Input columns: open, high, low, close, volume. Example:\n"
    + '{"cleaning":{"drop_duplicates":true,"fill_method":"ffill","winsorize_pct":0.001},'
    + '"features":{"target_column":"close","features":[{"operator":"zscore","input_column":"close",'
    + '"params":{"window":20},"output_column":"z_20"},{"operator":"log_return","input_column":"close",'
    + '"output_column":"ret_1"}]},"validation":{"train_window":500,"val_window":100,"step_size":100,"vault_split":0.2}}'
)


def sanitize(obj):
    """Make results JSON-safe (numpy -> python, NaN/inf -> None)."""
    if isinstance(obj, dict):
        return {str(k): sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return None if (math.isnan(obj) or math.isinf(obj)) else float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


class SignalForgeOrchestrator:
    """
    Decoupled pipeline: AI *proposes* JSON configs, the deterministic engine *executes* them.
    Implemented as a plain sequential runner (no graph framework): the flow is linear, so this
    is the fastest and simplest option, with zero orchestration overhead and no extra dependency.

    Steps: data_architect -> quant_analyst -> modeler -> risk_auditor
    """
    def __init__(self):
        self.llm = LocalLLM()
        self.engine = QuantEngine()
        self.steps: List[Tuple[str, Callable[[Dict[str, Any]], Dict[str, Any]]]] = [
            ("data_architect", self.node_data_architect),
            ("quant_analyst", self.node_quant_analyst),
            ("modeler", self.node_modeler),
            ("risk_auditor", self.node_risk_auditor),
        ]

    # ---------- AI proposal (single LLM call, validated, with deterministic fallback) ----------
    def _propose(self, state) -> Tuple[Dict[str, Any], str]:
        default = copy.deepcopy(DEFAULT_PROPOSAL)
        if not self.llm.enabled:
            return default, "deterministic-defaults"
        try:
            raw = self.llm.generate_json(
                PROPOSER_SYSTEM, f"Symbol: {state['symbol']}, timeframe: {state['timeframe']}. Propose configs.")
            feats = raw["features"]
            if isinstance(feats, list):
                feats = {"features": feats}
            feats.setdefault("target_column", "close")
            val = {**default["validation"], **(raw.get("validation") or {})}
            FeatureSuiteConfig(**feats)
            ValidationConfig(**val)
            return {"cleaning": raw.get("cleaning") or default["cleaning"], "features": feats, "validation": val}, "llm"
        except Exception as e:
            return default, f"fallback-defaults ({type(e).__name__})"

    # ---------- Nodes ----------
    def node_data_architect(self, state) -> Dict[str, Any]:
        proposal, source = self._propose(state)
        raw = state.get("raw_data")
        if raw is None:
            raw = load_ohlcv(state["symbol"], state["timeframe"])
        cleaned = clean_data(raw, proposal["cleaning"])
        return {"proposal": proposal, "proposal_source": source, "raw_data": raw,
                "cleaned_data": cleaned, "cleaning_config": proposal["cleaning"],
                "iteration_count": state.get("iteration_count", 0) + 1}

    def node_quant_analyst(self, state) -> Dict[str, Any]:
        df = state["cleaned_data"]
        errors = list(state.get("errors", []))

        def build(feature_json):
            cfg = FeatureSuiteConfig(**feature_json)           # deterministic guardrail
            out, kept = df, []
            for op in cfg.features:                            # skip any bad operator, keep the rest
                try:
                    out = self.engine.execute_pipeline(out, [op])
                    kept.append(op)
                except Exception as e:
                    errors.append(f"Skipped feature {op.operator.value}({op.input_column}): {e}")
            return out, kept, cfg

        processed, kept, cfg = build(state["proposal"]["features"])
        if not kept:
            processed, kept, cfg = build(DEFAULT_PROPOSAL["features"])

        # Quant rigor: ADF test on every feature, only stationary ones reach the model
        report, good = {}, []
        for op in kept:
            col = self.engine.output_name(op)
            report[col] = self.engine.run_stationarity_test(processed[col])
            if report[col].get("is_stationary"):
                good.append(col)
        if not good:
            raise ValueError("No stationary features survived the ADF test")
        report["_selected_features"] = good
        feature_json = {"target_column": cfg.target_column,
                        "features": [o.model_dump(mode="json") for o in kept]}
        return {"feature_config": feature_json, "processed_features": processed,
                "stationarity_report": report, "errors": errors}

    def node_modeler(self, state) -> Dict[str, Any]:
        df, n = state["processed_features"], len(state["processed_features"])
        v = state["proposal"]["validation"]
        tr = int(min(max(v["train_window"], 100), n * 0.45))
        va = int(min(max(v["val_window"], 50), n * 0.15))
        val_cfg = {"train_window": tr, "val_window": va,
                   "step_size": int(max(v["step_size"], va)),
                   "vault_split": float(min(max(v.get("vault_split", 0.2), 0.1), 0.4))}
        cols = state["stationarity_report"]["_selected_features"]
        metrics = run_backtest(df, cols, val_cfg, float(state.get("cost_bps", 5.0)), state["timeframe"])
        metrics["features_used"] = cols
        return {"validation_config": val_cfg, "performance_metrics": metrics}

    def node_risk_auditor(self, state) -> Dict[str, Any]:
        """Adversarial, rule-based audit: tries to reject the signal."""
        m = state["performance_metrics"]
        problems, notes = [], []
        if m["t_stat"] < 2.0:
            problems.append(f"out-of-sample t-stat {m['t_stat']:.2f} < 2 (not statistically significant)")
        if m["sharpe"] <= m["benchmark_sharpe"]:
            problems.append(f"Sharpe {m['sharpe']:.2f} does not beat buy-and-hold {m['benchmark_sharpe']:.2f}")
        if m["vault_sharpe"] <= 0:
            problems.append(f"fails on the untouched vault (Sharpe {m['vault_sharpe']:.2f})")
        gap = m["in_sample_sharpe"] - m["sharpe"]
        if gap > 1.5:
            problems.append(f"overfitting: in-sample Sharpe {m['in_sample_sharpe']:.2f} vs out-of-sample {m['sharpe']:.2f}")
        if m["max_dd"] < -0.35:
            problems.append(f"max drawdown {m['max_dd']:.1%} exceeds 35%")
        notes.append(f"{m['n_folds']} walk-forward folds, {m['cost_bps']:g} bps cost per unit turnover, "
                     f"accuracy {m['accuracy']:.1%} vs naive {m['benchmark_accuracy']:.1%}")
        verdict = "REJECTED" if problems else "APPROVED"
        reasoning = ("Rejected: " + "; ".join(problems) + ". " if problems
                     else "Approved: significant, benchmark-beating, and holds on the unseen vault. ") + "; ".join(notes)
        return {"audit_verdict": verdict, "audit_reasoning": reasoning}

    # ---------- Runner ----------
    def run_experiment(self, initial_state: ResearchState) -> Dict[str, Any]:
        state: Dict[str, Any] = dict(initial_state)
        state.setdefault("errors", [])
        for name, step in self.steps:
            try:
                state.update(step(state))
            except Exception as e:
                state["errors"] = list(state["errors"]) + [f"{name}: {e}"]
                state["audit_verdict"] = "ERROR"
                state["audit_reasoning"] = f"Pipeline failed in {name}: {e}"
                break
        return state
