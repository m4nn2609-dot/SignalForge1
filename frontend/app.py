import time
import pandas as pd
import requests
import streamlit as st

API = "http://localhost:8000"
st.set_page_config(page_title="SignalForge Terminal", layout="wide")
st.title("⚡ SignalForge Research Terminal")
st.caption("Quantitative research & experiment management: walk-forward validation, costs, no look-ahead")

with st.sidebar:
    st.header("Research Parameters")
    symbol = st.text_input("Trading Symbol", value="BTCUSDT")
    timeframe = st.selectbox("Timeframe", ["1m", "5m", "15m", "1h", "1d"], index=3)
    cost_bps = st.slider("Transaction cost (bps / trade)", 0.0, 50.0, 5.0, 0.5)
    up = st.file_uploader("Upload OHLCV CSV (optional)", type="csv")
    if up and st.button("Upload dataset", use_container_width=True):
        r = requests.post(f"{API}/datasets/upload", params={"symbol": symbol},
                          files={"file": (up.name, up.getvalue(), "text/csv")})
        st.success(r.json()) if r.ok else st.error(r.json().get("detail", r.text))
    if st.button("🚀 Launch Experiment", use_container_width=True):
        try:
            r = requests.post(f"{API}/research/start",
                              json={"symbol": symbol, "timeframe": timeframe, "cost_bps": cost_bps})
            r.raise_for_status()
            st.session_state.current_exp_id = r.json()["experiment_id"]
        except Exception as e:
            st.error(f"Connection error: {e}")

tab_run, tab_hist = st.tabs(["Experiment", "History"])

with tab_run:
    exp_id = st.session_state.get("current_exp_id")
    if not exp_id:
        st.write("Enter a symbol, then click **Launch Experiment**.")
    else:
        st.info(f"Experiment `{exp_id}`")
        data = None
        with st.spinner("Agents running: Architect → Analyst → Modeler → Auditor"):
            for _ in range(90):
                try:
                    d = requests.get(f"{API}/research/status/{exp_id}", timeout=5).json()
                    if d.get("status") in ("DONE", "ERROR"):
                        data = d
                        break
                except Exception:
                    pass
                time.sleep(1)
        if not data:
            st.warning("Still running... refresh in a moment.")
        else:
            m = data.get("metrics") or {}
            c = st.columns(5)
            c[0].metric("Verdict", data["verdict"])
            c[1].metric("OOS Sharpe", f"{data['sharpe']:.2f}" if data["sharpe"] is not None else "N/A")
            c[2].metric("Buy&Hold Sharpe", f"{m['benchmark_sharpe']:.2f}" if m.get("benchmark_sharpe") is not None else "N/A")
            c[3].metric("Max Drawdown", f"{data['drawdown']:.2%}" if data["drawdown"] is not None else "N/A")
            c[4].metric("Vault Sharpe (unseen)", f"{m['vault_sharpe']:.2f}" if m.get("vault_sharpe") is not None else "N/A")
            st.subheader("Risk Auditor's Reasoning")
            st.write(data["reasoning"])
            if m.get("equity_curve"):
                st.subheader("Out-of-sample equity (after costs) vs buy & hold")
                st.line_chart(pd.DataFrame({"Strategy": m["equity_curve"], "Buy & Hold": m["benchmark_curve"]}))
                st.subheader("Overfitting check")
                st.bar_chart(pd.DataFrame({"Sharpe": [m["in_sample_sharpe"], m["sharpe"], m["vault_sharpe"]]},
                                          index=["In-sample (train)", "Walk-forward OOS", "Vault (unseen)"]))
            with st.expander("Reproducible configuration (saved in DB)"):
                st.json({"features": data["config"], "validation": data["validation_config"],
                         "cleaning": data["cleaning_config"], "proposal_source": m.get("proposal_source"),
                         "stationarity": m.get("stationarity")})

with tab_hist:
    try:
        h = requests.get(f"{API}/research/history", timeout=5).json()
        st.dataframe(pd.DataFrame(h), use_container_width=True)
    except Exception as e:
        st.write(f"API not reachable: {e}")
