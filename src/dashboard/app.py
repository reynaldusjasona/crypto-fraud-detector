"""Streamlit dashboard for the Crypto Fraud Detector.

Thin UI over the FastAPI service. The user enters an Ethereum address,
the dashboard calls POST /predict, and renders the risk assessment and
the top contributing factors.

Run the API first (make api), then: streamlit run src/dashboard/app.py
"""

import os

import pandas as pd
import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
REQUEST_TIMEOUT = 60

RISK_COLORS = {
    "CRITICAL": "#b91c1c",
    "HIGH": "#ea580c",
    "MEDIUM": "#ca8a04",
    "LOW": "#2563eb",
    "MINIMAL": "#16a34a",
}

st.set_page_config(page_title="Crypto Fraud Detector", page_icon="🔎", layout="centered")


def get_model_info() -> dict | None:
    try:
        resp = requests.get(f"{API_URL}/model/info", timeout=REQUEST_TIMEOUT)
        if resp.status_code == 200:
            return resp.json()
    except requests.RequestException:
        return None
    return None


def score_address(address: str) -> tuple[dict | None, str | None]:
    """Call the API. Returns (result, error_message)."""
    try:
        resp = requests.post(
            f"{API_URL}/predict",
            json={"address": address},
            timeout=REQUEST_TIMEOUT,
        )
    except requests.RequestException as exc:
        return None, f"Cannot reach the API at {API_URL}. Is it running? ({exc})"

    if resp.status_code == 200:
        return resp.json(), None
    if resp.status_code == 422:
        return None, "Invalid Ethereum address. Use 0x followed by 40 hex characters."
    if resp.status_code == 503:
        return None, "Model not loaded on the server. Train the model first."
    detail = resp.json().get("detail", resp.text)
    return None, f"Server error ({resp.status_code}): {detail}"


st.title("Crypto Fraud Detector")
st.caption("Score an Ethereum wallet address for fraud risk using graph-based ML.")

info = get_model_info()
if info is None:
    st.warning(f"API not reachable at {API_URL}. Start it with `make api`.")
else:
    cols = st.columns(2)
    cols[0].metric("Model", info["best_model"].upper())
    cols[1].metric("Features", info["feature_count"])

address = st.text_input(
    "Ethereum address",
    placeholder="0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045",
)

if st.button("Analyze", type="primary", disabled=not address):
    with st.spinner("Fetching transactions and scoring..."):
        result, error = score_address(address.strip())

    if error:
        st.error(error)
    else:
        prob = result["fraud_probability"]
        risk = result["risk_level"]
        color = RISK_COLORS.get(risk, "#6b7280")

        st.markdown(
            f"<h3 style='color:{color};margin-bottom:0'>{result['prediction']} "
            f"&middot; {risk} RISK</h3>",
            unsafe_allow_html=True,
        )
        st.progress(min(max(prob, 0.0), 1.0))
        st.metric("Fraud probability", f"{prob:.1%}")

        factors = result.get("top_factors", [])
        if factors:
            st.subheader("Top contributing factors")
            df = pd.DataFrame(factors)
            df = df.rename(
                columns={
                    "feature": "Feature",
                    "value": "Value",
                    "direction": "Effect on risk",
                    "shap_value": "SHAP",
                }
            )
            st.dataframe(df, hide_index=True, use_container_width=True)

        with st.expander("Raw API response"):
            st.json(result)
