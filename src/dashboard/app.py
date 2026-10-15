"""Streamlit dashboard for the Crypto Fraud Detector.

Runs in two modes, chosen automatically:
- Local: if a FastAPI server is reachable (API_URL), the dashboard calls it.
- Standalone (e.g. Streamlit Cloud): if no API is reachable, the dashboard
  loads the model in-process via FraudPredictor. This lets it deploy as a
  single app with no separate backend.

Run locally with the API:  make api  (terminal 1), then
                           streamlit run src/dashboard/app.py  (terminal 2)
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


# On Streamlit Cloud the Etherscan key arrives via st.secrets, not .env.
# Copy it into the environment so EtherscanClient (which reads os.getenv) finds it.
try:
    if "ETHERSCAN_API_KEY" in st.secrets and not os.getenv("ETHERSCAN_API_KEY"):
        os.environ["ETHERSCAN_API_KEY"] = st.secrets["ETHERSCAN_API_KEY"]
except Exception:
    pass




# --- API mode ---------------------------------------------------------------

def api_reachable() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except requests.RequestException:
        return False


def api_model_info() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/model/info", timeout=REQUEST_TIMEOUT)
        return r.json() if r.status_code == 200 else None
    except requests.RequestException:
        return None


def api_score(address: str) -> tuple[dict | None, str | None]:
    try:
        r = requests.post(
            f"{API_URL}/predict", json={"address": address}, timeout=REQUEST_TIMEOUT
        )
    except requests.RequestException as exc:
        return None, f"Cannot reach the API at {API_URL}. ({exc})"
    if r.status_code == 200:
        return r.json(), None
    if r.status_code == 422:
        return None, "Invalid Ethereum address. Use 0x followed by 40 hex characters."
    if r.status_code == 503:
        return None, "Model not loaded on the server."
    return None, f"Server error ({r.status_code}): {r.json().get('detail', r.text)}"


# --- standalone mode (in-process model) -------------------------------------

@st.cache_resource
def load_predictor():
    """Load FraudPredictor once and cache it across reruns."""
    from src.models.predict import FraudPredictor

    return FraudPredictor()


def local_model_info() -> dict | None:
    try:
        p = load_predictor()
        return {
            "best_model": p.meta["best_model"],
            "feature_count": len(p.feature_names),
            "metrics": p.meta.get("metrics", {}),
        }
    except Exception:
        return None


def local_score(address: str) -> tuple[dict | None, str | None]:
    import re

    if not re.match(r"^0x[0-9a-fA-F]{40}$", address):
        return None, "Invalid Ethereum address. Use 0x followed by 40 hex characters."
    try:
        return load_predictor().score_address(address), None
    except Exception as exc:  # noqa: BLE001
        return None, f"Failed to score address: {exc}"


# --- pick mode once per session ---------------------------------------------

USE_API = api_reachable()
get_info = api_model_info if USE_API else local_model_info
score_address = api_score if USE_API else local_score


# --- UI ---------------------------------------------------------------------

st.title("Crypto Fraud Detector")
st.caption("Score an Ethereum wallet address for fraud risk using graph-based ML.")

info = get_info()
if info is None:
    st.error(
        "Model is unavailable. "
        + ("The API is running but has no model loaded." if USE_API
           else "Could not load the model. Check that model files and the "
                "ETHERSCAN_API_KEY secret are present.")
    )
else:
    cols = st.columns(3)
    cols[0].metric("Model", info["best_model"].upper())
    cols[1].metric("Features", info["feature_count"])
    cols[2].metric("Mode", "API" if USE_API else "Standalone")

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
            df = pd.DataFrame(factors).rename(
                columns={
                    "feature": "Feature",
                    "value": "Value",
                    "direction": "Effect on risk",
                    "shap_value": "SHAP",
                }
            )
            st.dataframe(df, hide_index=True, use_container_width=True)

        with st.expander("Raw response"):
            st.json(result)
