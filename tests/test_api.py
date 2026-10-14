"""Tests for the FastAPI service.

The predictor is mocked so tests never call Etherscan or load the real
model. This keeps them fast, offline, and deterministic.
"""

import pytest
from fastapi.testclient import TestClient

from src.api import main


class FakePredictor:
    """Stand-in for FraudPredictor with no network or model dependency."""

    meta = {
        "best_model": "xgboost",
        "metrics": {"xgboost": {"f1": 0.88, "auc_roc": 0.98}},
    }
    feature_names = ["avg_value_received", "total_ether_sent"]

    def score_address(self, address: str) -> dict:
        return {
            "address": address,
            "fraud_probability": 0.91,
            "prediction": "FRAUD",
            "risk_level": "CRITICAL",
            "top_factors": [
                {
                    "feature": "total_ether_sent",
                    "value": 1234.5,
                    "direction": "increases",
                    "shap_value": 0.42,
                }
            ],
            "model_type": "xgboost",
        }


@pytest.fixture
def client():
    """TestClient with a mocked predictor injected into app state."""
    main.state["predictor"] = FakePredictor()
    with TestClient(main.app) as c:
        # TestClient's lifespan would overwrite state with a real predictor,
        # so re-inject the fake after startup.
        main.state["predictor"] = FakePredictor()
        yield c
    main.state.clear()


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_model_info(client):
    resp = client.get("/model/info")
    assert resp.status_code == 200
    body = resp.json()
    assert body["best_model"] == "xgboost"
    assert body["feature_count"] == 2
    assert "metrics" in body


def test_predict_valid_address(client):
    addr = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
    resp = client.post("/predict", json={"address": addr})
    assert resp.status_code == 200
    body = resp.json()
    assert body["address"] == addr
    assert body["prediction"] == "FRAUD"
    assert body["risk_level"] == "CRITICAL"
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert len(body["top_factors"]) == 1


def test_predict_rejects_bad_address(client):
    resp = client.post("/predict", json={"address": "not-an-address"})
    assert resp.status_code == 422


def test_predict_rejects_short_address(client):
    resp = client.post("/predict", json={"address": "0x123"})
    assert resp.status_code == 422


def test_predict_unavailable_when_no_model(client):
    main.state["predictor"] = None
    resp = client.post(
        "/predict",
        json={"address": "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"},
    )
    assert resp.status_code == 503
