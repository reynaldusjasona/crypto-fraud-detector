"""Inference module: score a new Ethereum address for fraud risk.

Loads the trained model and scaler, fetches live transaction data,
extracts features, and returns a risk assessment with explanations.
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.collector.etherscan import EtherscanClient
from src.features.pipeline import extract_features_for_address

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


class FraudPredictor:
    """Load a trained model and score Ethereum addresses."""

    def __init__(self, models_dir: str | None = None):
        models_path = Path(models_dir) if models_dir else MODELS_DIR

        model_path = models_path / "best_model.joblib"
        scaler_path = models_path / "scaler.joblib"
        meta_path = models_path / "model_meta.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"No trained model found at {model_path}. Run training first."
            )

        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        with open(meta_path) as f:
            self.meta = json.load(f)

        self.feature_names = self.meta["feature_names"]
        self.client = EtherscanClient()

        logger.info(f"Loaded {self.meta['best_model']} model with {len(self.feature_names)} features")

    def score_address(self, address: str) -> dict:
        """Score an Ethereum address for fraud risk.

        Args:
            address: Ethereum address (0x...).

        Returns:
            Dict with fraud_probability, risk_level, and top_factors.
        """
        # Extract features
        logger.info(f"Extracting features for {address}")
        raw_features = extract_features_for_address(address, self.client)

        # Build feature vector in model's expected order
        feature_vector = []
        for name in self.feature_names:
            feature_vector.append(raw_features.get(name, 0.0))

        X = pd.DataFrame([feature_vector], columns=self.feature_names)

        # Handle NaN/inf
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

        # Scale
        X_scaled = pd.DataFrame(
            self.scaler.transform(X),
            columns=self.feature_names,
        )

        # Predict
        prob = float(self.model.predict_proba(X_scaled)[0, 1])
        pred = int(prob >= 0.5)

        # Get SHAP explanation
        from src.models.evaluate import explain_prediction

        explanation = explain_prediction(
            self.model, raw_features, self.feature_names,
        )

        return {
            "address": address,
            "fraud_probability": prob,
            "prediction": "FRAUD" if pred == 1 else "LEGITIMATE",
            "risk_level": explanation["risk_level"],
            "top_factors": explanation["top_factors"],
            "feature_count": len(self.feature_names),
            "model_type": self.meta["best_model"],
        }


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    predictor = FraudPredictor()

    # Test with Vitalik's address (should be legitimate)
    test_address = "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
    print(f"\nScoring {test_address}...")

    result = predictor.score_address(test_address)
    print(f"\nPrediction: {result['prediction']}")
    print(f"Fraud probability: {result['fraud_probability']:.4f}")
    print(f"Risk level: {result['risk_level']}")
    print(f"\nTop contributing factors:")
    for factor in result["top_factors"]:
        print(f"  {factor['feature']}: {factor['value']:.4f} "
              f"({factor['direction']} fraud risk, SHAP={factor['shap_value']:.4f})")
