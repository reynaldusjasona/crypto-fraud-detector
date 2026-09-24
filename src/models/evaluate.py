"""Model evaluation: metrics, confusion matrix, and SHAP analysis.

Provides standardized evaluation for any sklearn-compatible classifier,
plus SHAP-based feature importance for explainability.
"""

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "Model",
) -> dict:
    """Evaluate a trained model on test data.

    Args:
        model: Trained sklearn-compatible classifier.
        X_test: Test features.
        y_test: Test labels.
        model_name: Name for logging.

    Returns:
        Dict of metric_name -> value.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "auc_roc": float(roc_auc_score(y_test, y_prob)),
    }

    cm = confusion_matrix(y_test, y_pred)
    metrics["confusion_matrix"] = cm.tolist()

    report = classification_report(y_test, y_pred, output_dict=True)
    metrics["classification_report"] = report

    logger.info(f"\n{model_name} Results:")
    logger.info(f"  Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Recall:    {metrics['recall']:.4f}")
    logger.info(f"  F1:        {metrics['f1']:.4f}")
    logger.info(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    logger.info(f"  Confusion Matrix:\n{cm}")

    return metrics


def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    max_samples: int = 500,
) -> dict:
    """Compute SHAP values for feature importance.

    Args:
        model: Trained model (tree-based preferred).
        X: Feature DataFrame.
        max_samples: Max samples to explain (for speed).

    Returns:
        Dict with shap_values array and feature importance ranking.
    """
    import shap

    if len(X) > max_samples:
        X_sample = X.sample(max_samples, random_state=42)
    else:
        X_sample = X

    # Use TreeExplainer for tree-based models, otherwise KernelExplainer
    model_type = type(model).__name__
    if model_type in ("XGBClassifier", "RandomForestClassifier", "GradientBoostingClassifier"):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(model.predict_proba, X_sample.iloc[:50])

    shap_values = explainer.shap_values(X_sample)

    # For binary classification, shap_values may be a list of two arrays
    if isinstance(shap_values, list):
        # Use the positive class (fraud)
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    # Feature importance: mean absolute SHAP value per feature
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    importance = pd.Series(mean_abs_shap, index=X_sample.columns)
    importance = importance.sort_values(ascending=False)

    logger.info("\nTop 10 features by SHAP importance:")
    for feat, val in importance.head(10).items():
        logger.info(f"  {feat}: {val:.4f}")

    return {
        "shap_values": shap_vals,
        "feature_names": list(X_sample.columns),
        "feature_importance": importance.to_dict(),
        "sample_indices": list(X_sample.index),
    }


def explain_prediction(
    model: Any,
    features: dict[str, float],
    feature_names: list[str],
) -> dict:
    """Explain a single prediction using SHAP.

    Args:
        model: Trained model.
        features: Feature dict for one address.
        feature_names: Ordered list of feature names the model expects.

    Returns:
        Dict with prediction, probability, and top contributing features.
    """
    import shap

    # Build feature vector in the right order
    X = pd.DataFrame([{name: features.get(name, 0.0) for name in feature_names}])

    # Predict
    prob = model.predict_proba(X)[0, 1]
    pred = int(prob >= 0.5)

    # SHAP explanation
    model_type = type(model).__name__
    if model_type in ("XGBClassifier", "RandomForestClassifier"):
        explainer = shap.TreeExplainer(model)
    else:
        explainer = shap.KernelExplainer(model.predict_proba, X)

    shap_values = explainer.shap_values(X)

    if isinstance(shap_values, list):
        shap_vals = shap_values[1][0]
    else:
        shap_vals = shap_values[0]

    # Build explanation: top features pushing toward fraud
    explanations = []
    shap_series = pd.Series(shap_vals, index=feature_names)
    top_features = shap_series.abs().sort_values(ascending=False).head(5)

    for feat_name in top_features.index:
        shap_val = shap_series[feat_name]
        feat_val = features.get(feat_name, 0.0)
        direction = "increases" if shap_val > 0 else "decreases"
        explanations.append({
            "feature": feat_name,
            "value": feat_val,
            "shap_value": float(shap_val),
            "direction": direction,
        })

    return {
        "prediction": pred,
        "fraud_probability": float(prob),
        "risk_level": _risk_level(prob),
        "top_factors": explanations,
    }


def _risk_level(prob: float) -> str:
    """Convert probability to a human-readable risk level."""
    if prob >= 0.8:
        return "CRITICAL"
    elif prob >= 0.6:
        return "HIGH"
    elif prob >= 0.4:
        return "MEDIUM"
    elif prob >= 0.2:
        return "LOW"
    else:
        return "MINIMAL"
