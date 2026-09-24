"""Training pipeline for fraud detection models.

Trains Random Forest (baseline) and XGBoost (primary) classifiers
on the Kaggle Ethereum fraud dataset. Handles class imbalance
with SMOTE and class weights, runs cross-validation, and saves
the best model.
"""

import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    f1_score,
    make_scorer,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"


def load_and_prepare_data(
    csv_path: str | None = None,
) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """Load the Kaggle dataset and prepare features/labels.

    Args:
        csv_path: Path to the dataset CSV. Defaults to data/transaction_dataset.csv.

    Returns:
        Tuple of (X features DataFrame, y labels Series, feature_names list).
    """
    if csv_path is None:
        csv_path = str(DATA_DIR / "transaction_dataset.csv")

    from src.collector.dataset import load_dataset
    from src.features.pipeline import extract_features_from_dataset

    df = load_dataset(csv_path)
    feature_df = extract_features_from_dataset(df)

    # Separate features and label
    label_col = "flag"
    feature_cols = [c for c in feature_df.columns if c != label_col]

    # Drop non-numeric columns (like token names)
    numeric_cols = []
    for col in feature_cols:
        if feature_df[col].dtype in ["int64", "float64", "int32", "float32"]:
            numeric_cols.append(col)

    X = feature_df[numeric_cols].copy()
    y = feature_df[label_col].copy()

    # Handle any remaining NaN/inf
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    logger.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
    logger.info(f"Fraud rate: {y.mean():.1%} ({y.sum()}/{len(y)})")

    return X, y, numeric_cols


def train_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    use_class_weights: bool = True,
) -> RandomForestClassifier:
    """Train a Random Forest baseline model.

    Args:
        X_train: Training features.
        y_train: Training labels.
        use_class_weights: Whether to use balanced class weights.

    Returns:
        Trained RandomForestClassifier.
    """
    class_weight = "balanced" if use_class_weights else None

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)
    logger.info("Random Forest trained")
    return rf


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    tune_hyperparams: bool = True,
) -> XGBClassifier:
    """Train an XGBoost model with optional hyperparameter tuning.

    Args:
        X_train: Training features.
        y_train: Training labels.
        tune_hyperparams: Whether to run GridSearchCV.

    Returns:
        Trained XGBClassifier.
    """
    # Calculate scale_pos_weight for class imbalance
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / max(n_pos, 1)

    if tune_hyperparams:
        param_grid = {
            "max_depth": [4, 6, 8],
            "learning_rate": [0.01, 0.05, 0.1],
            "n_estimators": [200, 300],
            "subsample": [0.8],
            "colsample_bytree": [0.8],
        }

        xgb = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            use_label_encoder=False,
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        scorer = make_scorer(f1_score)

        grid = GridSearchCV(
            xgb,
            param_grid,
            cv=cv,
            scoring=scorer,
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)

        logger.info(f"Best XGBoost params: {grid.best_params_}")
        logger.info(f"Best CV F1: {grid.best_score_:.4f}")
        return grid.best_estimator_

    else:
        xgb = XGBClassifier(
            max_depth=6,
            learning_rate=0.05,
            n_estimators=300,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=42,
            use_label_encoder=False,
        )
        xgb.fit(X_train, y_train)
        logger.info("XGBoost trained (default params)")
        return xgb


def train_pipeline(
    csv_path: str | None = None,
    tune: bool = True,
    test_size: float = 0.2,
) -> dict:
    """Run the full training pipeline.

    1. Load and prepare data
    2. Split into train/test
    3. Train Random Forest baseline
    4. Train XGBoost (with optional tuning)
    5. Evaluate both
    6. Save the best model

    Args:
        csv_path: Path to dataset CSV.
        tune: Whether to tune XGBoost hyperparameters.
        test_size: Fraction of data for test set.

    Returns:
        Dict with metrics and model paths.
    """
    from src.models.evaluate import evaluate_model

    # Load data
    X, y, feature_names = load_and_prepare_data(csv_path)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y,
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train), columns=feature_names, index=X_train.index,
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test), columns=feature_names, index=X_test.index,
    )

    # Train models
    logger.info("Training Random Forest baseline...")
    rf = train_random_forest(X_train_scaled, y_train)

    logger.info("Training XGBoost...")
    xgb = train_xgboost(X_train_scaled, y_train, tune_hyperparams=tune)

    # Evaluate
    logger.info("Evaluating Random Forest...")
    rf_metrics = evaluate_model(rf, X_test_scaled, y_test, "Random Forest")

    logger.info("Evaluating XGBoost...")
    xgb_metrics = evaluate_model(xgb, X_test_scaled, y_test, "XGBoost")

    # Pick the best model by F1 score
    if xgb_metrics["f1"] >= rf_metrics["f1"]:
        best_model = xgb
        best_name = "xgboost"
        best_metrics = xgb_metrics
    else:
        best_model = rf
        best_name = "random_forest"
        best_metrics = rf_metrics

    logger.info(f"Best model: {best_name} (F1={best_metrics['f1']:.4f})")

    # Save artifacts
    MODELS_DIR.mkdir(exist_ok=True)

    model_path = MODELS_DIR / "best_model.joblib"
    scaler_path = MODELS_DIR / "scaler.joblib"
    meta_path = MODELS_DIR / "model_meta.json"

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)

    meta = {
        "best_model": best_name,
        "feature_names": feature_names,
        "metrics": {
            "random_forest": rf_metrics,
            "xgboost": xgb_metrics,
        },
        "train_size": len(X_train),
        "test_size": len(X_test),
        "fraud_rate": float(y.mean()),
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    logger.info(f"Model saved to {model_path}")
    logger.info(f"Scaler saved to {scaler_path}")
    logger.info(f"Metadata saved to {meta_path}")

    return meta


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )
    results = train_pipeline(tune=True)

    print("\n=== Training Results ===")
    for model_name, metrics in results["metrics"].items():
        print(f"\n{model_name}:")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1:        {metrics['f1']:.4f}")
        print(f"  AUC-ROC:   {metrics['auc_roc']:.4f}")
    print(f"\nBest model: {results['best_model']}")
