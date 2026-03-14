import json
import os
import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from src.logger import logger
from src.utils import load_params


def ensure_directory(path: str) -> None:
    """Create directory if it does not exist."""
    try:
        os.makedirs(path, exist_ok=True)
        logger.debug("Ensured directory exists: %s", path)
    except Exception as e:
        logger.error("Failed to create directory %s: %s", path, e)
        raise


def load_evaluation_artifacts(x_test_path: str, y_test_path: str, model_path: str):
    """Load test features, labels, and trained model."""
    try:
        X_test = joblib.load(x_test_path)
        y_test = pd.read_csv(y_test_path).squeeze("columns")
        model = joblib.load(model_path)

        logger.info("Test features loaded from %s", x_test_path)
        logger.info("Test labels loaded from %s", y_test_path)
        logger.info("Trained model loaded from %s", model_path)
        logger.debug("X_test shape: %s", X_test.shape)
        logger.debug("y_test shape: %s", y_test.shape)

        return X_test, y_test, model

    except FileNotFoundError as e:
        logger.error("Evaluation artifact not found: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error while loading evaluation artifacts: %s", e)
        raise


def evaluate_model(X_test, y_test, model) -> dict:
    """Generate predictions and compute evaluation metrics."""
    try:
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, pos_label="spam")),
            "recall": float(recall_score(y_test, y_pred, pos_label="spam")),
            "f1_score": float(f1_score(y_test, y_pred, pos_label="spam"))
        }

        logger.info("Model evaluation completed successfully")
        logger.debug("Evaluation metrics: %s", metrics)

        return metrics

    except Exception as e:
        logger.error("Failed during model evaluation: %s", e)
        raise


def save_metrics(metrics: dict, metrics_path: str) -> None:
    """Save evaluation metrics to JSON."""
    try:
        ensure_directory(os.path.dirname(metrics_path))

        with open(metrics_path, "w") as file:
            json.dump(metrics, file, indent=4)

        logger.info("Evaluation metrics saved to %s", metrics_path)

    except Exception as e:
        logger.error("Failed to save metrics: %s", e)
        raise


def main() -> None:
    """Main model evaluation pipeline."""
    try:
        logger.info("Model evaluation started")

        params = load_params("params.yaml")

        x_test_path = params["feature_engineering"]["x_test_path"]
        y_test_path = params["feature_engineering"]["y_test_path"]
        model_path = params["model_building"]["model_path"]
        metrics_path = params["model_evaluation"]["metrics_path"]

        X_test, y_test, model = load_evaluation_artifacts(
            x_test_path=x_test_path,
            y_test_path=y_test_path,
            model_path=model_path
        )

        metrics = evaluate_model(X_test, y_test, model)
        save_metrics(metrics, metrics_path)

        logger.info("Model evaluation completed successfully")

    except Exception as e:
        logger.exception("Model evaluation pipeline failed: %s", e)
        raise


if __name__ == "__main__":
    main()