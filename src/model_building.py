import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

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


def load_training_data(x_train_path: str, y_train_path: str):
    """Load training features and labels."""
    try:
        X_train = joblib.load(x_train_path)
        y_train = pd.read_csv(y_train_path).squeeze("columns")

        logger.info("Training features loaded from %s", x_train_path)
        logger.info("Training labels loaded from %s", y_train_path)
        logger.debug("X_train shape: %s", X_train.shape)
        logger.debug("y_train shape: %s", y_train.shape)

        return X_train, y_train

    except FileNotFoundError as e:
        logger.error("Training data file not found: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error while loading training data: %s", e)
        raise


def train_model(X_train, y_train, max_iter: int, random_state: int):
    """Train Logistic Regression model."""
    try:
        model = LogisticRegression(
            max_iter=max_iter,
            random_state=random_state
        )
        model.fit(X_train, y_train)

        logger.info("Model training completed successfully")
        logger.debug("Model class: %s", model.__class__.__name__)

        return model

    except Exception as e:
        logger.error("Failed during model training: %s", e)
        raise


def save_model(model, model_path: str) -> None:
    """Save trained model to disk."""
    try:
        ensure_directory(os.path.dirname(model_path))
        joblib.dump(model, model_path)

        logger.info("Trained model saved to %s", model_path)

    except Exception as e:
        logger.error("Failed to save trained model: %s", e)
        raise


def main() -> None:
    """Main model building pipeline."""
    try:
        logger.info("Model building started")

        params = load_params("params.yaml")

        x_train_path = params["feature_engineering"]["x_train_path"]
        y_train_path = params["feature_engineering"]["y_train_path"]

        model_path = params["model_building"]["model_path"]
        max_iter = params["model_building"]["max_iter"]
        random_state = params["model_building"]["random_state"]

        X_train, y_train = load_training_data(x_train_path, y_train_path)
        model = train_model(X_train, y_train, max_iter, random_state)
        save_model(model, model_path)

        logger.info("Model building completed successfully")

    except Exception as e:
        logger.exception("Model building pipeline failed: %s", e)
        raise


if __name__ == "__main__":
    main()