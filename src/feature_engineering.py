import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

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


def load_processed_data(train_path: str, test_path: str):
    """Load processed train and test datasets."""
    try:
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        logger.info("Processed train data loaded from %s", train_path)
        logger.info("Processed test data loaded from %s", test_path)
        logger.debug("Train dataframe shape: %s", train_df.shape)
        logger.debug("Test dataframe shape: %s", test_df.shape)

        return train_df, test_df
    except FileNotFoundError as e:
        logger.error("Processed data file not found: %s", e)
        raise
    except pd.errors.ParserError as e:
        logger.error("Failed to parse processed data: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error while loading processed data: %s", e)
        raise


def extract_features(train_df: pd.DataFrame, test_df: pd.DataFrame, max_features: int, ngram_range: tuple):
    """Generate TF-IDF features from text data."""
    try:
        vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range
        )

        X_train = vectorizer.fit_transform(train_df["text"])
        X_test = vectorizer.transform(test_df["text"])

        y_train = train_df["target"]
        y_test = test_df["target"]

        logger.info("TF-IDF feature extraction completed successfully")
        logger.debug("X_train shape: %s", X_train.shape)
        logger.debug("X_test shape: %s", X_test.shape)
        logger.debug("Number of TF-IDF features: %d", len(vectorizer.get_feature_names_out()))

        return X_train, X_test, y_train, y_test, vectorizer

    except KeyError as e:
        logger.error("Missing required column during feature extraction: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error during feature extraction: %s", e)
        raise


def save_features(
    X_train,
    X_test,
    y_train,
    y_test,
    vectorizer,
    x_train_path: str,
    x_test_path: str,
    y_train_path: str,
    y_test_path: str,
    vectorizer_path: str
) -> None:
    """Save transformed features, labels, and vectorizer."""
    try:
        ensure_directory(os.path.dirname(x_train_path))
        ensure_directory(os.path.dirname(vectorizer_path))

        joblib.dump(X_train, x_train_path)
        joblib.dump(X_test, x_test_path)
        y_train.to_csv(y_train_path, index=False)
        y_test.to_csv(y_test_path, index=False)
        joblib.dump(vectorizer, vectorizer_path)

        logger.info("X_train saved to %s", x_train_path)
        logger.info("X_test saved to %s", x_test_path)
        logger.info("y_train saved to %s", y_train_path)
        logger.info("y_test saved to %s", y_test_path)
        logger.info("Vectorizer saved to %s", vectorizer_path)

    except Exception as e:
        logger.error("Failed to save feature artifacts: %s", e)
        raise


def main() -> None:
    """Main feature engineering pipeline."""
    try:
        logger.info("Feature engineering started")

        params = load_params("params.yaml")

        train_path = params["data_preprocessing"]["train_path"]
        test_path = params["data_preprocessing"]["test_path"]

        x_train_path = params["feature_engineering"]["x_train_path"]
        x_test_path = params["feature_engineering"]["x_test_path"]
        y_train_path = params["feature_engineering"]["y_train_path"]
        y_test_path = params["feature_engineering"]["y_test_path"]
        vectorizer_path = params["feature_engineering"]["vectorizer_path"]
        max_features = params["feature_engineering"]["max_features"]
        ngram_range = tuple(params["feature_engineering"]["ngram_range"])

        train_df, test_df = load_processed_data(train_path, test_path)

        X_train, X_test, y_train, y_test, vectorizer = extract_features(
            train_df=train_df,
            test_df=test_df,
            max_features=max_features,
            ngram_range=ngram_range
        )

        save_features(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            vectorizer=vectorizer,
            x_train_path=x_train_path,
            x_test_path=x_test_path,
            y_train_path=y_train_path,
            y_test_path=y_test_path,
            vectorizer_path=vectorizer_path
        )

        logger.info("Feature engineering completed successfully")

    except Exception as e:
        logger.exception("Feature engineering pipeline failed: %s", e)
        raise


if __name__ == "__main__":
    main()