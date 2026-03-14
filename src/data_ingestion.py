import os
import pandas as pd

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


def load_data(data_url: str) -> pd.DataFrame:
    """Load data from source URL."""
    try:
        df = pd.read_csv(data_url, encoding="latin-1")
        logger.info("Data loaded successfully from %s", data_url)
        logger.debug("Loaded dataframe shape: %s", df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse CSV from %s: %s", data_url, e)
        raise
    except Exception as e:
        logger.error("Unexpected error while loading data from %s: %s", data_url, e)
        raise


def save_raw_data(df: pd.DataFrame, output_path: str) -> None:
    """Save raw dataframe to local storage."""
    try:
        output_dir = os.path.dirname(output_path)
        ensure_directory(output_dir)
        df.to_csv(output_path, index=False)
        logger.info("Raw data saved successfully to %s", output_path)
    except Exception as e:
        logger.error("Failed to save raw data to %s: %s", output_path, e)
        raise


def main() -> None:
    """Main data ingestion pipeline."""
    try:
        logger.info("Data ingestion started")

        params = load_params("params.yaml")
        data_url = params["data_source"]["url"]
        raw_data_path = params["data_source"]["raw_data_path"]

        df = load_data(data_url)
        save_raw_data(df, raw_data_path)

        logger.info("Data ingestion completed successfully")

    except Exception as e:
        logger.exception("Data ingestion pipeline failed: %s", e)
        raise


if __name__ == "__main__":
    main()