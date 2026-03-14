import os
import pandas as pd
from sklearn.model_selection import train_test_split

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


def load_raw_data(file_path: str) -> pd.DataFrame:
    """Load raw dataset from local storage."""
    try:
        df = pd.read_csv(file_path)
        logger.info("Raw data loaded successfully from %s", file_path)
        logger.debug("Raw dataframe shape: %s", df.shape)
        return df
    except FileNotFoundError as e:
        logger.error("Raw data file not found at %s", file_path)
        raise e
    except pd.errors.ParserError as e:
        logger.error("Failed to parse raw CSV file %s: %s", file_path, e)
        raise e
    except Exception as e:
        logger.error("Unexpected error while loading raw data: %s", e)
        raise e


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize dataset columns."""
    try:
        columns_to_drop = ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"]
        existing_columns_to_drop = [col for col in columns_to_drop if col in df.columns]
        df = df.drop(columns=existing_columns_to_drop)

        df = df.rename(columns={"v1": "target", "v2": "text"})

        logger.info("Data cleaning completed successfully")
        logger.debug("Columns after cleaning: %s", df.columns.tolist())
        logger.debug("Cleaned dataframe shape: %s", df.shape)
        return df
    except KeyError as e:
        logger.error("Expected columns missing during cleaning: %s", e)
        raise e
    except Exception as e:
        logger.error("Unexpected error during data cleaning: %s", e)
        raise e


def split_data(df: pd.DataFrame, test_size: float, random_state: int):
    """Split dataset into train and test sets."""
    try:
        train_df, test_df = train_test_split(
            df,
            test_size=test_size,
            random_state=random_state,
            stratify=df["target"]
        )
        logger.info("Data split completed successfully")
        logger.debug("Train shape: %s", train_df.shape)
        logger.debug("Test shape: %s", test_df.shape)
        return train_df, test_df
    except Exception as e:
        logger.error("Failed to split data: %s", e)
        raise e


def save_processed_data(train_df: pd.DataFrame, test_df: pd.DataFrame, train_path: str, test_path: str) -> None:
    """Save processed train and test datasets."""
    try:
        ensure_directory(os.path.dirname(train_path))
        ensure_directory(os.path.dirname(test_path))

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        logger.info("Processed train data saved to %s", train_path)
        logger.info("Processed test data saved to %s", test_path)
    except Exception as e:
        logger.error("Failed to save processed datasets: %s", e)
        raise e


def main() -> None:
    """Main preprocessing pipeline."""
    try:
        logger.info("Data preprocessing started")

        params = load_params("params.yaml")
        raw_data_path = params["data_source"]["raw_data_path"]

        train_path = params["data_preprocessing"]["train_path"]
        test_path = params["data_preprocessing"]["test_path"]
        test_size = params["data_preprocessing"]["test_size"]
        random_state = params["data_preprocessing"]["random_state"]

        df = load_raw_data(raw_data_path)
        cleaned_df = clean_data(df)
        train_df, test_df = split_data(cleaned_df, test_size, random_state)
        save_processed_data(train_df, test_df, train_path, test_path)

        logger.info("Data preprocessing completed successfully")

    except Exception as e:
        logger.exception("Data preprocessing pipeline failed: %s", e)
        raise


if __name__ == "__main__":
    main()