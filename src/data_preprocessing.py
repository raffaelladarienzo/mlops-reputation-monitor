import pandas as pd
import re
from sklearn.model_selection import train_test_split
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, TEST_DATA_PATH
from typing import Tuple


def clean_text(text: str) -> str:
    """
    Cleans a single text string by removing URLs, mentions, and non-alphanumeric characters.

    Args:
        text (str): The raw text input.

    Returns:
        str: The cleaned and lowercased text.
    """
    text = re.sub(r"http\S+", "", text)      # Remove URLs
    text = re.sub(r"@\w+", "", text)         # Remove mentions
    text = re.sub(r"[^A-Za-z0-9\s]+", "", text)  # Remove non-alphanumeric chars
    return text.strip().lower()


def preprocess_data() -> None:
    """
    Loads raw CSV data, cleans and preprocesses it, maps labels to a 0-1-2 scale,
    splits into training and test sets, and saves the processed files.

    Steps:
    1. Read raw CSV data.
    2. Keep only 'text' and 'label' columns.
    3. Clean the text using `clean_text`.
    4. Map original labels (0, 2, 4) to (0=negative, 1=neutral, 2=positive).
    5. Split into train/test with stratification.
    6. Save processed CSV files.

    Args:
        None

    Returns:
        None
    """
    # Load raw data
    df: pd.DataFrame = pd.read_csv(RAW_DATA_PATH, encoding='latin1', header=None)
    df = df.rename(columns={0: 'label', 5: 'text'})
    df = df[['text', 'label']]

    # Clean text
    df['text'] = df['text'].apply(clean_text)

    # Map labels: 0 -> negative, 2 -> neutral, 4 -> positive
    df['label'] = df['label'].map({0: 0, 2: 1, 4: 2})

    # Split into train and test sets
    train: pd.DataFrame
    test: pd.DataFrame
    train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label'])

    # Save processed files
    train.to_csv(PROCESSED_DATA_PATH, index=False)
    test.to_csv(TEST_DATA_PATH, index=False)

    print(f"✅ Train samples: {len(train)} | Test samples: {len(test)}")


if __name__ == "__main__":
    preprocess_data()
