import pandas as pd
from sklearn.metrics import classification_report
from src.train_model import predict_label
from src.config import TEST_DATA_PATH

def evaluate() -> None:
    """
    Evaluates the performance of the sentiment model on a test dataset.

    Steps:
    1. Load a sample of the test data from a CSV file.
    2. Predict sentiment labels for each text using "predict_label".
    3. Print the classification report (precision, recall, F1-score).

    Args:
        None

    Returns:
        None
    """
    # Load test data and sample 500 rows
    df: pd.DataFrame = pd.read_csv(TEST_DATA_PATH).sample(500)

    # Predict labels
    preds: list[int] = [predict_label(t) for t in df['text']]

    # Print classification report
    print(classification_report(df['label'], preds))


if __name__ == "__main__":
    evaluate()
