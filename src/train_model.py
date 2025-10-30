import mlflow
import mlflow.pyfunc

import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from src.config import (
    MODEL_NAME,
    PROCESSED_DATA_PATH,
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME
)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


def predict_label(text: str) -> int:
    """
    Predicts the sentiment label for a single text.

    Args:
        text (str): The text to classify.

    Returns:
        int: Predicted sentiment label.
             Example: 0 = negative, 1 = neutral, 2 = positive.
    """
    tokens = tokenizer(text, return_tensors="pt")
    outputs = model(**tokens)
    probs = softmax(outputs.logits.detach().numpy()[0])
    return int(np.argmax(probs))


def train_and_log() -> None:
    """
    Trains (or evaluates on a small dataset) and logs the model, parameters,
    and metrics to MLflow.

    Main steps:
    1. Load a sample of preprocessed data from CSV.
    2. Predict labels for each text using "predict_label".
    3. Compute accuracy and F1-score.
    4. Log metrics, parameters, and model to MLflow.
    5. Save a PyFunc wrapper for future inference.

    Returns:
        None
    """
    # Load data and sample
    df = pd.read_csv(PROCESSED_DATA_PATH).sample(1000)
    y_true = df['label'].tolist()
    y_pred = [predict_label(t) for t in df['text']]

    # Compute metrics
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')

    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    with mlflow.start_run():
        mlflow.log_param("model_name", MODEL_NAME)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        # Wrapper for future inference
        class SentimentModelWrapper(mlflow.pyfunc.PythonModel):
            """
            MLflow wrapper for sentiment model inference.
            """
            def predict(self, context, model_input: pd.DataFrame) -> np.ndarray:
                """
                Predicts labels for a DataFrame of texts.

                Args:
                    context: Context provided by MLflow (ignored here).
                    model_input (pd.DataFrame): DataFrame containing a 'text' column.

                Returns:
                    np.ndarray: Array of predicted labels.
                """
                return np.array([predict_label(t) for t in model_input['text']])

        # Log the model
        mlflow.pyfunc.log_model("sentiment_model", python_model=SentimentModelWrapper())

    print(f"✅ Run completed | Accuracy: {acc:.3f}, F1: {f1:.3f}")


if __name__ == "__main__":
    train_and_log()