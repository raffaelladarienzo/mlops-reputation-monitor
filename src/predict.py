from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
from src.config import MODEL_NAME, LABELS
from typing import Dict

# Load tokenizer and model
tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model: AutoModelForSequenceClassification = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)


def predict_sentiment(text: str) -> Dict[str, float]:
    """
    Predicts the sentiment probabilities for a given text.

    Args:
        text (str): The input text to classify.

    Returns:
        Dict[str, float]: A dictionary mapping each label to its predicted probability.
                          For example: {"negative": 0.1, "neutral": 0.3, "positive": 0.6}
    """
    tokens = tokenizer(text, return_tensors="pt")
    outputs = model(**tokens)
    probs: np.ndarray = softmax(outputs.logits.detach().numpy()[0])
    return dict(zip(LABELS, probs))


if __name__ == "__main__":
    print(predict_sentiment("I love this company!"))
