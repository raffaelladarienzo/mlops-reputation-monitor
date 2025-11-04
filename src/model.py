import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification
    )

from typing import Dict

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

class SentimentModel:
    """
    A class that wraps a pre-trained Hugging Face model for sentiment analysis.
    
    This class loads the 'cardiffnlp/twitter-roberta-base-sentiment-latest' model,
    designed to classify social media text into three sentiment categories:
    'negative', 'neutral', and 'positive'.
    
    Methods
    -------
    predict(text: str) -> Dict[str, float or str]:
        Predicts the sentiment of the given input text and returns both
        the predicted label and the confidence score.
    """

    def __init__(self) -> None:
        """
        Initialize the SentimentModel by loading the tokenizer and model from Hugging Face.
        """
        print("Loading sentiment model...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        self.labels = ['negative', 'neutral', 'positive']

    def predict(self, text: str) -> Dict[str, float or str]:
        """
        Predict the sentiment of a given input text.
        
        Parameters
        ----------
        text : str
            The input text (e.g., a tweet or a comment) to be classified.
        
        Returns
        -------
        Dict[str, float or str]
            A dictionary containing:
            - "label": The predicted sentiment ('negative', 'neutral', or 'positive')
            - "score": The confidence score of the prediction (float between 0 and 1)
        """
        inputs = self.tokenizer(text, return_tensors="pt")
        outputs = self.model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        label = self.labels[torch.argmax(probs)]
        score = float(torch.max(probs))
        return {"label": label, "score": round(score, 3)}
