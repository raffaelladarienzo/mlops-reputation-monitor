import requests


BASE_URL = "http://127.0.0.1:8000"

def check_health():
    """
    Check the /health endpoint.
    """
    url = f"{BASE_URL}/health"
    response = requests.get(url)
    print("Health Check Response:")
    print(response.json())
    print("-" * 50)


def predict_sentiment(text: str):
    """
    Send a text to the /predict endpoint and print the result.

    Parameters
    ----------
    text : str
        Text to analyze for sentiment.
    """
    url = f"{BASE_URL}/predict"
    payload = {"text": text}
    response = requests.post(url, json=payload)
    print(f"Input Text: {text}")
    print("Prediction Response:", response.json())
    print("-" * 50)


if __name__ == "__main__":
    # Ensure the API is running before executing this script
    check_health()

    # Example predictions
    predict_sentiment("I love this product!")
    predict_sentiment("This is terrible, I hate it.")
    predict_sentiment("It's okay, not great but not bad either.")
