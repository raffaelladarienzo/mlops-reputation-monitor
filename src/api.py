from fastapi import FastAPI, status
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, Any
from src.model import SentimentModel



app = FastAPI(
    title="Sentiment Analysis API",
    description=(
        "A REST API for predicting sentiment (positive, neutral, negative) "
        "using a pre-trained RoBERTa model hosted locally or via Hugging Face."
    ),
    version="1.0.0"
)

# Load the model once when the application starts
model = SentimentModel()




class TextInput(BaseModel):
    """
    Data schema for input text payload.

    Attributes
    ----------
    text : str
        The text to be analyzed for sentiment classification.
    """
    text: str



@app.get("/", response_class=HTMLResponse)
async def root() -> str:
    """
    Root endpoint that confirms the API is running and returns an HTML response.

    Returns
    -------
    str
        A simple HTML page confirming that the Sentiment Analysis API is active.
    """
    return """
    <!DOCTYPE html>
    <html lang="it">
    <head>
        <meta charset="UTF-8">
        <title>Sentiment Analysis API</title>
        <link rel="shortcut icon" href="#">
        <style>
            body {
                font-family: Arial, sans-serif;
                background-color: #f9f9f9;
                color: #333;
            }
            .container {
                text-align: center;
                margin-top: 100px;
            }
            a {
                color: #007BFF;
                text-decoration: none;
            }
            a:hover {
                text-decoration: underline;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Sentiment Analysis API up and running</h1>
        </div>
    </body>
    </html>
    """



@app.get("/health", status_code=status.HTTP_200_OK)
async def health_check() -> Dict[str, str]:
    """
    Health check endpoint to verify API and model availability.

    Returns
    -------
    Dict[str, str]
        A JSON response with the current health status of the API and model.
    """
    try:
        _ = model.predict("health check")
        return {"status": "ok", "model": "loaded", "message": "API and model are healthy"}

    except Exception as e:
        return {
            "status": "error",
            "model": "unavailable",
            "message": f"Health check failed: {str(e)}"
        }



@app.post("/predict")
async def predict_sentiment(input_data: TextInput) -> Dict[str, Any]:
    """
    Predict the sentiment of the provided text using the SentimentModel.

    Parameters
    ----------
    input_data : TextInput
        The input text payload for which the sentiment will be predicted.

    Returns
    -------
    Dict[str, Any]
        A JSON response containing:
        - "input": Original text
        - "label": Predicted sentiment ('negative', 'neutral', 'positive')
        - "score": Model confidence score (float between 0 and 1)
    """
    prediction = model.predict(input_data.text)
    return {"input": input_data.text, **prediction}


if __name__ == '__main__':

    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)