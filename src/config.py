import os

# HuggingFace pre-trained model
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LABELS = ['negative', 'neutral', 'positive']

# Paths
RAW_DATA_PATH = "data/raw/raw_dataset.csv"
PROCESSED_DATA_PATH = "data/processed/train.csv"
TEST_DATA_PATH = "data/processed/test.csv"

# MLflow
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns")
EXPERIMENT_NAME = "sentiment_analysis_roberta"
