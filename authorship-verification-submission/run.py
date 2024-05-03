import os
from pathlib import Path
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import pandas as pd

# Define the base path relative to the script location in the container
base_path = Path(__file__).parent
model_path = base_path / "model.joblib"

def train_model():
    # Example data for training
    text_data = ["sample text", "more samples", "text data", "text mining"]
    labels = [0, 1, 0, 1]

    model = Pipeline([
        ("vectorizer", TfidfVectorizer()),
        ("classifier", LogisticRegression())
    ])
    model.fit(text_data, labels)
    dump(model, model_path)

def predict():
    if not model_path.exists():
        print("Model not found, training now...")
        train_model()

    model = load(model_path)
    test_data = ["unknown text"]
    predictions = model.predict(test_data)
    print("Predictions:", predictions)

if __name__ == "__main__":
    predict()
