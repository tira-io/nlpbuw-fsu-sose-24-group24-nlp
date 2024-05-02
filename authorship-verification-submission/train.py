from pathlib import Path

from joblib import dump
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from tira.rest_api_client import Client

if __name__ == "__main__":
    # Load the data
    tira = Client()
    text = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    text = text.set_index("id")
    labels = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "authorship-verification-train-20240408-training"
    )
    df = text.join(labels.set_index("id"))

    # Train the model
    model = Pipeline([
        ("vectorizer", TfidfVectorizer()),  # Using TF-IDF vectorizer instead of CountVectorizer
        ("classifier", LogisticRegression())  # Using Logistic Regression instead of Naive Bayes
    ])
    model.fit(df["text"], df["generated"])

    # Save the model
    dump(model, Path(__file__).parent / "model.joblib")