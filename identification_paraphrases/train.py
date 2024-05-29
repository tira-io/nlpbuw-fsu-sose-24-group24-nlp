from tira.rest_api_client import Client
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
import joblib
from pathlib import Path

if __name__ == "__main__":
    # Load the data
    tira = Client()
    text = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training"
    ).set_index("id")
    labels = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-train-20240515-training"
    ).set_index("id")
    df = text.join(labels)

    # Vectorize the sentences using TF-IDF
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(df["sentence1"] + " " + df["sentence2"])
    y = df["label"]

    # Train the SVC model
    model = SVC(kernel='linear', probability=True)
    model.fit(X, y)

    # Save the model and vectorizer
    output_directory = Path(__file__).parent
    joblib.dump(model, output_directory / "svc_model.joblib")
    joblib.dump(vectorizer, output_directory / "vectorizer.joblib")
