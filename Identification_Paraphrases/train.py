

from tira.rest_api_client import Client
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import matthews_corrcoef
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

    # Calculate decision function values
    decision_function = model.decision_function(X)

    # Find the best threshold for the decision function
    mccs = {}
    thresholds = sorted(set(decision_function))
    for threshold in thresholds:
        y_pred = (decision_function >= threshold).astype(int)
        mcc = matthews_corrcoef(y, y_pred)
        mccs[threshold] = mcc

    best_threshold = max(mccs, key=mccs.get)
    print(f"Best threshold: {best_threshold}")

    # Save the model, vectorizer, and best threshold
    output_directory = Path(__file__).parent
    joblib.dump(model, output_directory / "svc_model.pkl")
    joblib.dump(vectorizer, output_directory / "vectorizer.pkl")
    joblib.dump(best_threshold, output_directory / "best_threshold.pkl")
