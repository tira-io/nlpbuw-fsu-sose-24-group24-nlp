from pathlib import Path
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import joblib

if __name__ == "__main__":
    # Load the data
    tira = Client()
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "paraphrase-identification-validation-20240515-training"
    ).set_index("id")

    # Load the trained model and vectorizer
    output_directory = get_output_directory(str(Path(__file__).parent))
    model = joblib.load((Path(__file__).parent) / "svc_model.joblib")
    vectorizer = joblib.load((Path(__file__).parent) / "vectorizer.joblib")

    # Vectorize the sentences
    X = vectorizer.transform(df["sentence1"] + " " + df["sentence2"])

    # Predict the labels using the SVC model
    decision_function = model.decision_function(X)
    best_threshold = 0  # You might want to set an appropriate threshold
    df["label"] = (decision_function >= best_threshold).astype(int)
    df = df.drop(columns=["sentence1", "sentence2"]).reset_index()

    # Save the predictions
    df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
