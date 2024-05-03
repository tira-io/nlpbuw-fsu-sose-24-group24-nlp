from pathlib import Path

from joblib import load
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

if __name__ == "__main__":
    # Initialize TIRA client
    tira = Client()

    # Load the data
    df = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "authorship-verification-validation-20240408-training"
    )

    # Define the path for the model
    model_path = Path(__file__).parent / "model.joblib"

    # Check if the model exists, if not, handle the case appropriately
    if not model_path.exists():
        print("Model file not found. Please ensure the model has been trained and the path is correct.")
        exit(1)  # Exit if the model is not found

    # Load the model and make predictions
    model = load(model_path)
    predictions = model.predict(df["text"])
    df["generated"] = predictions
    df = df[["id", "generated"]]

    # Save the predictions
    output_directory = get_output_directory(str(Path(__file__).parent))
    output_file = Path(output_directory) / "predictions.jsonl"
    df.to_json(output_file, orient="records", lines=True)
    print(f"Predictions saved to {output_file}.")
