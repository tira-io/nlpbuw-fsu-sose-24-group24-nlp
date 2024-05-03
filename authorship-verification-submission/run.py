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

    # Load the model and make predictions
model = load(Path(__file__).parent / "model.joblib")
predictions = model.predict(df["text"])
df["generated"] = predictions
df = df[["id", "generated"]]

# Save the predictions
output_directory = get_output_directory(str(Path(__file__).parent))
output_file = Path(output_directory) / "predictions.jsonl"
df.to_json(output_file, orient="records", lines=True)
print(f"Predictions saved to {output_file}.")
