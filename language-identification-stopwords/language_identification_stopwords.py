from pathlib import Path
from tqdm import tqdm
import pandas as pd
from scipy.sparse import lil_matrix
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

# Function to generate character n-grams
def generate_ngrams(text, n=3):
    ngrams = []
    for i in range(len(text) - n + 1):
        ngrams.append(text[i:i + n])
    return ngrams

# Function to count n-grams frequency
def count_ngrams(text, n=3):
    ngram_counts = {}
    ngrams = generate_ngrams(text, n)
    for ngram in ngrams:
        if ngram in ngram_counts:
            ngram_counts[ngram] += 1
        else:
            ngram_counts[ngram] = 1
    return ngram_counts

# Function to extract n-gram features
def extract_features(texts, n=3):
    num_texts = len(texts)
    features = lil_matrix((num_texts, len(lang_ids)), dtype=int)
    for i, text in enumerate(texts):
        ngram_counts = count_ngrams(text, n)
        for j, lang_id in enumerate(lang_ids):
            features[i, j] = ngram_counts.get(lang_id, 0)
    return features

if __name__ == "__main__":

    tira = Client()

    # loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs(
        "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
    )
    targets_validation = tira.pd.truths(
        "nlpbuw-fsu-sose-24", "language-identification-validation-20240429-training"
    )

    # Define language IDs
    lang_ids = ["af", "az", "bg", "cs", "da", "de", "el", "en", "es", "fi", "fr", "hr", "it", "ko", "nl", "no", "pl", "ru", "ur", "zh"]

    # Extracting features using character n-grams
    n = 3  # Adjust n-gram size as needed
    features = extract_features(text_validation['text'], n)

    # Classifying the data based on features
    prediction = []
    for i in tqdm(range(features.shape[0])):
        max_lang_index = features[i].todense().argmax(axis=1)
        max_lang = lang_ids[max_lang_index[0, 0]]
        prediction.append(max_lang)

    # Create DataFrame for predictions
    prediction_df = pd.DataFrame({'lang': prediction, 'id': text_validation['id']})
    print(prediction_df)

    # saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    prediction_df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
