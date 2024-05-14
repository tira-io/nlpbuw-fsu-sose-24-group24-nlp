from pathlib import Path
from tqdm import tqdm
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

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

    # Convert text data into character n-grams
    vectorizer = CountVectorizer(analyzer='char', ngram_range=(3, 3))
    X = vectorizer.fit_transform(text_validation['text'])

    # Train Naive Bayes classifier
    clf = MultinomialNB()
    clf.fit(X, targets_validation['lang'])

    # Predict language for validation data
    prediction = clf.predict(X)

    # Create DataFrame for predictions
    prediction_df = pd.DataFrame({'lang': prediction, 'id': text_validation['id']})

    # saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    prediction_df.to_json(
        Path(output_directory) / "predictions.jsonl", orient="records", lines=True
    )
