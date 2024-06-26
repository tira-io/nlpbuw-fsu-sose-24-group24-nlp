import json
import pandas as pd
from pathlib import Path
from transformers import T5Tokenizer, T5ForConditionalGeneration
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory

def load_data(text_path, labels_path):
    with open(text_path, 'r') as f:
        text_data = [json.loads(line) for line in f]
    with open(labels_path, 'r') as f:
        labels_data = [json.loads(line) for line in f]
    return pd.DataFrame(text_data), pd.DataFrame(labels_data)

def preprocess_data(text_df, labels_df):
    data = []
    for _, text_row in text_df.iterrows():
        text_id = text_row['id']
        sentence = text_row['sentence']
        labels = labels_df[labels_df['id'] == text_id]['tags'].values[0]
        tokens = sentence.split()
        tag_sequence = ' '.join(labels)
        data.append((sentence, tag_sequence))
    return data

def main():
    tira = Client()

    # Loading validation data (automatically replaced by test data when run on tira)
    text_validation = tira.pd.inputs("nlpbuw-fsu-sose-24", "ner-validation-20240612-training")
    targets_validation = tira.pd.truths("nlpbuw-fsu-sose-24", "ner-validation-20240612-training")

    # Load and preprocess the data
    text_validation, targets_validation = load_data(text_validation, targets_validation)
    validation_data = preprocess_data(text_validation, targets_validation)

    # Initialize tokenizer and model
    model_name = 't5-small'
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    predictions = []
    for sentence, _ in validation_data:
        input_text = f"ner: {sentence}"
        input_ids = tokenizer.encode(input_text, return_tensors='pt')

        output_ids = model.generate(input_ids, max_length=512)
        predicted_tags = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        tags = predicted_tags.split()
        
        tokens = sentence.split()
        assert len(tags) == len(tokens), "Mismatch between number of tokens and tags"
        
        id = text_validation[text_validation['sentence'] == sentence]['id'].values[0]
        predictions.append({"id": id, "tags": tags})

    # Saving the prediction
    output_directory = get_output_directory(str(Path(__file__).parent))
    with open(Path(output_directory) / "predictions.jsonl", 'w') as f:
        for prediction in predictions:
            f.write(json.dumps(prediction) + "\n")

if __name__ == "__main__":
    main()
