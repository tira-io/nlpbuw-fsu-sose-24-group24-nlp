import re
import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tira.rest_api_client import Client
from tira.third_party_integrations import get_output_directory
import networkx as nx
from pathlib import Path

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

stop_words = set(nltk.corpus.stopwords.words('english'))
lemmatizer = nltk.WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\d', ' ', text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(token) for token in tokens if token not in stop_words]
    return ' '.join(tokens)

def preprocess_text(text):
    sentences = nltk.sent_tokenize(text)
    cleaned_sentences = [clean_text(sentence) for sentence in sentences]
    cleaned_sentences = [sentence for sentence in cleaned_sentences if sentence.strip() != ""]
    return cleaned_sentences

def embed_sentences(sentences):
    if not sentences:
        return None
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(sentences)
    return tfidf_matrix

def rank_sentences(tfidf_matrix, sentences, top_n=3):
    if tfidf_matrix is None or len(sentences) == 0:
        return "No content available."

    sim_matrix = cosine_similarity(tfidf_matrix)

    nx_graph = nx.from_numpy_array(sim_matrix)
    scores = nx.pagerank(nx_graph)

    ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
    top_sentences = [sentence for score, sentence in ranked_sentences[:top_n]]
    summary = ' '.join(top_sentences)
    return summary

def summarize_story(story, top_n=3):
    sentences = preprocess_text(story)
    tfidf_matrix = embed_sentences(sentences)
    if tfidf_matrix is None:
        return "No content available."
    return rank_sentences(tfidf_matrix, sentences, top_n)

if __name__ == "__main__":
    tira = Client()
    df = tira.pd.inputs("nlpbuw-fsu-sose-24", "summarization-validation-20240530-training").set_index("id")

    df["summary"] = df["story"].apply(lambda x: summarize_story(x, top_n=3))
    df = df.drop(columns=["story"]).reset_index()

    output_directory = get_output_directory(str(Path(__file__).parent))
    df.to_json(Path(output_directory) / "predictions.jsonl", orient="records", lines=True)
    print("Predictions saved to", Path(output_directory) / "predictions.jsonl")
