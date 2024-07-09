from flask import Flask, request, jsonify
import joblib
import numpy as np
import pandas as pd
from textblob import TextBlob
from scipy.sparse import hstack, csr_matrix
import spacy
import re
import emoji
from gensim.models import KeyedVectors
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
# Load spaCy English model
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
nlp.max_length = 2000000

# Load necessary components
model = joblib.load('XGBoost_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
word2vec_embeddings = KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)

# Define necessary preprocessing functions
def clean_text(text):
    text = emoji.demojize(text, delimiters=("", " "))
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

def preprocess_text(text, slang_dict):
    if text is None or text.strip() == "":
        return None
    text = replace_slang(text, slang_dict)
    text = clean_text(text)
    doc = nlp(text)
    return " ".join(token.lemma_ for token in doc if not token.is_stop and token.is_alpha)

def replace_slang(text, slang_dict):
    pattern = re.compile(r'\b(' + '|'.join(re.escape(key) for key in slang_dict.keys()) + r')\b')
    return pattern.sub(lambda x: slang_dict[x.group()], text)

def document_vector(word_list, embeddings):
    valid_words = [word for word in word_list if word in embeddings.key_to_index]
    if not valid_words:
        return np.zeros(embeddings.vector_size)
    return np.mean([embeddings[word] for word in valid_words], axis=0)

def prepare_new_data(text, tfidf_vectorizer, word2vec_embeddings, slang_dict):
    processed_text = preprocess_text(text, slang_dict)
    sentiment = TextBlob(text).sentiment.polarity
    tfidf_features = tfidf_vectorizer.transform([processed_text])
    sentiment = np.array([sentiment]).reshape(-1, 1)
    word2vec_features = document_vector(processed_text.split(), word2vec_embeddings).reshape(1, -1)
    return hstack([tfidf_features, csr_matrix(sentiment), csr_matrix(word2vec_features)])

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    text = data['text']
    slang_dict = {'lol': 'laughing out loud', 'brb': 'be right back', 'gtg': 'got to go', 'imo': 'in my opinion'}
    features = prepare_new_data(text, tfidf_vectorizer, word2vec_embeddings, slang_dict)
    prediction = model.predict(features)[0]
    label_to_category = {0: 'unrelated', 1: 'pro-ed', 2:'pro-recovery'}
    category = label_to_category.get(prediction, "Unknown")
    return jsonify({'category': category})

if __name__ == '__main__':
    app.run(debug=True)


