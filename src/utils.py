# src/utils.py

import pandas as pd
import re
import nltk
import logging
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from transformers import pipeline, DistilBertForSequenceClassification, DistilBertTokenizer

# Download NLTK resources if not already available
nltk.download('stopwords')
nltk.download('punkt')

# Setup logging
logging.basicConfig(level=logging.INFO)


# === Dataset Handling Functions ===

def load_data(file_path, text_column='text', label_column='stars'):
    """Load a dataset from a JSON file and select relevant columns."""
    logging.info("Loading data...")
    df = pd.read_json(file_path, lines=True)
    logging.info("Data loaded successfully.")
    df = df[[text_column, label_column]]
    df = df.rename(columns={text_column: 'text', label_column: 'label'})
    return df


# === Preprocessing Utilities ===

def preprocess_text(text):
    """Preprocess text by cleaning, tokenizing, and removing stop words."""
    text = re.sub(r'http\S+|www\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)  # Remove mentions
    text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
    text = text.lower().strip()  # Lowercase and strip whitespace
    tokens = word_tokenize(text)  # Tokenize text
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]  # Remove stop words
    return ' '.join(tokens)


# === Classification Utilities ===

def load_transformer_classifier(model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    """
    Load a transformer-based classifier for sentiment analysis.
    """
    tokenizer = DistilBertTokenizer.from_pretrained(model_name)
    model = DistilBertForSequenceClassification.from_pretrained(model_name)
    return {"model": model, "tokenizer": tokenizer}


# === Tagging Utilities ===

def load_ner_pipeline(model_name="dbmdz/bert-large-cased-finetuned-conll03-english"):
    """Load a transformer-based named entity recognition (NER) pipeline."""
    return pipeline("ner", model=model_name, grouped_entities=True)


def tag_entities(text, ner_pipeline):
    """Identify entities in the text using the provided NER pipeline."""
    try:
        entities = ner_pipeline(text)
        return [
            {
                "entity": entity["entity_group"],
                "word": entity["word"],
                "score": float(entity["score"])  # Convert numpy.float32 to float
            }
            for entity in entities
        ]
    except Exception as e:
        logging.error(f"Error during tagging: {str(e)}")
        return {"error": str(e)}