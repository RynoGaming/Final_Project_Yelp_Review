# src/text_tagging.py

from transformers import pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import logging


def load_ner_pipeline(model_name="dbmdz/bert-large-cased-finetuned-conll03-english"):
    """
    Load a transformer-based Named Entity Recognition (NER) pipeline.
    """
    try:
        return pipeline("ner", model=model_name, grouped_entities=True)
    except Exception as e:
        logging.error(f"Error loading NER pipeline: {str(e)}")
        raise


def tag_entities(text, ner_pipeline):
    """
    Identify entities in the text using the provided NER pipeline.
    """
    try:
        entities = ner_pipeline(text)
        return [
            {
                "entity": entity["entity_group"],
                "word": entity["word"],
                "score": float(entity["score"]),  # Convert numpy.float32 to float
            }
            for entity in entities
        ]
    except Exception as e:
        logging.error(f"Error during tagging: {str(e)}")
        return {"error": str(e)}


def extract_keywords(texts, n_keywords=5, max_features=1000):
    """
    Extract keywords using CountVectorizer and Latent Dirichlet Allocation (LDA).
    """
    try:
        vectorizer = CountVectorizer(stop_words="english", max_features=max_features)
        text_features = vectorizer.fit_transform(texts)

        lda = LatentDirichletAllocation(n_components=1, random_state=42)
        lda.fit(text_features)

        keywords = []
        for topic_idx, topic in enumerate(lda.components_):
            keywords.append([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-n_keywords:]])
        return keywords
    except Exception as e:
        logging.error(f"Error during keyword extraction: {str(e)}")
        return []


def tag_and_extract_keywords(text, ner_pipeline, n_keywords=5):
    """
    Combines NER tagging and keyword extraction for better topic identification.
    """
    try:
        # NER tagging
        entities = tag_entities(text, ner_pipeline)

        # Keyword extraction
        keywords = extract_keywords([text], n_keywords=n_keywords)

        # Combine entities and keywords
        combined_tags = {"entities": entities, "keywords": keywords[0] if keywords else []}
        return combined_tags
    except Exception as e:
        logging.error(f"Error in tag_and_extract_keywords: {str(e)}")
        return {"error": str(e)}


def tag_multiple_texts(input_texts, ner_pipeline, n_keywords=5):
    """
    Identify entities and extract keywords for multiple input texts.
    Combines NER tagging and keyword extraction for each review.
    """
    results = []
    for text in input_texts:
        if not text or len(text.strip()) == 0:
            results.append({"error": "Input text is empty or invalid."})
        else:
            results.append(tag_and_extract_keywords(text, ner_pipeline, n_keywords=n_keywords))
    return results