# src/text_classification.py

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch
import logging


def classify_text(text, model, neutral_threshold=0.85, neutral_gap=0.05):
    """
    Classify the sentiment of the text with support for a NEUTRAL class.
    Incorporates additional logic to handle mixed or ambiguous reviews.
    """
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)

    # Perform inference
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=-1)

    # Extract scores for POSITIVE and NEGATIVE classes
    positive_score = probabilities[0][1].item()
    negative_score = probabilities[0][0].item()

    # Log detailed information
    logging.info(f"Positive score: {positive_score}, Negative score: {negative_score}, Neutral threshold: {neutral_threshold}")
    logging.info(f"Score difference: {abs(positive_score - negative_score)}")

    # Detect ambiguous language
    text_lower = text.lower()
    ambiguous_phrases = ["mixed experience", "partly good", "hit or miss"]
    negative_phrases = ["bland", "uninspiring", "bad", "slow", "poor"]
    positive_phrases = ["amazing", "exceptional", "great", "delicious", "perfect"]

    # Weight phrases
    negative_hits = sum(1 for phrase in negative_phrases if phrase in text_lower)
    positive_hits = sum(1 for phrase in positive_phrases if phrase in text_lower)
    ambiguous_hits = sum(1 for phrase in ambiguous_phrases if phrase in text_lower)

    # Adjust score based on detected phrases
    if ambiguous_hits > 0 or abs(positive_score - negative_score) < neutral_gap:
        label = "NEUTRAL"
        score = min(positive_score, negative_score)
    elif positive_hits > negative_hits:
        label = "POSITIVE"
        score = positive_score
    else:
        label = "NEGATIVE"
        score = negative_score

    logging.info(f"Phrase-weighted classification -> Label: {label}, Positive Hits: {positive_hits}, Negative Hits: {negative_hits}, Ambiguous Hits: {ambiguous_hits}")
    logging.info(f"Final label: {label}, Final score: {score}")
    return label, score


def generate_sentiment_explanation(text: str, sentiment_label: str) -> str:
    """Generate an explanation for the sentiment classification."""
    if sentiment_label == "POSITIVE":
        return "The review highlights positive aspects like 'delicious', 'perfect', or similar terms."
    elif sentiment_label == "NEGATIVE":
        return "The review mentions negative aspects like 'slow', 'bad', or 'inattentive'."
    elif sentiment_label == "NEUTRAL":
        return "The review reflects a balanced mix of positive and negative aspects."
    else:
        return "The sentiment could not be clearly identified."
