# src/handlers.py

from fastapi import HTTPException
from src.text_classification import classify_text, generate_sentiment_explanation
from src.text_summarization import generate_summary, summarize_batch, load_summarization_model
from src.text_tagging import tag_and_extract_keywords, tag_multiple_texts
from src.utils import load_transformer_classifier, load_ner_pipeline
import time
import logging


# Load necessary models
classifier = load_transformer_classifier(model_name="distilbert-base-uncased-finetuned-sst-2-english")
summarizer_tokenizer, summarizer_model = load_summarization_model()
ner_pipeline = load_ner_pipeline(model_name="dbmdz/bert-large-cased-finetuned-conll03-english")


def classify_review_handler(text: str, classifier: dict):
    """
    Classify the sentiment of a single review.
    """
    label, score = classify_text(text, classifier["model"])
    explanation = generate_sentiment_explanation(text, label)  # Correctly pass the label here
    return {"label": label, "score": score, "explanation": explanation}


def summarize_review_handler(text, tokenizer, model):
    if not text:
        raise HTTPException(status_code=400, detail="Text is required.")
    try:
        return generate_summary(text, tokenizer, model)
    except Exception as e:
        logging.error(f"Error in summarize_review_handler for text: {text[:50]}... - {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to summarize the review.")


def handle_summarize_multiple(texts, tokenizer, model):
    """
    Handle summarizing multiple reviews.
    Consistent with single review summarization logic.
    """
    summaries = [
        generate_summary(
            text,
            tokenizer,
            model,
            max_length=50,
            min_length=10,
            length_penalty=2.0,
            num_beams=4
        )
        for text in texts
    ]
    return summaries


def summarize_reviews_handler(request, tokenizer, model):
    """
    Handler for summarizing multiple reviews.
    Validates input and calls the core summarization logic.
    """
    texts = request.texts
    if not texts or not isinstance(texts, list) or not all(isinstance(t, str) and t.strip() for t in texts):
        raise Exception("A list of non-empty text strings is required.")
    return handle_summarize_multiple(texts, tokenizer, model)


def tag_review_handler(text, ner_pipeline):
    if not text:
        raise HTTPException(status_code=400, detail="Text is required.")
    try:
        return tag_and_extract_keywords(text, ner_pipeline)
    except Exception as e:
        logging.error(f"Error in tag_review_handler for text: {text[:50]}... - {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to tag the review.")


def handle_tag_multiple(request):
    """
    Tag entities and extract keywords from multiple reviews.
    """
    texts = request.texts
    if not texts or len(texts) == 0:
        raise HTTPException(status_code=400, detail="No reviews provided.")
    try:
        tagged_reviews = tag_multiple_texts(texts, ner_pipeline)
        return {"tagged_reviews": tagged_reviews}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to tag the reviews. Error: {str(e)}")


import time

def process_review_handler(text, classifier, summarizer_tokenizer, summarizer_model, ner_pipeline):
    """
    Process a single review.
    """
    try:
        start_time = time.time()

        # Sentiment Classification
        start_classify = time.time()
        label, score = classify_text(text, classifier["model"])
        classify_time = time.time() - start_classify

        explanation = generate_sentiment_explanation(text, sentiment_label=label)

        # Summarization
        start_summarize = time.time()
        summary = generate_summary(text, summarizer_tokenizer, summarizer_model)
        summarize_time = time.time() - start_summarize

        # Named Entity Recognition and Keyword Extraction
        start_tagging = time.time()
        entities = tag_and_extract_keywords(text, ner_pipeline)
        tagging_time = time.time() - start_tagging

        total_time = time.time() - start_time
        processing_time = round(total_time, 2)

        logging.info(f"Time breakdown: Classify: {classify_time:.2f}s, Summarize: {summarize_time:.2f}s, Tagging: {tagging_time:.2f}s, Total: {total_time:.2f}s")

        return {
            "sentiment": {
                "label": label,
                "score": score,
                "explanation": explanation
            },
            "summary": summary,
            "entities": entities,
            "metadata": {
                "processing_time_s": processing_time,
                "classify_time_s": round(classify_time, 2),
                "summarize_time_s": round(summarize_time, 2),
                "tagging_time_s": round(tagging_time, 2)
            }
        }
    except Exception as e:
        logging.error(f"Error in process_review_handler for text: {text[:50]}... - {str(e)}")
        raise e