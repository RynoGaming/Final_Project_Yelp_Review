# src/main.py

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from src.utils import load_transformer_classifier, load_ner_pipeline
from src.text_summarization import load_summarization_model
from src.handlers import (
    classify_review_handler,
    summarize_review_handler,
    tag_review_handler,
    process_review_handler,
    handle_tag_multiple,
    summarize_reviews_handler
)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for testing. Restrict in production.
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Load models at startup
classifier = load_transformer_classifier(model_name="distilbert-base-uncased-finetuned-sst-2-english")
summarizer_tokenizer, summarizer_model = load_summarization_model()
ner_pipeline = load_ner_pipeline(model_name="dbmdz/bert-large-cased-finetuned-conll03-english")

# Define request models for inputs
class ReviewRequest(BaseModel):
    text: str

class MultiReviewRequest(BaseModel):
    texts: list[str]

class SentimentRequest(BaseModel):
    text: str

# Endpoint: Classify a single review
@app.post("/classify")
def classify_review(request: ReviewRequest):
    return classify_review_handler(request.text, classifier)

# Endpoint: Summarize a single review
@app.post("/summarize")
def summarize_review(request: ReviewRequest):
    return {"summary": summarize_review_handler(request.text, summarizer_tokenizer, summarizer_model)}


# Endpoint: Summarize multiple reviews
@app.post("/summarize-multiple")
def summarize_reviews(request: MultiReviewRequest):
    return summarize_reviews_handler(request, summarizer_tokenizer, summarizer_model)


# Endpoint: Tag entities and keywords in a single review
@app.post("/tag")
def tag_review(request: ReviewRequest):
    return {"entities": tag_review_handler(request.text, ner_pipeline)}


# Endpoint: Tag entities and keywords in multiple reviews
@app.post("/tag-multiple")
def tag_multiple_reviews(request: MultiReviewRequest):
    return handle_tag_multiple(request)


# Endpoint: Process a single review
@app.post("/process")
def process_review(request: SentimentRequest):
    return process_review_handler(request.text, classifier, summarizer_tokenizer, summarizer_model, ner_pipeline)



"""""""""""""""
Run the app using `uvicorn src.main:app --reload --host 127.0.0.1 --port 8001`
Examples for each:
1. /classify
{
  "text": "The service was exceptional, and the food was absolutely delicious. Highly recommend this place!"
}
2. /summarize
{
  "text": "The service was exceptional, and the food was absolutely delicious. The pasta was cooked perfectly, and the dessert was divine. However, the wait time for the main course was a bit longer than expected."
}
3. /summarize-multiple
{
  "texts": [
    "The service was exceptional, and the food was absolutely delicious. The pasta was cooked perfectly, and the dessert was divine. However, the wait time for the main course was a bit longer than expected.",
    "The atmosphere was cozy, and the staff was friendly. The appetizers were great, but the steak was overcooked and lacked seasoning."
  ]
}
4. /tag
{
  "text": "We visited Bella's Italian Bistro and loved the ambiance. The bruschetta was fantastic, but the pasta was disappointing."
}
5. /tag-multiple
{
  "texts": [
    "We visited Bella's Italian Bistro and loved the ambiance. The bruschetta was fantastic, but the pasta was disappointing.",
    "The service at Joe's Grill was slow, but the burgers were amazing."
  ]
}
6. /process
{
  "text": "We visited Olive's Kitchen last night. The ambiance was lovely, and the appetizers were amazing. However, the main course was bland and uninspiring. The staff were attentive, and the dessert was exceptional. Overall, a mixed experience."
}
"""""""""""""""""