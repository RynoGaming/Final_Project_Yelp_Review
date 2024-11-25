# Review Insight Engine (RIE)

The Review Insight Engine (RIE) processes customer reviews by classifying sentiment, summarizing content, and extracting keywords/entities. This project is built with FastAPI, Python, and state-of-the-art transformer models.

## Features
1. Sentiment Classification (Positive, Negative, Neutral).
2. Review Summarization.
3. Named Entity Recognition (NER) and Keyword Extraction.
4. REST API for integration with frontend or other systems.

## API Endpoints
### 1. `/classify` 
- Input: Single review
- Output: Sentiment label, score, explanation.

### 2. `/summarize` 
- Input: Single review
- Output: Summary.

### 3. `/summarize-multiple` 
- Input: Multiple reviews
- Output: Summaries for each review.

### 4. `/tag` 
- Input: Single review
- Output: Entities and keywords.

### 5. `/tag-multiple` 
- Input: Multiple reviews
- Output: Entities and keywords for each review.

### 6. `/process` 
- Input: Single review
- Output: Combined sentiment, summary, entities, and keywords.

## Installation
1. Clone the repository:
    ```bash
    git clone <repository-link>
    ```
2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the backend server:
    ```bash
    uvicorn src.main:app --reload --host 127.0.0.1 --port 8001
    ```

## Frontend Usage
1. Open the `index.html` in a browser using Live Server or local hosting.
2. Paste your review in the input box and click submit.
3. View the processed results.

## GitHub Repository
Find the complete code [here](https://github.com/username/repository-name).

## Processing Time Insights
- **Classify:** ~0.5 seconds
- **Summarization:** ~1.5 seconds
- **Tagging (NER and Keywords):** ~2.0 seconds

**Note:** Times may vary depending on hardware and review length.
