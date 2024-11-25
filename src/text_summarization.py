# src/text_summarization.py

from transformers import BartForConditionalGeneration, BartTokenizer
import logging


def load_summarization_model(model_name="facebook/bart-large-cnn"):
    """
    Load a BART model and tokenizer for abstractive summarization.
    """
    try:
        tokenizer = BartTokenizer.from_pretrained(model_name)
        model = BartForConditionalGeneration.from_pretrained(model_name)
        return tokenizer, model
    except Exception as e:
        logging.error(f"Error in load_summarization_model: {str(e)}")
        raise


def generate_summary(text, tokenizer, model, max_length=50, min_length=25, length_penalty=2.0, num_beams=4):
    """
    Generate a summary for the given text using a BART model.
    """
    try:
        inputs = tokenizer.encode(
            "summarize: " + text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        )
        summary_ids = model.generate(
            inputs,
            max_length=max_length,
            min_length=min_length,
            length_penalty=length_penalty,
            num_beams=num_beams,
            early_stopping=True,
        )
        return tokenizer.decode(summary_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    except Exception as e:
        logging.error(f"Error during summarization: {str(e)}")
        return f"Error: {str(e)}"


def summarize_batch(input_texts, tokenizer, model, max_length=50, min_length=25):
    """
    Summarize a batch of input texts.
    """
    summaries = []
    for text in input_texts:
        try:
            summary = generate_summary(text, tokenizer, model, max_length=max_length, min_length=min_length)
            summaries.append(summary)
        except Exception as e:
            summaries.append(f"Error: {str(e)}")
            logging.error(f"Error in summarize_batch for text: {text[:50]}... - {str(e)}")
    return summaries