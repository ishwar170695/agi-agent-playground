from transformers import pipeline, AutoTokenizer

# Load model and tokenizer once
model_id = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(model_id)
summarizer = pipeline("text2text-generation", model=model_id, tokenizer=tokenizer, max_new_tokens=300)

def summarize_text(text: str) -> str:
    prompt = (
        "You are an expert summarizer. Your task is to create a concise, clear, and professional summary "
        "of the following text. Focus on the most important details and avoid redundancy.\n\n" + text
    )
    result = summarizer(prompt, truncation=True)
    return result[0]["generated_text"].strip() if result else ""
