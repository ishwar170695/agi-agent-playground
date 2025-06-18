from transformers import pipeline, AutoTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

model_id = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_id)

# For keyword extraction, use embeddings and simple scoring
def extract_keywords(text: str, top_k: int = 5) -> str:
    # Split text into candidate keywords (simple: unique words, can be improved)
    words = list(set([w.strip('.,;:!?()[]') for w in text.split() if len(w) > 3]))
    if not words:
        return "No keywords found."
    # Get embeddings
    model = pipeline("feature-extraction", model=model_id, tokenizer=tokenizer)
    text_emb = np.mean(model(text)[0], axis=0)
    word_embs = [np.mean(model(word)[0], axis=0) for word in words]
    # Score by cosine similarity
    scores = cosine_similarity([text_emb], word_embs)[0]
    top_indices = np.argsort(scores)[-top_k:][::-1]
    keywords = [words[i] for i in top_indices]
    return ", ".join(keywords)
