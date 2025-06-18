from transformers import AutoTokenizer
from nltk.tokenize import sent_tokenize

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")

def trim_to_token_limit(sentences, max_tokens):
    trimmed = []
    total_tokens = 0
    for s in reversed(sentences):
        t_len = len(tokenizer.encode(s, add_special_tokens=False))
        if total_tokens + t_len > max_tokens:
            break
        trimmed.insert(0, s)
        total_tokens += t_len
    return trimmed

def chunk_text(text, max_chunk_tokens=400, overlap=50):
    sentences = sent_tokenize(text)
    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        token_len = len(tokenizer.encode(sentence, add_special_tokens=False))

        # If a sentence is too large, split it further
        if token_len > max_chunk_tokens:
            words = sentence.split()
            sub_chunk = []
            sub_tokens = 0
            for word in words:
                word_tokens = len(tokenizer.encode(word, add_special_tokens=False))
                if sub_tokens + word_tokens > max_chunk_tokens:
                    if sub_chunk:
                        chunks.append(" ".join(sub_chunk))
                    sub_chunk = [word]
                    sub_tokens = word_tokens
                else:
                    sub_chunk.append(word)
                    sub_tokens += word_tokens
            if sub_chunk:
                chunks.append(" ".join(sub_chunk))
            continue

        if current_tokens + token_len > max_chunk_tokens:
            chunks.append(" ".join(current_chunk))
            current_chunk = trim_to_token_limit(current_chunk, overlap)
            current_tokens = sum(len(tokenizer.encode(s, add_special_tokens=False)) for s in current_chunk)

        current_chunk.append(sentence)
        current_tokens += token_len

    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks
