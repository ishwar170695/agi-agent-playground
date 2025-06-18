import re

def clean_and_format_summary(text, mode="general", max_bullets=5):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    seen = set()
    unique_sentences = [s.strip() for s in sentences if s.strip() and not (s.strip().lower() in seen or seen.add(s.strip().lower()))]
    return '\n'.join(unique_sentences)

def preprocess_for_planner(doc, max_chars=1000):
    # Remove lines with URLs, emails, phone numbers
    lines = doc.splitlines()
    filtered = []
    for line in lines:
        if re.search(r'(https?://|www\.|@|\d{3,}[- .]\d{3,}[- .]\d{4,})', line):
            continue
        filtered.append(line)
    cleaned = '\n'.join(filtered)
    return cleaned[:max_chars]
