import re

def clean_and_format_summary(text, mode="general", max_bullets=5):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    seen = set()
    unique_sentences = [s.strip() for s in sentences if s.strip() and not (s.strip().lower() in seen or seen.add(s.strip().lower()))]
    return '\n'.join(unique_sentences)
