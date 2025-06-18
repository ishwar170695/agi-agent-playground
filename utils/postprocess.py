import re

def clean_and_format_summary(text, mode="general", max_bullets=5):
    """
    Simple post-processing: deduplicate sentences and clean whitespace.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    seen = set()
    unique_sentences = [s.strip() for s in sentences if s.strip() and not (s.strip().lower() in seen or seen.add(s.strip().lower()))]
    return '\n'.join(unique_sentences)

if __name__ == "__main__":
    # Example usage
    raw_summary = """YOUR_RAW_SUMMARY_HERE"""
    print(clean_and_format_summary(raw_summary, mode="resume"))
    print(clean_and_format_summary(raw_summary, mode="general"))
