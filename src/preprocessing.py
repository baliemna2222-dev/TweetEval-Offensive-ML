import re

def clean_text(text, lowercase=True, remove_urls=True, remove_mentions=False):
    """Light cleaning; keep hashtags, emojis, punctuation and negations."""
    text = str(text)
    if remove_urls:
        text = re.sub(r"https?://\S+|www\\.\S+", " URL ", text, flags=re.I)
    if remove_mentions:
        text = re.sub(r"(?<!\\w)@\w+", " USER ", text)
    if lowercase:
        text = text.lower()
    return re.sub(r"\s+", " ", text).strip()

def clean_texts(texts, **kwargs):
    return [clean_text(t, **kwargs) for t in texts]
