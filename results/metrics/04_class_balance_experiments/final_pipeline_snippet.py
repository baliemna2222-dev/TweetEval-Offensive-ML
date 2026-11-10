# src/final_pipeline.py  — keep only what the experiments justified
from src.preprocessing import clean_text
from src.stopwords_config import build_stopwords, remove_stopwords
from src.lemmatization import lemmatize_text

_STOPWORDS = build_stopwords()          # NLTK minus every negation


def preprocess_text(text, use_stopwords=True, use_lemmatization=True):
    # Emna's cleaning -> Youssef's stopwords (negations protected) -> lemmatisation.
    # Balancing is NOT here: it happens at training time (class_weight="balanced").
    text = clean_text(text)
    if use_stopwords:
        text = remove_stopwords(text, _STOPWORDS)
    if use_lemmatization:
        text = lemmatize_text(text)
    return text
