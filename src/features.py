from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

from src import config


def fit_tfidf(train_texts, **kwargs):
    """Fit TF-IDF on TRAINING text only and return (vectorizer, X_train_tfidf)."""
    params = dict(config.TFIDF_PARAMS)
    params.update(kwargs)
    vec = TfidfVectorizer(**params)
    return vec, vec.fit_transform(train_texts)


def transform_tfidf(vectorizer, texts):
    """Transform val/test text with the already-fitted training vectorizer."""
    return vectorizer.transform(texts)


def save_vectorizer(vectorizer, path="models/tfidf_vectorizer.joblib"):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(vectorizer, path)


def load_vectorizer(path="models/tfidf_vectorizer.joblib"):
    return joblib.load(path)
