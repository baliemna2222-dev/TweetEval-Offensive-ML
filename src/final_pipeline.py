"""Full text pipeline = Emna's cleaning + Youssef's optional steps, driven by src/config.py.

    raw tweet --clean_text--> text_clean --[stopwords]--[lemmatisation]--> text_final --TF-IDF--> model

Balancing is NOT part of this pipeline: it only happens on the training rows (src/balancing.py).
"""
from functools import lru_cache

from src import config
from src.preprocessing import clean_text

# The variants compared in notebook 03 (settings understood by make_preprocessor)
VARIANTS = {
    "A_clean_only":               dict(stopwords=False, lemmatize=False),
    "B_stopwords":                dict(stopwords=True, lemmatize=False),
    "B2_stopwords_keep_pronouns": dict(stopwords=True, lemmatize=False, keep_pronouns=True),
    "B3_stopwords_drop_user":     dict(stopwords=True, lemmatize=False, drop_user=True),
    "C_stopwords_lemma":          dict(stopwords=True, lemmatize=True),
}


def current_settings():
    """Settings chosen in src/config.py (decisions of notebooks 03 and 04)."""
    return dict(stopwords=config.USE_STOPWORDS, lemmatize=config.USE_LEMMATIZATION,
                keep_pronouns=config.KEEP_PRONOUNS, drop_user=config.DROP_USER)


@lru_cache(maxsize=None)
def _stopword_set(keep_pronouns, drop_user):
    from src.stopwords_config import build_stopwords
    return build_stopwords(keep_pronouns=keep_pronouns, drop_user=drop_user)


def make_preprocessor(stopwords=False, lemmatize=False, keep_pronouns=False, drop_user=False):
    """Return a function text_clean -> text_final (the input must already be cleaned)."""
    def fn(text):
        if stopwords:
            from src.stopwords_config import remove_stopwords
            text = remove_stopwords(text, _stopword_set(keep_pronouns, drop_user))
        if lemmatize:
            from src.lemmatization import lemmatize_text
            text = lemmatize_text(text)
        return text
    return fn


def preprocess_text(text, already_clean=False, **overrides):
    """Full pipeline for ONE tweet (use already_clean=True when `text` is a text_clean value)."""
    settings = {**current_settings(), **overrides}
    text = text if already_clean else clean_text(text)
    return make_preprocessor(**settings)(text)


def preprocess_series(series, already_clean=True, **overrides):
    settings = {**current_settings(), **overrides}
    fn = make_preprocessor(**settings)
    return series.apply(fn if already_clean else (lambda t: fn(clean_text(t))))


def predict_tweets(texts, vectorizer, model):
    """Raw tweets -> predicted labels (0 = non-offensive, 1 = offensive)."""
    cleaned = [preprocess_text(t) for t in texts]
    return model.predict(vectorizer.transform(cleaned))
