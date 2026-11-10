"""Stopword configuration for the TweetEval Offensive task (Youssef).

Why a *custom* stopword list instead of ``nltk.corpus.stopwords`` as-is?
-----------------------------------------------------------------------
NLTK's English list contains negations (``not``, ``no``, ``nor``, ``don't``,
``isn't`` ...). In an offensive-language task a negation reverses the meaning
of a sentence:

    "you are an idiot"          -> offensive
    "you are not an idiot"      -> not offensive

Removing ``not`` therefore destroys the very signal the classifier needs, so
every negation (including the contraction forms with an apostrophe that
``src/preprocessing.clean_text`` deliberately keeps) is *protected*: it is
removed from NLTK's list before the list is used.

The placeholder ``user`` (a ``@mention`` rewritten by the cleaning step) can
also be dropped on demand, because it appears in ~93 % of the tweets of both
classes and therefore carries almost no discriminative information.

Nothing in this module is applied automatically: the variants are compared
with numbers in ``notebooks/03_preprocessing_experiments.ipynb`` and only the
winner is kept for the final pipeline.

Public API
----------
NEGATIONS, PRONOUNS, PROTECTED_TOKENS
build_stopwords(keep_pronouns=False, drop_user=False, extra_keep=())
remove_stopwords(text, stop_words=None)
remove_stopwords_series(series, stop_words=None)
stopword_stats(texts, stop_words=None)

Author: Youssef (preprocessing experiments) - CI1 Machine Learning mini-project.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Sequence

__all__ = [
    "NEGATIONS",
    "PRONOUNS",
    "PLACEHOLDER_TOKENS",
    "PROTECTED_TOKENS",
    "build_stopwords",
    "remove_stopwords",
    "remove_stopwords_series",
    "stopword_stats",
    "missing_protected_tokens",
    "DEFAULT_STOPWORDS",
]

# --------------------------------------------------------------------------- #
# 1. Tokens that must NEVER be removed
# --------------------------------------------------------------------------- #
#: Negations. Plain forms, "n't" forms with an apostrophe (kept by the
#: cleaning step), and the fused forms NLTK knows about.
NEGATIONS: frozenset[str] = frozenset(
    {
        # plain negations
        "no", "not", "nor", "never", "nothing", "nobody", "none", "neither",
        "nowhere", "cannot", "without",
        # "not" + verb, written as one token
        "cannot", "cant", "wont", "dont", "doesnt", "didnt", "isnt", "arent",
        "wasnt", "werent", "havent", "hasnt", "hadnt", "wouldnt", "couldnt",
        "shouldnt", "aint",
        # contractions preserved by the cleaning step (apostrophe kept)
        "can't", "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't",
        "weren't", "won't", "wouldn't", "couldn't", "shouldn't", "hasn't",
        "haven't", "hadn't", "ain't", "mustn't", "mightn't", "needn't",
        "shan't", "let's not",
    }
)

#: Pronouns and possessives. They are stopwords for NLTK, but in a tweet the
#: *target* of an insult matters ("you are an idiot" vs "I am an idiot"), so
#: variant B2 keeps them.
PRONOUNS: frozenset[str] = frozenset(
    {
        "i", "me", "my", "mine", "myself",
        "you", "your", "yours", "yourself", "yourselves",
        "he", "him", "his", "himself",
        "she", "her", "hers", "herself",
        "it", "its", "itself",
        "we", "us", "our", "ours", "ourselves",
        "they", "them", "their", "theirs", "themselves",
    }
)

#: Token created by the cleaning step for every ``@mention``.
PLACEHOLDER_TOKENS: frozenset[str] = frozenset({"user"})

#: Everything ``remove_stopwords`` must **always** leave untouched, whatever
#: list it is given: the negations. Pronouns and ``user`` are *conditionally*
#: protected - only because :func:`build_stopwords` does not put them in the
#: list unless you ask for it.
PROTECTED_TOKENS: frozenset[str] = NEGATIONS


# --------------------------------------------------------------------------- #
# 2. Building the list
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _nltk_english_stopwords() -> frozenset[str]:
    """NLTK's English stopword list, with a readable error if the corpus is missing."""
    try:
        from nltk.corpus import stopwords
    except ImportError as exc:  # pragma: no cover - dependency problem
        raise ImportError(
            "NLTK is required for stopword removal: pip install nltk"
        ) from exc

    try:
        words = stopwords.words("english")
    except LookupError as exc:  # corpus not downloaded yet
        raise LookupError(
            "The NLTK stopwords corpus is missing. Run once:\n"
            "    python -c \"import nltk; nltk.download('stopwords')\""
        ) from exc
    return frozenset(w.lower() for w in words)


def build_stopwords(
    keep_pronouns: bool = False,
    drop_user: bool = False,
    extra_keep: Iterable[str] = (),
) -> frozenset[str]:
    """Return the stopword set used by the experiments.

    Parameters
    ----------
    keep_pronouns:
        ``True`` keeps pronouns (variant *B2*: the target of the insult matters).
        ``False`` uses NLTK's behaviour and removes them.
    drop_user:
        ``True`` adds the ``user`` placeholder to the list (variant *B3*), so a
        tweet that only contained mentions can become empty - counted and
        reported in the notebook.
    extra_keep:
        Any additional tokens to protect from removal (e.g. domain words such
        as ``offensive`` or ``hate`` that you never want to lose).

    Notes
    -----
    Negations are **always** protected, whatever the options: this is the whole
    point of this module.
    """
    stop_words = set(_nltk_english_stopwords())

    # 1. negations are never stopwords
    stop_words -= NEGATIONS

    # 2. optional: keep the pronouns (and, with them, the placeholder "user")
    if keep_pronouns:
        stop_words -= PRONOUNS

    # 3. optional: drop the @mention placeholder
    if drop_user:
        stop_words.add("user")

    # 4. caller-specific protections (applied last: they always win)
    stop_words -= {token.lower() for token in extra_keep}

    return frozenset(stop_words)


#: Default list used by ``remove_stopwords`` when no list is given:
#: NLTK minus negations, pronouns kept out of the vocabulary, ``user`` kept.
DEFAULT_STOPWORDS: frozenset[str] = build_stopwords()


# --------------------------------------------------------------------------- #
# 3. Applying the list
# --------------------------------------------------------------------------- #
def remove_stopwords(text: str, stop_words: Iterable[str] | None = None) -> str:
    """Remove stopwords from an already-cleaned tweet.

    The tokenisation is a plain ``str.split()`` on purpose: the cleaning step
    produced single spaces, and a whitespace split keeps ``don't`` / ``f***``
    as one token, exactly what ``TfidfVectorizer(token_pattern=r"[a-z0-9'*]+")``
    needs. A tweet that becomes empty stays an empty string - never ``NaN`` -
    so ``TfidfVectorizer`` can still handle it (all-zero row).

    >>> remove_stopwords("you are not an idiot")
    'not idiot'
    >>> remove_stopwords("you are not an idiot", build_stopwords(keep_pronouns=True))
    'you not idiot'
    >>> remove_stopwords("this is f***ing bullshit", build_stopwords())
    'f***ing bullshit'
    """
    words = str(text).split()
    if not words:
        return ""

    vocabulary_to_drop = DEFAULT_STOPWORDS if stop_words is None else stop_words

    kept = [word for word in words
            if word in PROTECTED_TOKENS or word not in vocabulary_to_drop]
    return " ".join(kept)


def remove_stopwords_series(series, stop_words: Iterable[str] | None = None):
    """Vectorised helper for a pandas Series (returns a new Series of ``str``)."""
    return series.astype(str).map(lambda t: remove_stopwords(t, stop_words))


# --------------------------------------------------------------------------- #
# 4. Diagnostics used by the notebook
# --------------------------------------------------------------------------- #
def stopword_stats(texts: Sequence[str], stop_words: Iterable[str] | None = None) -> dict:
    """Report what the removal cost: tweets emptied and tokens removed.

    Returns a dict with ``n_tweets``, ``n_empty_after``, ``pct_empty``,
    ``avg_words_before``, ``avg_words_after`` and ``avg_words_removed``.
    """
    vocabulary_to_drop = DEFAULT_STOPWORDS if stop_words is None else set(stop_words)

    n_tweets = len(texts)
    total_before = total_after = n_empty = 0
    for text in texts:
        before = str(text).split()
        after = remove_stopwords(text, vocabulary_to_drop).split()
        total_before += len(before)
        total_after += len(after)
        n_empty += int(len(after) == 0)

    return {
        "n_tweets": n_tweets,
        "n_empty_after": n_empty,
        "pct_empty": round(100 * n_empty / n_tweets, 2) if n_tweets else 0.0,
        "avg_words_before": round(total_before / n_tweets, 2) if n_tweets else 0.0,
        "avg_words_after": round(total_after / n_tweets, 2) if n_tweets else 0.0,
        "avg_words_removed": round((total_before - total_after) / n_tweets, 2) if n_tweets else 0.0,
    }


def missing_protected_tokens(stop_words: Iterable[str]) -> list[str]:
    """Self-check: which negations leaked into a stopword list (should be none).

    Used by the notebook and by ``tests/test_stopwords_and_lemmatization.py``
    to prove that no negation is removed by mistake.
    """
    stop_words = set(stop_words)
    return sorted(token for token in PROTECTED_TOKENS if token in stop_words)


if __name__ == "__main__":  # tiny manual check: python -m src.stopwords_config
    demo = "you are not an idiot , don't be one"
    print("default  :", remove_stopwords(demo))
    print("keep pron:", remove_stopwords(demo, build_stopwords(keep_pronouns=True)))
    print("negations leaked:", missing_protected_tokens(DEFAULT_STOPWORDS) or "none")
    print("size:", len(DEFAULT_STOPWORDS), "words")
