"""Lemmatisation for the TweetEval Offensive task (Youssef).

Goal
----
Reduce the vocabulary (``fucking``, ``fucked``, ``fucks`` -> ``fuck``) so that
rare inflected forms share statistical strength with frequent ones. On a
11,916-tweet training set with a 18k-token vocabulary this is the main lever
against sparsity.

What is protected (and why)
---------------------------
* ``user``            the ``@mention`` placeholder created by the cleaning step;
* ``f***``, ``sh*t``  masked profanity - WordNet does not know it and would
                      mangle it (``f***`` is also a strong offensive signal);
* ``don't``, ``isn't`` contractions - lemmatising them would destroy the
                      negation that :mod:`src.stopwords_config` protects;
* numbers             ``2020`` must not become ``2020``'s lemma;
* any other token that is not purely alphabetic (hashtags glued to digits,
                      leftover symbols).

Two WordNet passes are needed
-----------------------------
``WordNetLemmatizer`` needs the part of speech. Without ``pos=`` every word is
treated as a **noun**, and ``was`` becomes ``wa``. We therefore lemmatise as a
**verb** first (``was`` -> ``be``, ``hating`` -> ``hate``) then as a **noun**
(``idiots`` -> ``idiot``, ``bitches`` -> ``bitch``).

Known limitations (reported in the notebook, not bugs to fix)
-------------------------------------------------------------
* ``hating`` -> ``hat`` (WordNet maps the verb *to hate* onto the noun *hat*
  through its ``-ing`` stripping rules) - a well-known WordNet weakness;
* unknown slang / Twitter spellings (``idiottt``, ``libtards``) are returned
  unchanged;
* lemmatising is ~3x slower than the other steps: measured in the notebook.

Public API
----------
PROTECTED_TOKENS
is_protected(token)
lemmatize_word(word)
lemmatize_text(text)
lemmatize_series(series)
lemmatization_stats(texts)

Author: Youssef (preprocessing experiments) - CI1 Machine Learning mini-project.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

__all__ = [
    "PROTECTED_TOKENS",
    "is_protected",
    "lemmatize_word",
    "lemmatize_text",
    "lemmatize_series",
    "lemmatization_stats",
]

#: Placeholder produced by the cleaning step for every ``@mention``.
PROTECTED_TOKENS: frozenset[str] = frozenset({"user"})


@lru_cache(maxsize=1)
def _lemmatizer():
    """Cached WordNet lemmatizer, with a readable error if the corpus is missing."""
    try:
        from nltk.stem import WordNetLemmatizer
    except ImportError as exc:  # pragma: no cover - dependency problem
        raise ImportError("NLTK is required: pip install nltk") from exc

    lemmatizer = WordNetLemmatizer()
    try:  # trigger the corpus lookup once, so the error message is actionable
        lemmatizer.lemmatize("testing")
    except LookupError as exc:
        raise LookupError(
            "The NLTK wordnet corpus is missing. Run once:\n"
            "    python -c \"import nltk; nltk.download('wordnet')\""
        ) from exc
    return lemmatizer


def is_protected(token: str) -> bool:
    """``True`` when the token must be returned untouched.

    Protects the placeholder, masked profanity (contains ``*``), contractions
    (contain ``'``) and anything that is not a plain alphabetic word (digits,
    mixed alphanumeric hashtags, leftovers).
    """
    if not token:
        return True
    if token in PROTECTED_TOKENS:
        return True
    if "*" in token or "'" in token:  # masked profanity / contraction
        return True
    if not token.isalpha():  # numbers and alphanumeric tokens
        return True
    return False


def lemmatize_word(word: str) -> str:
    """Lemmatise a single token (verb pass, then noun pass).

    >>> lemmatize_word("was")
    'be'
    >>> lemmatize_word("idiots")
    'idiot'
    >>> lemmatize_word("don't")
    "don't"
    >>> lemmatize_word("f***")
    'f***'
    """
    if is_protected(word):
        return word
    lemmatizer = _lemmatizer()
    return lemmatizer.lemmatize(lemmatizer.lemmatize(word, pos="v"), pos="n")


def lemmatize_text(text: str) -> str:
    """Lemmatise every whitespace-separated token of an already-cleaned tweet.

    Tokens are joined back with single spaces, so the output stays compatible
    with ``TfidfVectorizer(token_pattern=r"[a-z0-9'*]+")``.
    """
    words = str(text).split()
    if not words:
        return ""
    return " ".join(lemmatize_word(word) for word in words)


def lemmatize_series(series):
    """Vectorised helper for a pandas Series (returns a new Series of ``str``)."""
    return series.astype(str).map(lemmatize_text)


def lemmatization_stats(texts: Sequence[str]) -> dict:
    """Report the effect of lemmatisation: tokens changed and vocabulary size.

    Returns ``n_tweets``, ``n_tokens``, ``n_tokens_changed``,
    ``pct_tokens_changed``, ``vocab_before`` and ``vocab_after``.
    """
    before_vocab: set[str] = set()
    after_vocab: set[str] = set()
    n_tokens = n_changed = 0

    for text in texts:
        for word in str(text).split():
            lemma = lemmatize_word(word)
            n_tokens += 1
            n_changed += int(lemma != word)
            before_vocab.add(word)
            after_vocab.add(lemma)

    return {
        "n_tweets": len(texts),
        "n_tokens": n_tokens,
        "n_tokens_changed": n_changed,
        "pct_tokens_changed": round(100 * n_changed / n_tokens, 2) if n_tokens else 0.0,
        "vocab_before": len(before_vocab),
        "vocab_after": len(after_vocab),
    }


if __name__ == "__main__":  # tiny manual check: python -m src.lemmatization
    demo = "he was hating the idiots and bitches , don't you think ? f*** that user 2020"
    print("before:", demo)
    print("after :", lemmatize_text(demo))
    print(lemmatization_stats([demo]))
