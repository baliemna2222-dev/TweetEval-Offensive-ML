"""Tests for Youssef's preprocessing modules (stopwords + lemmatisation).

Run with:  pytest -q tests/test_stopwords_and_lemmatization.py
(the NLTK corpora stopwords/wordnet must be downloaded once - see the conftest helper
at the bottom of this file, which does it automatically)
"""

import pytest

# --------------------------------------------------------------------------- #
# make the run self-sufficient: download the two corpora on first use
# --------------------------------------------------------------------------- #
import nltk

for _package in ("stopwords", "wordnet", "omw-1.4"):
    try:
        nltk.data.find(f"corpora/{_package}")
    except LookupError:  # pragma: no cover - first run only
        nltk.download(_package, quiet=True)

from src.stopwords_config import (  # noqa: E402
    NEGATIONS,
    PRONOUNS,
    build_stopwords,
    missing_protected_tokens,
    remove_stopwords,
    stopword_stats,
)
from src.lemmatization import (  # noqa: E402
    is_protected,
    lemmatization_stats,
    lemmatize_text,
    lemmatize_word,
)


# --------------------------------------------------------------------------- #
# stopwords_config
# --------------------------------------------------------------------------- #
def test_no_negation_is_ever_a_stopword():
    """The whole point of the module: negations must never be removed."""
    for keep_pronouns in (False, True):
        for drop_user in (False, True):
            stop_words = build_stopwords(keep_pronouns=keep_pronouns, drop_user=drop_user)
            assert missing_protected_tokens(stop_words) == []


def test_negation_survives_in_a_sentence():
    assert "not" in remove_stopwords("you are not an idiot").split()
    assert "don't" in remove_stopwords("i don't like you").split()
    assert "never" in remove_stopwords("this is never acceptable").split()


def test_pronouns_are_kept_only_when_asked():
    text = "you are an idiot"
    assert "you" in remove_stopwords(text, build_stopwords(keep_pronouns=True)).split()
    assert "you" not in remove_stopwords(text, build_stopwords(keep_pronouns=False)).split()


def test_user_placeholder_behaviour():
    text = "user user this is stupid"
    assert "user" in remove_stopwords(text).split()                        # kept by default
    assert "user" not in remove_stopwords(text, build_stopwords(drop_user=True)).split()


def test_a_tweet_can_become_empty_but_never_nan():
    result = remove_stopwords("user user user", build_stopwords(drop_user=True))
    assert result == ""


def test_stopword_list_is_smaller_than_nltk_after_protection():
    assert len(NEGATIONS) > 40
    assert "user" not in build_stopwords()                       # placeholder not removed by default
    assert build_stopwords() != build_stopwords(keep_pronouns=True)


def test_extra_keep_always_wins():
    stop_words = build_stopwords(extra_keep=["offensive", "hate"])
    assert "offensive" not in stop_words and "hate" not in stop_words
    assert "offensive" in remove_stopwords("this is offensive", stop_words).split()


def test_stopword_stats_reports_empties_and_lengths():
    stats = stopword_stats(["user user", "the cat is on the table"])
    assert stats["n_tweets"] == 2
    assert stats["avg_words_before"] > stats["avg_words_after"]
    assert 0 <= stats["n_empty_after"] <= 2


# --------------------------------------------------------------------------- #
# lemmatization
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("word,expected", [
    ("was", "be"),        # verb pass is mandatory: without it NLTK gives "wa"
    ("idiots", "idiot"),  # noun pass
    ("fucking", "fuck"),
    ("bitches", "bitch"),
])
def test_lemmatize_known_cases(word, expected):
    assert lemmatize_word(word) == expected


@pytest.mark.parametrize("token", ["user", "f***", "s***", "don't", "isn't", "2020", "maga2020"])
def test_protected_tokens_are_untouched(token):
    assert is_protected(token)
    assert lemmatize_word(token) == token


def test_lemmatize_text_keeps_negations_and_placeholder():
    text = "user he was hating the idiots , don't you think ?"
    out = lemmatize_text(text)
    assert "'" in out and "don't" in out.split()
    assert "user" in out.split()
    assert "was" not in out.split()


def test_lemmatize_empty_string():
    assert lemmatize_text("") == ""


def test_lemmatization_stats_shrink_the_vocabulary():
    stats = lemmatization_stats(["fucking fucked fucks", "idiots idiot"])
    assert stats["vocab_after"] < stats["vocab_before"]
    assert stats["n_tokens_changed"] > 0


# --------------------------------------------------------------------------- #
# the retained pipeline (what notebook 03 hands over to notebook 04)
# --------------------------------------------------------------------------- #
def test_retained_pipeline_end_to_end():
    """clean (Emna) -> stopwords (negations protected) -> lemmatisation = variant C."""
    from src.preprocessing import clean_text

    raw = "@USER These idiots are NOT #winning, don't you get it ? f***"
    cleaned = clean_text(raw, remove_mentions=True)      # needs remove_mentions=True here
    tokens = lemmatize_text(remove_stopwords(cleaned, build_stopwords())).split()

    assert "user" in tokens          # placeholder kept (variant C does not drop it)
    assert "not" in tokens           # negation protected
    assert "don't" in tokens         # contraction protected
    assert "f***" in tokens          # masked profanity protected
    assert "idiot" in tokens         # lemmatised (idiots -> idiot)
