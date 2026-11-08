import pandas as pd
import pytest

from src.preprocessing import clean_dataframe, clean_text


def test_url_removed_and_mention_replaced():
    out = clean_text("Hello @Bob check https://t.co/xyz NOW")
    assert out == "hello user check now"


def test_hashtag_word_kept():
    assert clean_text("#Politics matter") == "politics matter"


def test_contractions_and_negations_kept():
    assert clean_text("I DON'T like this.") == "i don't like this"
    assert clean_text("I don\u2019t know") == "i don't know"


def test_masked_profanity_kept_but_isolated_symbols_dropped():
    assert clean_text("You are f*** annoying !!! ***") == "you are f*** annoying"


def test_emphasis_asterisks_removed_masks_kept():
    assert clean_text("sigh *INHALES* ok") == "sigh inhales ok"
    assert clean_text("what a sh*t f**k f***") == "what a sh*t f**k f***"


def test_html_entities_and_emojis():
    assert clean_text("cats &amp; dogs \U0001F602 <3") == "cats dogs"


def test_spaces_and_missing_values():
    assert clean_text(" A   B ") == "a b"
    assert clean_text(None) == ""
    assert clean_text(float("nan")) == ""


def test_dataframe_keeps_labels_and_rows():
    df = pd.DataFrame({"text": ["Hi @x", "BAD!"], "label": [0, 1]})
    out = clean_dataframe(df)
    assert out["label"].tolist() == [0, 1]
    assert list(out.columns) == ["text", "label", "text_clean"]
    assert "text_clean" not in df.columns          # input not modified


def test_missing_column_raises():
    with pytest.raises(ValueError):
        clean_dataframe(pd.DataFrame({"a": [1]}))