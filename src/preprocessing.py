"""Basic tweet cleaning for the TweetEval Offensive project (Emna's part).

Design rules (justified in notebooks/02_text_cleaning.ipynb):
  * lowercase, decode HTML entities (&amp; -> &)
  * URLs removed, @mentions replaced by the token "user"
  * hashtags: keep the word, drop the '#'
  * punctuation / emojis removed, BUT apostrophes inside words (don't, can't)
    and masked-profanity asterisks (f***, sh*t, f**k) are kept; asterisks used as
    emphasis (*inhales*) are removed
  * stopwords are NOT removed and NO lemmatisation here (Youssef's experiments)
  * labels and class proportions are never touched
"""
import html
import re

import pandas as pd

MENTION_TOKEN = "user"

URL_RE = re.compile(r"https?://\S+|www\.\S+")
MENTION_RE = re.compile(r"@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
DISALLOWED_RE = re.compile(r"[^a-z0-9'*\s]")  # everything except letters, digits, ' and *
LEADING_STAR_RE = re.compile(r"^\*+")          # *word -> word
LONE_TRAILING_STAR_RE = re.compile(r"(?<!\*)\*$")  # word* -> word  (but f*** is kept)


def clean_text(text):
    """Return the cleaned version of one tweet (empty string for missing values)."""
    if pd.isna(text):
        return ""

    text = html.unescape(str(text))                    # &amp; -> &, &lt; -> <
    text = text.lower()
    text = text.replace("\u2019", "'").replace("\u2018", "'")   # curly -> straight apostrophe
    text = text.replace("<3", " ")                     # heart emoticon would leave a stray "3"
    text = URL_RE.sub(" ", text)
    text = MENTION_RE.sub(f" {MENTION_TOKEN} ", text)
    text = HASHTAG_RE.sub(r" \1 ", text)               # keep the hashtag word
    text = DISALLOWED_RE.sub(" ", text)                # punctuation, emojis, symbols

    tokens = []
    for tok in text.split():
        tok = tok.strip("'")                           # quotes around a word: 'hello' -> hello
        tok = LEADING_STAR_RE.sub("", tok)             # emphasis markers: *inhales* -> inhales
        tok = LONE_TRAILING_STAR_RE.sub("", tok)
        if any(c.isalnum() for c in tok):              # drop isolated "***" or "'"
            tokens.append(tok)
    return " ".join(tokens)


def clean_series(series):
    return series.apply(clean_text)


def clean_dataframe(df, text_column="text"):
    """Return a copy of df with an extra `text_clean` column (other columns untouched)."""
    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in DataFrame.")
    result = df.copy()
    result["text_clean"] = clean_series(result[text_column])
    return result
