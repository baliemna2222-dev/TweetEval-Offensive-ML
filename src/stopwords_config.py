import nltk
try:
    nltk.data.find("corpora/stopwords")
except LookupError:                         # download the corpus automatically the first time
    nltk.download("stopwords", quiet=True)

from nltk.corpus import stopwords          # once: nltk.download("stopwords")

# Words that carry meaning for offensive-language detection and must NOT be removed.
NEGATIONS = {
    "no", "not", "nor", "never", "nothing", "nobody", "none", "neither", "cannot",
    "can't", "don't", "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "won't", "wouldn't", "couldn't", "shouldn't", "hasn't", "haven't", "hadn't",
}
# "you are an idiot": the target of an insult matters -> optional to keep.
PRONOUNS = {"i", "me", "my", "you", "your", "yours", "yourself", "he", "she", "they",
            "them", "we", "us"}


def build_stopwords(keep_pronouns=False, drop_user=False):
    """Standard NLTK list minus negations (and optionally minus pronouns)."""
    sw = set(stopwords.words("english")) - NEGATIONS
    if keep_pronouns:
        sw -= PRONOUNS
    if drop_user:
        sw.add("user")                     # placeholder present in ~93 % of tweets
    return sw


DEFAULT_STOPWORDS = build_stopwords()


def remove_stopwords(text, stop_words=DEFAULT_STOPWORDS):
    return " ".join(w for w in text.split() if w not in stop_words)
