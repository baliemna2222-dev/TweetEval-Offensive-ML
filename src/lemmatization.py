import nltk
try:
    nltk.data.find("corpora/wordnet")
except LookupError:                         # download the corpus automatically the first time
    nltk.download("wordnet", quiet=True)

from nltk.stem import WordNetLemmatizer    # once: nltk.download("wordnet")

_lem = WordNetLemmatizer()
PROTECTED = {"user", "us"}                 # "user" = placeholder created by Emna; "us" would become "u"


def lemmatize_word(word):
    # keep placeholder, masked profanity (f***), contractions (don't) and numbers untouched.
    # Words ending in "ss" are protected too: WordNet would turn "ass" into "as" (an offensive
    # word becoming a stopword-like token); plurals such as "asses" are not affected by this rule.
    if word in PROTECTED or word.endswith("ss") or "*" in word or "'" in word or word.isdigit():
        return word
    # verb first (was -> be, hating -> hate), then noun (idiots -> idiot).
    # Without pos=, NLTK treats everything as a noun and turns "was" into "wa".
    lemma = _lem.lemmatize(word, pos="v")
    if lemma.endswith("ss"):               # "asses" -> "ass": stop here, the noun pass would give "as"
        return lemma
    return _lem.lemmatize(lemma, pos="n")


def lemmatize_text(text):
    return " ".join(lemmatize_word(w) for w in text.split())
