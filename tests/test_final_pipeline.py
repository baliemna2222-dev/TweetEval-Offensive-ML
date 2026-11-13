from src.final_pipeline import VARIANTS, preprocess_text


def test_negations_survive_all_variants():
    for name, settings in VARIANTS.items():
        out = preprocess_text("I DON'T like this @bob, it is NOT good!", **settings)
        assert "don't" in out and "not" in out, name


def test_stopwords_removed_when_requested():
    assert preprocess_text("this is the best", stopwords=True, lemmatize=False) == "best"
    assert preprocess_text("this is the best", stopwords=False, lemmatize=False) == "this is the best"


def test_lemmatisation_and_protections():
    out = preprocess_text("these idiots were f*** @bob", stopwords=False, lemmatize=True)
    assert "idiot" in out and "f***" in out and "user" in out


def test_already_clean_flag():
    assert preprocess_text("hello user", already_clean=True, stopwords=False, lemmatize=False) == "hello user"


def test_lemmatiser_does_not_corrupt_offensive_words():
    out = preprocess_text("you ass these asses and us", stopwords=False, lemmatize=True)
    assert out == "you ass these ass and us"
