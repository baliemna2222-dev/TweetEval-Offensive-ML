from src.features import fit_tfidf, transform_tfidf


def test_vocabulary_comes_from_train_only():
    vec, X = fit_tfidf(["you are an idiot", "have a nice day", "nice idiot"], min_df=1)
    assert "idiot" in vec.vocabulary_
    assert "zebra" not in vec.vocabulary_
    Xv = transform_tfidf(vec, ["zebra zebra"])       # unseen word -> all-zero row, vocabulary unchanged
    assert Xv.nnz == 0 and "zebra" not in vec.vocabulary_


def test_contractions_and_masked_words_survive_tokenizer():
    vec, _ = fit_tfidf(["i don't like f*** this"], min_df=1)
    assert "don't" in vec.vocabulary_ and "f***" in vec.vocabulary_
