from src.features import build_tfidf_vectorizer

def test_tfidf_fit_transform():
    vectorizer = build_tfidf_vectorizer(min_df=1)
    matrix = vectorizer.fit_transform(["good tweet", "bad tweet"])
    assert matrix.shape[0] == 2 and matrix.shape[1] > 0
