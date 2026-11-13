import numpy as np

from src.features import fit_tfidf, transform_tfidf
from src.modeling import MODELS, train_and_evaluate, train_one

TEXTS = ["you are an idiot", "what a stupid idiot", "shut up stupid", "idiot idiot", "stupid fool",
         "have a nice day", "lovely weather today", "thank you friend", "nice work team",
         "great game tonight", "happy birthday friend", "good morning all"] * 4
LABELS = np.array([1] * 5 + [0] * 7 + [1] * 5 + [0] * 7 + [1] * 5 + [0] * 7 + [1] * 5 + [0] * 7)[:len(TEXTS)]


def test_all_five_models_train_and_predict():
    vec, X = fit_tfidf(TEXTS, min_df=1)
    table, preds = train_and_evaluate(X, LABELS, transform_tfidf(vec, TEXTS), LABELS,
                                      balancings=("none", "class_weight"), verbose=False)
    assert set(table["model"]) == set(MODELS) and len(table) == 10
    assert table["f1"].between(0, 1).all()
    # models without class_weight fall back to oversampling
    nb = table[(table.model == "Naive Bayes") & (table.balancing == "class_weight")].iloc[0]
    assert nb["effective_balancing"] == "oversampling"


def test_train_one_returns_fitted_model():
    vec, X = fit_tfidf(TEXTS, min_df=1)
    model = train_one("Logistic Regression", X, LABELS, "class_weight")
    assert model.predict(transform_tfidf(vec, ["stupid idiot"])).shape == (1,)
