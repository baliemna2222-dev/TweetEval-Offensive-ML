"""Train / evaluate the five classifiers on the same features (used by notebooks 06 and 08)."""
import time

import pandas as pd

from src import knn, logistic_regression, naive_bayes, random_forest, svm
from src.evaluation import compute_metrics

MODELS = {
    "Logistic Regression": logistic_regression,
    "SVM (LinearSVC)": svm,
    "Random Forest": random_forest,
    "Naive Bayes": naive_bayes,
    "KNN (k=5, cosine)": knn,
}


def effective_balancing(module, balancing):
    """'class_weight' falls back to oversampling for models without a class_weight option."""
    if balancing == "class_weight" and not module.SUPPORTS_CLASS_WEIGHT:
        return "oversampling"
    return balancing


def train_and_evaluate(X_train, y_train, X_eval, y_eval, balancings=("none",), models=None, verbose=True):
    """Train every model with every balancing strategy (TRAIN rows only) and score on (X_eval, y_eval).

    Returns (table, predictions) where predictions maps 'model | balancing' -> predicted labels."""
    models = models or MODELS
    rows, preds = [], {}
    for name, module in models.items():
        for bal in balancings:
            t0 = time.time()
            model = module.train(X_train, y_train, balance=bal)
            fit_seconds = time.time() - t0
            pred = model.predict(X_eval)
            preds[f"{name} | {bal}"] = pred
            rows.append({"model": name, "balancing": bal,
                         "effective_balancing": effective_balancing(module, bal),
                         "train_seconds": round(fit_seconds, 1), **compute_metrics(y_eval, pred)})
            if verbose:
                print(f"{name:22s} | {bal:13s} | F1 = {rows[-1]['f1']:.4f} | {fit_seconds:5.1f}s")
    return pd.DataFrame(rows), preds


def train_one(name, X_train, y_train, balancing="none", models=None):
    """Train a single model by its display name and return the fitted estimator."""
    return (models or MODELS)[name].train(X_train, y_train, balance=balancing)
