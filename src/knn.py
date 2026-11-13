import numpy as np
from sklearn.neighbors import KNeighborsClassifier

from src.model_utils import tune_grid, oversample, prepare_training

NAME = "KNN"                      # display name: also the key of MODEL_COLORS / PRED in notebook 05
SUPPORTS_CLASS_WEIGHT = False     # False -> 'class_weight' falls back to over-sampling
DEFAULT_K_GRID = [1, 3, 5, 7, 11, 15, 21]
DEFAULT_GRID = {"n_neighbors": DEFAULT_K_GRID, "weights": ["uniform", "distance"]}


def build_model(**kwargs):
    """Untrained KNN with the cosine distance (brute force: the right choice for sparse TF-IDF)."""
    params = {"metric": "cosine", "algorithm": "brute", "n_jobs": -1, **kwargs}
    return KNeighborsClassifier(**params)


def train(X_train_tfidf, y_train, balance="oversampling", **kwargs):
    """Train on the training rows. Default = random over-sampling of the offensive tweets.
    balance: none/False -> untouched | class_weight/oversampling -> oversample | undersampling -> apply_balancing."""
    y_train = np.asarray(y_train)
    if balance in ("class_weight", "oversampling") or balance is True:
        X, y = oversample(X_train_tfidf, y_train)
    else:
        X, y, _ = prepare_training(X_train_tfidf, y_train, balance, SUPPORTS_CLASS_WEIGHT)
    return build_model(**kwargs).fit(X, y)


def tune(X_train, y_train, X_val, y_val, grid=None):
    """Grid search (k x vote) on validation F1; TRAIN is oversampled once, validation is untouched.
    Returns (model, best_params dict, table)."""
    Xo, yo = oversample(X_train, np.asarray(y_train))
    return tune_grid(lambda **p: build_model(**p), grid or DEFAULT_GRID, Xo, yo, X_val, y_val)