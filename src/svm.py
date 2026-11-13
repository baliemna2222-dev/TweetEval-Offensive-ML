from sklearn.svm import LinearSVC

from src.model_utils import SEED, tune_grid, prepare_training

NAME = "SVM (LinearSVC)"          # display name: also the key of MODEL_COLORS / PRED in notebook 05
SUPPORTS_CLASS_WEIGHT = True      # False -> 'class_weight' falls back to over-sampling
DEFAULT_C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3]


def build_model(**kwargs):
    """Return an untrained LinearSVC (kwargs override the defaults)."""
    params = {"max_iter": 5000, "random_state": SEED, **kwargs}
    return LinearSVC(**params)


def train(X_train_tfidf, y_train, balance="class_weight", **kwargs):
    """Train on the training rows. balance: none | undersampling | oversampling | class_weight."""
    X, y, cw = prepare_training(X_train_tfidf, y_train, balance, SUPPORTS_CLASS_WEIGHT)
    if cw is not None:
        kwargs["class_weight"] = cw
    return build_model(**kwargs).fit(X, y)


def tune(X_train, y_train, X_val, y_val, c_grid=None):
    """Pick C on the validation F1 (offensive). Returns (model, best_C, table)."""
    make = lambda C: build_model(C=C, class_weight="balanced")
    model, p, table = tune_grid(make, {"C": c_grid or DEFAULT_C_GRID}, X_train, y_train, X_val, y_val)
    return model, p["C"], table