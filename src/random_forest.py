from sklearn.ensemble import RandomForestClassifier

from src.model_utils import SEED, tune_grid, prepare_training

NAME = "Random Forest"            # display name: also the key of MODEL_COLORS / PRED in notebook 05
SUPPORTS_CLASS_WEIGHT = True      # False -> 'class_weight' falls back to over-sampling
DEFAULT_GRID = {"n_estimators": [100, 300], "max_depth": [30, None]}   # small grid: trees are slow on sparse text


def build_model(**kwargs):
    """Return an untrained RandomForestClassifier (kwargs override the defaults, e.g. n_estimators)."""
    params = {"n_estimators": 200, "n_jobs": -1, "random_state": SEED, **kwargs}
    return RandomForestClassifier(**params)


def train(X_train_tfidf, y_train, balance="class_weight", **kwargs):
    """Train on the training rows. balance: none | undersampling | oversampling | class_weight."""
    X, y, cw = prepare_training(X_train_tfidf, y_train, balance, SUPPORTS_CLASS_WEIGHT)
    if cw is not None:
        kwargs["class_weight"] = cw
    return build_model(**kwargs).fit(X, y)


def tune(X_train, y_train, X_val, y_val, grid=None):
    """Grid search (trees x depth) on validation F1. Returns (model, best_params dict, table)."""
    make = lambda **p: build_model(class_weight="balanced", **p)
    return tune_grid(make, grid or DEFAULT_GRID, X_train, y_train, X_val, y_val)