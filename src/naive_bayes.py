from sklearn.naive_bayes import MultinomialNB

from src.model_utils import tune_grid, prepare_training

NAME = "Naive Bayes"              # display name: also the key of MODEL_COLORS / PRED in notebook 05
SUPPORTS_CLASS_WEIGHT = True      # no class_weight option, but EQUAL PRIORS (fit_prior=False) play the same role
DEFAULT_ALPHA_GRID = [0.03, 0.1, 0.3, 1.0]


def build_model(**kwargs):
    """Return an untrained MultinomialNB."""
    return MultinomialNB(**kwargs)


def train(X_train_tfidf, y_train, balance="class_weight", **kwargs):
    """Train on the training rows.
    balance: "class_weight" (default) -> equal priors | none/False -> priors learned from the data |
             undersampling / oversampling -> resampled training rows."""
    if balance is True or balance == "class_weight":
        kwargs["fit_prior"] = False                       # equal priors = Naive Bayes version of class_weight
        X, y = X_train_tfidf, y_train
    else:
        X, y, _ = prepare_training(X_train_tfidf, y_train, balance, False)
    return build_model(**kwargs).fit(X, y)


def tune(X_train, y_train, X_val, y_val, alpha_grid=None):
    """Pick alpha (equal priors) on validation F1. Returns (model, best_alpha, table)."""
    make = lambda alpha: build_model(alpha=alpha, fit_prior=False)
    model, p, table = tune_grid(make, {"alpha": alpha_grid or DEFAULT_ALPHA_GRID}, X_train, y_train, X_val, y_val)
    return model, p["alpha"], table