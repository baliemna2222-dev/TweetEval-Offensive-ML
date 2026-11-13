"""
src/model_utils.py
==================
Small helpers shared by ALL the model files (logistic_regression.py, svm.py, ...).
Keeping them here means every model is tuned and scored with exactly the same procedure,
which is what makes the comparison between models fair.
"""
import numpy as np                                   # arrays / random generator
import pandas as pd                                  # tuning tables
from sklearn.model_selection import ParameterGrid    # iterate over all hyperparameter combinations

from src import config                               # shared random seed
from src.evaluation import compute_metrics           # accuracy / precision / recall / F1 (offensive = 1)
from src.balancing import apply_balancing            # none | undersampling | oversampling | class_weight

SEED = config.RANDOM_STATE                           # the group rule: ONE seed everywhere


def tune_grid(make_model, grid, X_train, y_train, X_val, y_val):
    """Grid search on the VALIDATION set (the test set is never touched).

    Parameters
    ----------
    make_model : callable(**params) -> unfitted sklearn-like estimator
    grid       : dict {param_name: [values]} e.g. {"C": [0.1, 1, 10]}
    Returns
    -------
    best_model  : the fitted estimator with the best validation F1 (offensive class)
    best_params : dict with the winning hyperparameters
    table       : DataFrame (one row per combination) sorted by validation F1, best first
    """
    rows, best = [], (-1.0, None, None)                      # (best F1 so far, model, params)
    for params in ParameterGrid(grid):                       # every combination of the grid
        model = make_model(**params).fit(X_train, y_train)   # train on TRAIN only
        f1 = compute_metrics(y_val, model.predict(X_val))["f1"]  # score on VAL (offensive class)
        rows.append({**params, "F1 val (offensive)": round(f1, 4)})
        if f1 > best[0]:                                     # strict ">" : ties keep the FIRST (simplest) value
            best = (f1, model, params)
    table = (pd.DataFrame(rows).sort_values("F1 val (offensive)", ascending=False)
             .reset_index(drop=True))
    return best[1], best[2], table


def oversample(X, y, seed=SEED):
    """Random oversampling of the minority class (TRAINING rows only).

    Offensive tweets are duplicated at random until both classes have the same size.
    Used for models without a `class_weight` option (KNN).
    """
    rng = np.random.default_rng(seed)                        # reproducible
    idx_min = np.where(y == 1)[0]                            # offensive tweets (minority)
    idx_maj = np.where(y == 0)[0]                            # non-offensive tweets (majority)
    extra = rng.choice(idx_min, size=len(idx_maj) - len(idx_min), replace=True)  # duplicates to add
    keep = np.concatenate([np.arange(len(y)), extra])        # all original rows + the duplicates
    return X[keep], y[keep]


def prepare_training(X, y, balance, supports_class_weight):
    """Shared entry point of every model's train(): returns (X, y, class_weight_or_None).

    balance : None / False / "none" -> data untouched (e.g. learned priors for Naive Bayes);
              True / "class_weight" / "oversampling" / "undersampling" -> delegated to apply_balancing.
    """
    if balance is None or balance is False or balance == "none":
        return X, y, None
    if balance is True:
        balance = "class_weight"
    return apply_balancing(X, y, balance, supports_class_weight)


def get_scores(model, X):
    """Continuous 'offensive' score used for PR curves / average precision.

    Returns P(offensive) if the model has predict_proba, else the signed distance
    (decision_function), else None (e.g. a hard-voting ensemble).
    """
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def top_linear_terms(model, feature_names, n=12):
    """Strongest terms of a LINEAR model (coef_ > 0 pushes towards 'offensive')."""
    w = model.coef_.ravel()                                  # one weight per TF-IDF feature
    order = np.argsort(w)                                    # ascending: most negative first
    top = order[::-1][:n]                                    # n largest weights  -> offensive
    bottom = order[:n]                                       # n smallest weights -> non-offensive
    off = pd.DataFrame({"term": np.asarray(feature_names)[top], "weight": w[top]})
    non = pd.DataFrame({"term": np.asarray(feature_names)[bottom], "weight": w[bottom]})
    return off, non