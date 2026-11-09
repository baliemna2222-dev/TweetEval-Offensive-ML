import pandas as pd
from sklearn.metrics import f1_score
from sklearn.naive_bayes import MultinomialNB

DEFAULT_ALPHA_GRID = (0.03, 0.1, 0.3, 1.0)         # Laplace/Lidstone smoothing values tried during tuning


def build_naive_bayes(alpha=1.0, fit_prior=False):
    return MultinomialNB(alpha=alpha, fit_prior=fit_prior)


def train_naive_bayes(X_train, y_train, alpha=1.0, fit_prior=False):
    """Train a Multinomial Naive Bayes on the training features and return it."""
    model = build_naive_bayes(alpha=alpha, fit_prior=fit_prior)
    model.fit(X_train, y_train)
    return model


def tune_naive_bayes(X_train, y_train, X_val, y_val, alpha_grid=DEFAULT_ALPHA_GRID,fit_prior=False):

    records, best_f1, best_model, best_alpha = [], -1.0, None, None
    for alpha in alpha_grid:
        model = train_naive_bayes(X_train, y_train, alpha=alpha, fit_prior=fit_prior)
        f1 = f1_score(y_val, model.predict(X_val), pos_label=1)
        records.append({"alpha": alpha, "F1 val (offensive)": round(f1, 4)})
        if f1 > best_f1:
            best_f1, best_model, best_alpha = f1, model, alpha
    return best_model, best_alpha, pd.DataFrame(records)