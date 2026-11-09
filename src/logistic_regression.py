import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

RANDOM_STATE = 42                                  
DEFAULT_C_GRID = (0.1, 0.3, 1, 3, 10)              # inverse regularisation strengths tried during tuning


def build_logistic_regression(C=1.0, random_state=RANDOM_STATE):
    return LogisticRegression(C=C, class_weight="balanced", max_iter=1000,random_state=random_state)


def train_logistic_regression(X_train, y_train, C=1.0, random_state=RANDOM_STATE):
    """Train a Logistic Regression on the training features and return it."""
    model = build_logistic_regression(C=C, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def tune_logistic_regression(X_train, y_train, X_val, y_val, c_grid=DEFAULT_C_GRID,random_state=RANDOM_STATE):
    records, best_f1, best_model, best_C = [], -1.0, None, None
    for C in c_grid:
        model = train_logistic_regression(X_train, y_train, C=C, random_state=random_state)
        f1 = f1_score(y_val, model.predict(X_val), pos_label=1)
        records.append({"C": C, "F1 val (offensive)": round(f1, 4)})
        if f1 > best_f1:
            best_f1, best_model, best_C = f1, model, C
    return best_model, best_C, pd.DataFrame(records)


def top_terms_logreg(model, feature_names, n=15):
    terms = np.asarray(feature_names)
    coefs = model.coef_[0]
    order = np.argsort(coefs)
    pos, neg = order[-n:][::-1], order[:n]
    offensive = pd.DataFrame({"term": terms[pos], "weight": coefs[pos]})
    non_offensive = pd.DataFrame({"term": terms[neg], "weight": coefs[neg]})
    return offensive, non_offensive