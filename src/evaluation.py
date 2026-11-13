import pandas as pd
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

LABELS = ["non-offensive", "offensive"]


def compute_metrics(y_true, y_pred):
    """Metrics of the positive (offensive) class + overall accuracy."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def compare_models(models, X, y):
    """models: dict name -> fitted model. Returns a table sorted by F1."""
    rows = {name: compute_metrics(y, m.predict(X)) for name, m in models.items()}
    return pd.DataFrame(rows).T.sort_values("f1", ascending=False)


def metrics_from_predictions(predictions, y_true):
    """predictions: dict name -> array of predicted labels. Returns a table sorted by F1."""
    rows = {name: compute_metrics(y_true, p) for name, p in predictions.items()}
    return pd.DataFrame(rows).T.sort_values("f1", ascending=False)


def get_confusion_matrix(y_true, y_pred):
    return confusion_matrix(y_true, y_pred, labels=[0, 1])


def report(y_true, y_pred):
    return classification_report(y_true, y_pred, target_names=LABELS, zero_division=0)
