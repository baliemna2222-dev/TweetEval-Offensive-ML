from pathlib import Path
import json
import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from src.data_loader import load_dataset
from src.preprocessing import clean_texts
from src.features import build_tfidf_vectorizer
from src.models import get_models
from src.evaluation import calculate_metrics, get_confusion_matrix
from src.visualization import plot_confusion_matrix

def make_pipeline(classifier):
    # Vectorizer is fitted only on training text by Pipeline.fit().
    return Pipeline([("tfidf", build_tfidf_vectorizer()), ("classifier", classifier)])

def run_experiments(data_dir="data/raw"):
    splits = load_dataset(data_dir)
    X_train = clean_texts(splits["train"]["text"].tolist())
    y_train = splits["train"]["label"]
    X_val = clean_texts(splits["validation"]["text"].tolist())
    y_val = splits["validation"]["label"]
    X_test = clean_texts(splits["test"]["text"].tolist())
    y_test = splits["test"]["label"]

    metric_dir, figure_dir, model_dir = Path("results/metrics"), Path("results/figures"), Path("models")
    for folder in (metric_dir, figure_dir, model_dir):
        folder.mkdir(parents=True, exist_ok=True)

    rows, trained = [], {}
    for name, classifier in get_models().items():
        print(f"Training {name}...")
        model = make_pipeline(classifier).fit(X_train, y_train)
        rows.append({"model": name, **calculate_metrics(y_val, model.predict(X_val))})
        trained[name] = model

    comparison = pd.DataFrame(rows).sort_values("f1", ascending=False)
    comparison.to_csv(metric_dir / "validation_model_comparison.csv", index=False)
    best_name = comparison.iloc[0]["model"]
    best_model = trained[best_name]

    # Test is used only once, after model selection on validation.
    test_pred = best_model.predict(X_test)
    test_metrics = calculate_metrics(y_test, test_pred)
    test_metrics["selected_model"] = best_name
    (metric_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2), encoding="utf-8")
    plot_confusion_matrix(get_confusion_matrix(y_test, test_pred),
                          figure_dir / "confusion_matrix_test.png",
                          title=f"Test — {best_name}")
    joblib.dump(best_model, model_dir / "best_model.joblib")
    print(comparison.to_string(index=False))
    print("Test:", test_metrics)
    return comparison, test_metrics, best_model

if __name__ == "__main__":
    run_experiments()
