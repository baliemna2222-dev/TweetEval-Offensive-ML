# TweetEval Offensive — ML project

Binary classification of tweets. Verify the official label mapping before interpreting results (the starter code assumes 0=non-offensive and 1=offensive).

## Install
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset
Put these six files in `data/raw/`:
- `train_text.txt`, `train_labels.txt`
- `val_text.txt`, `val_labels.txt`
- `test_text.txt`, `test_labels.txt`

Each text file has one tweet per line; its label file has one integer label per line in exactly the same order. See `data/raw/README.md`. If your files have different names or format, adapt `src/data_loader.py`.

## Run
From the repository root:
```bash
python -m src.pipeline
pytest -q
```
The pipeline trains Naive Bayes, Logistic Regression, Linear SVM and Random Forest; selects by validation F1; then evaluates the selected model once on test. It writes metrics to `results/metrics/`, a confusion matrix to `results/figures/`, and `models/best_model.joblib`.

## Structure
- `data/`: original and processed data
- `notebooks/`: exploration, preprocessing, features, models, evaluation, final pipeline
- `src/`: reusable Python code
- `tests/`: unit tests
- `docs/`: methodology, experiment log, report template
- `results/`: generated metrics and figures
- `models/`: saved final model

## Team of four
1. Dataset and EDA: `data_loader.py`, notebook 01
2. Preprocessing and TF-IDF: `preprocessing.py`, `features.py`, notebooks 02–03
3. Models: `models.py`, notebook 04
4. Evaluation: `evaluation.py`, `visualization.py`, notebook 05

Use feature branches and Pull Requests; do not commit directly to `main`. Keep train/validation/test identical across members. Never fit TF-IDF on validation/test. Do not select a model using test results.
