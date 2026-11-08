from pathlib import Path
import pandas as pd

def load_split(text_path, label_path):
    """Load one-text-per-line and one-integer-label-per-line files."""
    text_path, label_path = Path(text_path), Path(label_path)
    if not text_path.is_file() or not label_path.is_file():
        raise FileNotFoundError(f"Missing dataset file: {text_path} or {label_path}")
    texts = text_path.read_text(encoding="utf-8-sig").splitlines()
    raw_labels = label_path.read_text(encoding="utf-8-sig").splitlines()
    if len(texts) != len(raw_labels):
        raise ValueError(f"Text/label count mismatch: {len(texts)} != {len(raw_labels)}")
    try:
        labels = [int(x.strip()) for x in raw_labels]
    except ValueError as exc:
        raise ValueError("Labels must be integers.") from exc
    return pd.DataFrame({"text": texts, "label": labels})

def load_dataset(data_dir="data/raw"):
    """Load the official train, validation and test split filenames."""
    folder = Path(data_dir)
    names = {
        "train": ("train_text.txt", "train_labels.txt"),
        "validation": ("val_text.txt", "val_labels.txt"),
        "test": ("test_text.txt", "test_labels.txt"),
    }
    return {split: load_split(folder / a, folder / b) for split, (a, b) in names.items()}
