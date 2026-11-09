from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
LABEL_NAMES = {0: "non-offensive", 1: "offensive"}


def _read_lines(path):
    with open(path, encoding="utf-8") as f:
        return [line.rstrip("\n") for line in f]


def load_split(split, raw_dir=RAW_DIR):
    """Load one official split ('train', 'val' or 'test') as a DataFrame(text, label)."""
    raw_dir = Path(raw_dir)
    texts = _read_lines(raw_dir / f"{split}_text.txt")
    labels = [int(x) for x in _read_lines(raw_dir / f"{split}_labels.txt")]
    if len(texts) != len(labels):
        raise ValueError(f"{split}: {len(texts)} texts but {len(labels)} labels")
    return pd.DataFrame({"text": texts, "label": labels})


def load_all(raw_dir=RAW_DIR):
    """Return (train_df, val_df, test_df)."""
    return tuple(load_split(s, raw_dir) for s in ("train", "val", "test"))