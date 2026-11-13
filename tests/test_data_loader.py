import pytest
from src.data_loader import load_split


def test_load_split(tmp_path):
    (tmp_path / "train_text.txt").write_text("hi\nyou idiot\n", encoding="utf-8")
    (tmp_path / "train_labels.txt").write_text("0\n1\n", encoding="utf-8")
    df = load_split("train", tmp_path)
    assert list(df.columns) == ["text", "label"]
    assert df["label"].tolist() == [0, 1]


def test_mismatch_raises(tmp_path):
    (tmp_path / "val_text.txt").write_text("a\nb\n", encoding="utf-8")
    (tmp_path / "val_labels.txt").write_text("0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_split("val", tmp_path)
