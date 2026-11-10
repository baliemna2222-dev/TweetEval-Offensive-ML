from src.data_loader import load_split

def test_load_split(tmp_path):
    texts, labels = tmp_path / "texts.txt", tmp_path / "labels.txt"
    texts.write_text("hello\nbad tweet\n", encoding="utf-8")
    labels.write_text("0\n1\n", encoding="utf-8")
    df = load_split(texts, labels)
    assert df["text"].tolist() == ["hello", "bad tweet"]
    assert df["label"].tolist() == [0, 1]
