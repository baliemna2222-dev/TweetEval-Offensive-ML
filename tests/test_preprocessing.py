from src.preprocessing import clean_text

def test_clean_text_removes_url_and_lowercases():
    assert clean_text("HELLO https://example.com") == "hello URL".lower()

def test_clean_text_preserves_hashtags_and_emoji():
    result = clean_text("Great #day 😊")
    assert "#day" in result and "😊" in result
