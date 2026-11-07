from src.evaluation import calculate_metrics, get_confusion_matrix

def test_perfect_metrics():
    result = calculate_metrics([0, 1], [0, 1])
    assert result["accuracy"] == 1.0 and result["f1"] == 1.0

def test_confusion_matrix_is_two_by_two():
    assert get_confusion_matrix([0, 1], [1, 1]).shape == (2, 2)
