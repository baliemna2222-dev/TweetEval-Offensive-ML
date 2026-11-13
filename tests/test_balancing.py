import numpy as np
import pytest
from scipy.sparse import csr_matrix

from src.balancing import apply_balancing

y = np.array([0] * 8 + [1] * 2)
X = csr_matrix(np.arange(10).reshape(-1, 1))


def test_none_keeps_data():
    X2, y2, cw = apply_balancing(X, y, "none")
    assert X2.shape[0] == 10 and cw is None


def test_undersampling_balances():
    _, y2, _ = apply_balancing(X, y, "undersampling")
    assert (y2 == 1).sum() == (y2 == 0).sum() == 2


def test_oversampling_balances():
    X2, y2, _ = apply_balancing(X, y, "oversampling")
    assert (y2 == 1).sum() == (y2 == 0).sum() == 8 and X2.shape[0] == 16


def test_class_weight_and_fallback():
    assert apply_balancing(X, y, "class_weight")[2] == "balanced"
    _, y2, cw = apply_balancing(X, y, "class_weight", supports_class_weight=False)
    assert cw is None and (y2 == 1).sum() == (y2 == 0).sum()


def test_unknown_strategy():
    with pytest.raises(ValueError):
        apply_balancing(X, y, "magic")
