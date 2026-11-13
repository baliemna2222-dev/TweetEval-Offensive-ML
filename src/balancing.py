"""Class-balancing strategies. They are applied to the TRAINING rows only."""
import numpy as np

from src.config import RANDOM_STATE

STRATEGIES = ("none", "undersampling", "oversampling", "class_weight")


def resample_indices(y, strategy, random_state=RANDOM_STATE):
    """Row indices of the training set after under- or over-sampling."""
    y = np.asarray(y)
    rng = np.random.RandomState(random_state)
    idx_non, idx_off = np.where(y == 0)[0], np.where(y == 1)[0]
    if strategy == "undersampling":
        idx = np.concatenate([idx_off, rng.choice(idx_non, len(idx_off), replace=False)])
    elif strategy == "oversampling":
        idx = np.concatenate([idx_non, rng.choice(idx_off, len(idx_non), replace=True)])
    else:
        raise ValueError(f"No resampling for strategy '{strategy}'")
    return rng.permutation(idx)


def apply_balancing(X, y, strategy=None, supports_class_weight=True, random_state=RANDOM_STATE):
    """Return (X, y, class_weight) for the chosen strategy.

    Models without a class_weight option (Naive Bayes, KNN) fall back to over-sampling
    when the strategy is 'class_weight'."""
    strategy = "none" if strategy in (None, "original") else strategy
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{strategy}', choose from {STRATEGIES}")
    y = np.asarray(y)
    if strategy == "class_weight":
        if supports_class_weight:
            return X, y, "balanced"
        strategy = "oversampling"
    if strategy in ("undersampling", "oversampling"):
        idx = resample_indices(y, strategy, random_state)
        return X[idx], y[idx], None
    return X, y, None
