"""
src/extra_models.py  --  OPTIONAL classical models (scikit-learn only, no extra install)
=======================================================================================
Three cheap extras that give the comparison more depth:

1. Complement Naive Bayes - a NB variant designed for IMBALANCED text (it estimates word
   statistics from the COMPLEMENT of each class), so it needs no prior trick.
2. SGD classifier (modified Huber loss) - a linear model trained by stochastic gradient
   descent: very fast on large sparse data, behaves like a linear SVM but gives probabilities.
3. Voting ensemble - combines Logistic Regression + LinearSVC + Naive Bayes by MAJORITY vote
   (hard voting; LinearSVC has no probabilities, so soft voting is not possible).
"""
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import VotingClassifier
from src.model_utils import SEED, tune_grid

NAME_CNB, NAME_SGD, NAME_VOTE = "Complement NB", "SGD (modified Huber)", "Voting ensemble"
CNB_ALPHA_GRID = [0.03, 0.1, 0.3, 1.0]
SGD_ALPHA_GRID = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]      # alpha ~ 1/C : larger = stronger regularisation


def tune_complement_nb(X_train, y_train, X_val, y_val, alpha_grid=None):
    """Pick alpha for Complement NB. Returns (model, best_alpha, table)."""
    model, p, table = tune_grid(lambda alpha: ComplementNB(alpha=alpha),
                                {"alpha": alpha_grid or CNB_ALPHA_GRID}, X_train, y_train, X_val, y_val)
    return model, p["alpha"], table


def tune_sgd(X_train, y_train, X_val, y_val, alpha_grid=None):
    """Pick alpha for the SGD classifier. Returns (model, best_alpha, table)."""
    make = lambda alpha: SGDClassifier(loss="modified_huber", alpha=alpha, class_weight="balanced",
                                       max_iter=2000, tol=1e-4, random_state=SEED)
    model, p, table = tune_grid(make, {"alpha": alpha_grid or SGD_ALPHA_GRID}, X_train, y_train, X_val, y_val)
    return model, p["alpha"], table


def train_voting(X_train, y_train, C_lr=1.0, C_svm=0.1, alpha_nb=0.3):
    """Hard-voting ensemble of LogReg + LinearSVC + Naive Bayes (use each model's tuned value)."""
    vote = VotingClassifier(estimators=[
        ("lr", LogisticRegression(C=C_lr, class_weight="balanced", max_iter=2000, random_state=SEED)),
        ("svm", LinearSVC(C=C_svm, class_weight="balanced", max_iter=5000, random_state=SEED)),
        ("nb", MultinomialNB(alpha=alpha_nb, fit_prior=False)),
    ], voting="hard")                                  # each model gets one vote, majority wins
    return vote.fit(X_train, y_train)


def train_complement_nb(X, y, alpha=0.3):
    """Fit ONE Complement NB with a given alpha (used by notebook 06 to retrain the tuned model)."""
    return ComplementNB(alpha=alpha).fit(X, y)


def train_sgd(X, y, alpha=1e-4):
    """Fit ONE SGD classifier (modified Huber, balanced classes) with a given alpha."""
    return SGDClassifier(loss="modified_huber", alpha=alpha, class_weight="balanced",
                         max_iter=2000, tol=1e-4, random_state=SEED).fit(X, y)
