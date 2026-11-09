from src.knn import build_knn
from src.logistic_regression import build_logistic_regression
from src.naive_bayes import build_naive_bayes
from src.random_forest import build_random_forest
from src.svm import build_svm


def get_models(random_state=42):
    return {
        "Naive Bayes": build_naive_bayes(),
        "Logistic Regression": build_logistic_regression(random_state=random_state),
        "Linear SVM": build_svm(random_state=random_state),
        "Random Forest": build_random_forest(n_estimators=300, random_state=random_state),
        "KNN": build_knn(),
    }