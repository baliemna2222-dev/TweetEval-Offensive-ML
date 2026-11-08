from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

def get_models(random_state=42):
    return {
        "Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=random_state),
        "Linear SVM": LinearSVC(class_weight="balanced", random_state=random_state),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                                 random_state=random_state, n_jobs=-1),
    }
