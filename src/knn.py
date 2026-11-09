import pandas as pd
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

DEFAULT_K_GRID = (1, 3, 5, 7, 11, 21)                # numbers of neighbours tried during tuning
DEFAULT_WEIGHTS_GRID = ("uniform", "distance")    


def build_knn(n_neighbors=5, weights="uniform"):
    return KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights,
                                metric="cosine", algorithm="brute")


def train_knn(X_train, y_train, n_neighbors=5, weights="uniform"):
    model = build_knn(n_neighbors=n_neighbors, weights=weights)
    model.fit(X_train, y_train)
    return model


def tune_knn(X_train, y_train, X_val, y_val, k_grid=DEFAULT_K_GRID,
            weights_grid=DEFAULT_WEIGHTS_GRID):
    records, best_f1, best_model, best_params = [], -1.0, None, None
    for k in k_grid:
        for w in weights_grid:
            model = train_knn(X_train, y_train, n_neighbors=k, weights=w)
            f1 = f1_score(y_val, model.predict(X_val), pos_label=1)
            records.append({"n_neighbors": k, "weights": w, "F1 val (offensive)": round(f1, 4)})
            if f1 > best_f1:
                best_f1, best_model = f1, model
                best_params = {"n_neighbors": k, "weights": w}
    tuning_df = (pd.DataFrame(records)
                .sort_values("F1 val (offensive)", ascending=False)
                .reset_index(drop=True))
    return best_model, best_params, tuning_df