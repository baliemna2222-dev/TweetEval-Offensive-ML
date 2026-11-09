import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

RANDOM_STATE = 42                                  
DEFAULT_N_ESTIMATORS = (100, 300)                  # number of trees tried during tuning
DEFAULT_MAX_DEPTHS = (None, 30)                    # None = unlimited depth


def build_random_forest(n_estimators=100, max_depth=None, random_state=RANDOM_STATE):
    return RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth,class_weight="balanced", n_jobs=-1,
                                random_state=random_state)


def train_random_forest(X_train, y_train, n_estimators=100, max_depth=None,
                        random_state=RANDOM_STATE):
    """Train a Random Forest on the training features and return it."""
    model = build_random_forest(n_estimators, max_depth, random_state)
    model.fit(X_train, y_train)
    return model


def tune_random_forest(X_train, y_train, X_val, y_val,n_estimators_grid=DEFAULT_N_ESTIMATORS,max_depth_grid=DEFAULT_MAX_DEPTHS,random_state=RANDOM_STATE):

    records, best_f1, best_model, best_params = [], -1.0, None, None
    for n_trees in n_estimators_grid:
        for depth in max_depth_grid:
            model = train_random_forest(X_train, y_train, n_trees, depth, random_state)
            f1 = f1_score(y_val, model.predict(X_val), pos_label=1)
            records.append({"n_estimators": n_trees, "max_depth": str(depth),
                            "F1 val (offensive)": round(f1, 4)})
            if f1 > best_f1:
                best_f1, best_model = f1, model
                best_params = {"n_estimators": n_trees, "max_depth": depth}
    tuning_df = (pd.DataFrame(records)
                .sort_values("F1 val (offensive)", ascending=False)
                .reset_index(drop=True))
    return best_model, best_params, tuning_df