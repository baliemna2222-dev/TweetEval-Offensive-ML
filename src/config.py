"""Single source of truth for the modelling choices.

The three DECISION values below come from Youssef's experiments
(notebooks 03_preprocessing_experiments and 04_class_balance_experiments).
If the experiments are re-run and the conclusions change, edit them HERE:
notebooks 05 -> 08 follow automatically.
"""
RANDOM_STATE = 42

# TF-IDF (fitted on the training set only). The token_pattern keeps "don't" and "f***"
# intact (the default pattern would split "don't" into "don" and drop "f***").
TFIDF_PARAMS = dict(max_features=20000, ngram_range=(1, 2), min_df=2,
                    token_pattern=r"[a-z0-9'*]+")

# ---- DECISIONS taken from notebooks 03 and 04 (validation set) ------------------------
USE_STOPWORDS = True          # remove stopwords (negations are always protected)
USE_LEMMATIZATION = True      # lemmatise after stopword removal
KEEP_PRONOUNS = False         # keep you/he/she... (variant B2)
DROP_USER = False             # also remove the "user" placeholder (variant B3)
BALANCING = "class_weight"    # none | undersampling | oversampling | class_weight
