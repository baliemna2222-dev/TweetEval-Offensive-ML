Youssef — decision table (preprocessing & class balancing)
Hand-over document for Emna and for the report. Every number below comes from
results/metrics/03_preprocessing_experiments/ and results/metrics/04_class_balance_experiments/
and is reproduced automatically when the two notebooks are re-run (the decision cells build their
tables from the results, nothing is typed by hand).

1. The table requested by the guide (section 4, step 5)
Question	Decision	Evidence (validation set)
Remove stopwords (negations protected)?	Yes	F1 0.487 vs 0.478 (+0.9 pt) and 5-fold CV 0.476 vs 0.457 (+1.9 pt): same sign on both estimates; the token count of not is identical before/after removal in every variant, so no negation is lost
Keep pronouns / drop user?	No to both (pronouns dropped, user kept)	pronouns: validation +1.6 pt but CV −1.5 pt → the two estimates contradict each other, the effect is inside the noise (σ ≈ 1.2 pt); user: +1.2 pt validation / +1.0 pt CV but it empties 58 training and 6 validation tweets → rejected
Lemmatise?	Yes	F1 0.521 vs 0.487 (+3.3 pt) and CV 0.501 vs 0.476 (+2.6 pt); vocabulary 18,300 → 14,679 distinct tokens; the only large and consistent effect
Balancing strategy retained	class_weight="balanced"	F1 0.521 → 0.625, recall 0.388 → 0.625, offensive tweets caught 178 → 287 (out of 459) — no row deleted, no row duplicated, deterministic (F1 = 0.6246 on all 10 seeds)
Where balancing is applied	Training time only	it is never part of preprocess_text(); validation and test keep their natural class distribution (34.7 % / 27.9 % offensive)
Retained final text pipeline (variant C)
Python

# src/final_pipeline.py  — to be written together with Emna
from src.preprocessing import clean_text
from src.stopwords_config import build_stopwords, remove_stopwords
from src.lemmatization import lemmatize_text

_STOPWORDS = build_stopwords()          # NLTK's English list minus every negation


def preprocess_text(text, use_stopwords=True, use_lemmatization=True):
    text = clean_text(text)                     # Emna's cleaning
    if use_stopwords:
        text = remove_stopwords(text, _STOPWORDS)   # Youssef: negations protected
    if use_lemmatization:
        text = lemmatize_text(text)                 # Youssef: verb pass then noun pass
    return text
    # balancing is NOT here: LogisticRegression(class_weight="balanced") at training time
The same snippet is saved as
results/metrics/04_class_balance_experiments/final_pipeline_snippet.py.

2. Notebook 03 — the six text variants compared
Same splits, same TF-IDF (max_features=20000, ngram_range=(1,2), min_df=2,
token_pattern="[a-z0-9'*]+"), same model (LogisticRegression(max_iter=1000, random_state=42)),
same seed. Only the text transformation changes.

Variant	Text transformation	Accuracy	Precision	Recall	F1	5-fold CV F1
A	clean only (Emna) — reference	0.744	0.816	0.338	0.478	0.457
B	+ stopwords (negations protected)	0.746	0.808	0.349	0.487	0.476
B2	+ stopwords, pronouns kept	0.752	0.826	0.362	0.503	0.461
B3	+ stopwords, user placeholder dropped	0.749	0.806	0.362	0.499	0.486
C	+ stopwords + lemmatisation (kept)	0.752	0.791	0.388	0.521	0.501
C2	+ stopwords + lemmatisation, pronouns kept	0.762	0.836	0.390	0.532	0.485
Sanity check: variant A reproduces the guide's reference run (0.740 / 0.820 / 0.330 / 0.470) — the
comparison rests on a setup that is known to be correct.

Paired gains (same folds, so the noise partly cancels):

Comparison	Validation Δ F1	5-fold CV Δ F1	Same sign?
stopwords (B − A)	+0.9 pt	+1.9 pt	yes
pronouns (B2 − B)	+1.6 pt	−1.5 pt	no → noise
pronouns (C2 − C)	+1.1 pt	−1.6 pt	no → noise
drop user (B3 − B)	+1.2 pt	+1.0 pt	yes (but empties 58 tweets)
lemmatise (C − B)	+3.3 pt	+2.6 pt	yes
lemmatise (C2 − B2)	+2.9 pt	+2.5 pt	yes
Why C and not C2, although C2 has the best validation F1? Because the pronoun effect is not
established: validation prefers keeping the pronouns, cross-validation prefers dropping them, and the
gap (~1 pt) is smaller than the fold-to-fold spread (~1.2 pt). The rule was declared before looking
at the winner: when the two estimates disagree in sign, take the variant with the better 5-fold CV
mean — it is estimated on 5 × 9,533 tweets instead of 1,324 — and, on a tie, the smaller vocabulary.
C2 is nevertheless carried into notebook 04 as a sensitivity check, and the balancing conclusion does
not change with it (+8.5 pt vs +10.4 pt of F1).

3. Notebook 04 — the four balancing strategies
TF-IDF fitted once on the full original train set, then only rows of X_train are selected,
so the four strategies share exactly the same features. The validation set is never resampled.

Strategy	Train rows	% offensive	Accuracy	Precision	Recall	F1	Caught / 459	False alarms
original (no balancing)	11,916	33.1	0.752	0.791	0.388	0.521	178	47
undersampling	7,882	50.0	0.720	0.585	0.660	0.620	303	215
oversampling	15,950	50.0	0.739	0.630	0.597	0.613	274	161
class_weight (kept)	11,916	33.1	0.739	0.624	0.625	0.625	287	173
Stability over 10 resampling seeds (the TF-IDF matrix is fixed; only the random row selection
changes):

Strategy	mean F1	σ	min	max
class_weight	0.6246	0.0000	0.6246	0.6246
undersampling	0.6141	0.0075	0.6006	0.6261
oversampling	0.6080	0.0094	0.5952	0.6249
original	0.5205	0.0000	0.5205	0.5205
Why class_weight and not undersampling (which has a slightly better recall)?

the F1 difference is inside the seed noise (0.625 vs 0.614 ± 0.008);
undersampling deletes 4,034 real non-offensive tweets (34 % of the training set) — irreversible;
oversampling duplicates 4,552 offensive rows, giving a copied tweet a double weight and
distorting the output probability;
class_weight="balanced" is deterministic: the same model at every run, which matters for
reproducibility and for the report;
training cost is unchanged (no extra rows).
Accuracy goes down (0.752 → 0.739) while F1 and recall go up: with 2 non-offensive tweets per
offensive one, accuracy is not the right objective — we explain this in the report instead of hiding it.

4. Limits to state honestly in the report
WordNet is not perfect: hating → hat (and without the mandatory verb pass, was → wa).
Slang and typos (libtards, idiottt) are left unchanged. Measured impact: ~14 % of the tokens are
modified by lemmatisation, and the vocabulary shrinks by ~20 %.
Validation set = 1,324 tweets: differences under ~1 F1 point are noise. That is why every
decision above rests on two estimates (validation and 5-fold CV / 10 seeds) and not on one score.
Stopword removal can leave a tweet empty in principle. Measured: 0 empty tweets for the
retained variant C (58 for the rejected variant B3). TF-IDF handles an empty tweet as an all-zero
row; none of the retained transformations deletes a row.
Lemmatisation costs time: ~2× the other steps (2.1 s vs 1.3 s per experiment on 13k tweets) —
negligible here, but to be mentioned if the corpus grows.
Single model: all these conclusions are measured with Logistic Regression (the reference model
of the guide). They should be re-checked with the other algorithms of the project (SVM, NB, RF) —
a preprocessing that helps a linear model does not always help a tree.
5. Files to look at (nothing else was modified)
Deliverable	Path
stopword logic	src/stopwords_config.py
lemmatisation logic	src/lemmatization.py
notebook 03 (executed)	notebooks/03_preprocessing_experiments.ipynb
notebook 04 (executed)	notebooks/04_class_balance_experiments.ipynb
metric table 03	results/metrics/03_preprocessing_experiments/preprocessing_comparison.csv
metric table 04	results/metrics/04_class_balance_experiments/balancing_comparison.csv
decision tables	results/metrics/03_preprocessing_experiments/preprocessing_decision.csv, results/metrics/04_class_balance_experiments/final_decision_table.csv
figures	results/figures/03_preprocessing_experiments/, results/figures/04_class_balance_experiments/
unit tests	tests/test_stopwords_and_lemmatization.py (23 tests, pytest -q)