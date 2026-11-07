# Methodology

1. Load the official train, validation and test splits.
2. Explore class balance, examples and tweet lengths.
3. Apply light text cleaning; retain hashtags, emojis, punctuation and negations initially.
4. Convert text to TF-IDF unigram/bigram features.
5. Compare Naive Bayes, Logistic Regression, Linear SVM and Random Forest.
6. Select by validation F1, then evaluate once on test.

Confirm the dataset label mapping before writing conclusions. TF-IDF must be fitted on train only. The test set must not be used for model selection.