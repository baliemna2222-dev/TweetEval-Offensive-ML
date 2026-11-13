# Project report – TweetEval Offensive

## 1. Problem and data
Binary classification of tweets (offensive = 1). Official splits: train 11,916 (33.1 % offensive), validation 1,324 (34.7 %), test 860 (27.9 %).
No missing texts; ~1 % duplicates; mentions are anonymised (`@user`, ~93 % of tweets), URLs absent.

## 2. Decisions log
| # | Decision | Evidence |
|---|---|---|
| 1 | Main metric = F1 of the offensive class | imbalance: a constant "non-offensive" model has ~67 % accuracy |
| 2 | Lowercase, decode HTML entities, remove URLs/punctuation/emojis | vocabulary 36,457 → 18,364 distinct tokens |
| 3 | `@mention` → `user`; hashtag word kept | mentions carry no class signal; hashtag words may be informative |
| 4 | Keep apostrophes in contractions and masked insults (`f***`) | negations and obfuscated insults are informative |
| 5 | Custom TF-IDF `token_pattern` | default pattern splits `don't` into `don` and drops `f***` |
| 6 | Stopwords removed, negations protected | validation F1 0.469 → 0.494; CV 0.457 → 0.476 |
| 7 | Lemmatisation applied (`ass`, `us` protected) | F1 0.494 → 0.519; CV 0.476 → 0.501. Inspection found `ass → as`: fixed + unit test |
| 8 | Keep pronouns: no. Drop `user`: no | B2 0.481 (worse than B); B3 0.501 (within noise, C better) |
| 9 | Balancing = `class_weight="balanced"` (train rows only) | LR F1 0.519 → 0.626, recall 0.39 → 0.63; effect depends on the model |

## 3. Validation results (offensive class)
| Model | Balancing | Precision | Recall | F1 |
|---|---|---|---|---|
| **Logistic Regression** | class_weight | 0.625 | 0.627 | **0.626** |
| Random Forest | class_weight | 0.714 | 0.556 | 0.625 |
| Naive Bayes | oversampling | 0.559 | 0.638 | 0.596 |
| SVM (LinearSVC) | none | 0.676 | 0.532 | 0.595 |
| KNN (cosine) | oversampling | 0.464 | 0.442 | 0.453 |

Logistic Regression and Random Forest are tied (gap 0.001); Logistic Regression is preferred (equal F1, ~400× faster, interpretable).

## 4. Final test (used once)
accuracy 0.771 · precision 0.581 · recall 0.642 · F1 0.610.

## 5. Limits and perspectives
See section 7 of `notebooks/08_final_project.ipynb`: context/sarcasm, merged hashtags, subjective labels, topic bias, precision 0.58;
perspectives: BERT/BERTweet, camel-case hashtag splitting, threshold tuning, ensembles.
