# TweetEval – Offensive Language Detection

Binary classification of tweets (**offensive** vs **non-offensive**) with classical machine-learning models and optional deep-learning models (BiLSTM, BERT), on the *Offensive* task of the [TweetEval](https://github.com/cardiffnlp/tweeteval) benchmark.

> Machine Learning mini-project, Faculté des Sciences de Bizerte (FSB), 2026–2027.

---

## Table of contents
1. [Project at a glance](#1-project-at-a-glance)
2. [Quick start](#2-quick-start)
3. [Dataset](#3-dataset)
4. [Notebooks: what to run and in which order](#4-notebooks-what-to-run-and-in-which-order)
5. [Models](#5-models)
6. [Methodology and rules](#6-methodology-and-rules)
7. [Results](#7-results)
8. [Repository structure](#8-repository-structure)
9. [Reproducibility](#9-reproducibility)
10. [Optional models: PyTorch, BERT, GPU](#10-optional-models-pytorch-bert-gpu)
11. [Team workflow](#11-team-workflow)
12. [Troubleshooting](#12-troubleshooting)
13. [References](#13-references)

---

## 1. Project at a glance

| | |
|---|---|
| **Task** | Is a tweet offensive? (binary: `0` = non-offensive, `1` = offensive) |
| **Data** | TweetEval *offensive* (SemEval-2019 Task 6 / OffensEval): 11,916 train · 1,324 validation · 860 test tweets |
| **Challenge** | Class imbalance: about one tweet in three is offensive |
| **Main metric** | **F1-score of the offensive class** (accuracy is inflated by the majority class) |
| **Also reported** | accuracy, precision, recall, average precision (AP), confusion matrix |
| **Required models** | Logistic Regression, SVM (LinearSVC), Random Forest, Naive Bayes, KNN |
| **Optional models** | Complement NB, SGD classifier, voting ensemble, BiLSTM, BERT / DistilBERT |
| **Selection protocol** | Hyperparameters and final model chosen on the **validation** set (+ cross-validation); the **test** set is used once, at the end |
| **Final deliverable** | `07_final_project.ipynb`: one notebook containing all the files, steps and details of the project |

---

## 2. Quick start

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

# 2. Install the dependencies
pip install -r requirements.txt

# 3. (optional) deep-learning models (BiLSTM, BERT): PyTorch + GPU, see section 10

# 4. Put the dataset in data/raw/ (section 3), then launch Jupyter
jupyter lab
```

Open the notebooks in `notebooks/` and run them **in numerical order** (section 4). Start Jupyter from the repository root or from `notebooks/`: every notebook finds the project root automatically.

Minimal dependencies: `numpy`, `pandas`, `scipy`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`, `jupyter`.
Optional: `torch`, `transformers` (LSTM and BERT only).

---

## 3. Dataset

Put these six files in `data/raw/`:

| File | Content |
|---|---|
| `train_text.txt`, `train_labels.txt` | training split |
| `val_text.txt`, `val_labels.txt` | **official** validation split |
| `test_text.txt`, `test_labels.txt` | official test split |

* One tweet per line in `*_text.txt`; one integer label per line in `*_labels.txt`, **in exactly the same order**.
* Labels follow the official TweetEval mapping: `0` = non-offensive, `1` = offensive. Check the dataset's `mapping.txt` if your copy comes from elsewhere.
* User mentions are anonymised as `@user` in the original data.
* If your files have other names or formats, adapt `src/data_loader.py`.

Because the dataset ships with an **official validation set**, we use it as is: we do not re-split the training data.

---

## 4. Notebooks: what to run and in which order

Notebooks 01–06 each read the outputs of the previous ones and write their own files under `results/…/<notebook_name>/`. Notebook 07 is the self-contained final version of the whole project.

| # | Notebook | Purpose | Main outputs |
|---|---|---|---|
| 01 | `01_data_exploration` | Class balance, tweet length, vocabulary, first look at the data | figures, summary tables |
| 02 | `02_data_cleaning` | Cleaning rules (URLs, mentions, hashtags, emojis, casing…) and their effect | cleaned text |
| 03 | `03_preprocessing_experiments` | Compare cleaning / TF-IDF settings on validation, freeze the final choice | `data/processed/*_final.csv`, TF-IDF settings in `src/config.py` |
| 04 | `04_class_balancing` | Compare `class_weight`, oversampling, undersampling, SMOTE-like options | balancing decision (`BALANCING = "class_weight"`) |
| 05 | `05_model_training` | Train and tune all models on the **same** features; validation metrics, confusion matrices, PR curves, cross-validation | `results/metrics/05_model_training/`, validation predictions |
| 06 | `06_model_evaluation` | Final model selection, **single** evaluation on test, confidence intervals, significance tests, error analysis, saved final model | `results/metrics/06_model_evaluation/`, `models/final_model.joblib` |
| 07 | `07_final_project` | **Complete, self-contained version of the whole project**: gathers all the code (from `src/`), every step of notebooks 01–06 with detailed explanations, and the final results in one document, ready to hand in | `results/metrics/07_final_project/`, `results/figures/07_final_project/`, HTML export for the report |

Rules of thumb:
* **Never skip a notebook** among 01–06: 05 needs the `*_final.csv` files from 03; 06 needs the results of 05.
* Notebook 05 has switches at the top (`RUN_EXTRA`, `RUN_LSTM`, `RUN_BERT`, `RUN_CV`). The five required models always run; slow or optional parts can be turned off.
* **Notebook 07 is the hand-in notebook.** It reproduces the full pipeline (exploration → cleaning → preprocessing → balancing → training → evaluation) from the raw files to the single test evaluation, with all the details in one place. It does not replace notebooks 01–06, which stay the working notebooks of each team member.
* Notebook 07 follows the same rules as the others (section 6): same seed, no leakage, tuning on validation, test used once. Its numbers must match those of notebooks 05 and 06; if not, see `reproducibility_check.csv`.
* To run everything headless and export HTML for the report:
  ```bash
  jupyter nbconvert --to notebook --execute --inplace notebooks/0X_name.ipynb
  jupyter nbconvert --to html notebooks/0X_name.ipynb
  ```
* To export the final notebook for the report:
  ```bash
  jupyter nbconvert --to notebook --execute --inplace notebooks/07_final_project.ipynb
  jupyter nbconvert --to html notebooks/07_final_project.ipynb
  ```

---

## 5. Models

One file per model in `src/`, all with the same interface (`train(...)`, `tune(...)`).

| Model | File | Group | How the class imbalance is handled |
|---|---|---|---|
| Logistic Regression | `logistic_regression.py` | required | `class_weight="balanced"` |
| SVM (LinearSVC) | `svm.py` | required | `class_weight="balanced"` |
| Random Forest | `random_forest.py` | required | `class_weight="balanced"` |
| Naive Bayes | `naive_bayes.py` | required | equal class priors (`fit_prior=False`) |
| KNN (cosine) | `knn.py` | required | random oversampling of the **training** set |
| Complement NB, SGD, voting ensemble | `extra_models.py` | optional | built in / `class_weight` |
| BiLSTM | `lstm_model.py` | optional (needs `torch`) | class-weighted loss |
| BERT / DistilBERT | `bert_model.py` | optional (needs `torch`, `transformers`) | class-weighted loss |

Shared helpers: `model_utils.py` (grid search, oversampling, scores, top terms) and `torch_utils.py` (training loop of the neural models).

---

## 6. Methodology and rules

These rules apply to **every** team member and **every** model:

1. **Same splits for everyone.** Use the official train / validation / test files; never reshuffle them.
2. **No leakage.** TF-IDF (and any vocabulary) is fitted on **train only**, then applied to validation and test. Resampling touches **training rows only**.
3. **One random seed:** `config.RANDOM_STATE = 42`, used everywhere.
4. **Tune on validation, never on test.** Hyperparameters, thresholds and the final model are chosen on validation (and cross-validation).
5. **The test set is used once**, in notebook 06 (and reproduced identically in notebook 07), after all choices are frozen. Test results are reported as they are, even if disappointing.
6. **Main criterion:** F1 of the offensive class. If two models are within about 0.01 F1, treat them as tied and decide on cross-validation, then on simplicity and speed.
7. **Report uncertainty.** Small gaps between models are often noise: notebook 06 gives bootstrap confidence intervals and McNemar tests.
8. **BERT / LSTM text input.** BERT uses the original tweet; the LSTM and TF-IDF models use the cleaned text.

---

## 7. Results

> Fill this section from `results/metrics/06_model_evaluation/test_metrics.csv` once notebook 06 has been run on the real data. Do not edit numbers by hand.

| Model | Accuracy | Precision | Recall | F1 (offensive) | AP |
|---|---|---|---|---|---|
| … | … | … | … | … | … |

* **Final model:** … (chosen on validation + cross-validation)
* **Main weakness:** … (see the error analysis in notebook 06)
* **Figures:** `results/figures/06_model_evaluation/`
* **Full detailed walkthrough:** `notebooks/07_final_project.ipynb` (also exported as HTML).

---

## 8. Repository structure

```text
.
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                      # the six official files (never modified)
│   └── processed/                # train_final.csv, val_final.csv, test_final.csv
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_preprocessing_experiments.ipynb
│   ├── 04_class_balancing.ipynb
│   ├── 05_model_training.ipynb
│   ├── 06_model_evaluation.ipynb
│   └── 07_final_project.ipynb    # full project in one notebook (hand-in version)
├── src/                          # reusable code (notebooks only call it)
│   ├── config.py                 # seed, TF-IDF parameters, balancing decision
│   ├── utils.py                  # output folders, figure saving
│   ├── data_loader.py            # reads the raw files
│   ├── preprocessing.py          # text cleaning
│   ├── features.py               # TF-IDF (fit on train only)
│   ├── evaluation.py             # accuracy / precision / recall / F1, confusion matrix
│   ├── model_utils.py            # grid search, oversampling, scores
│   ├── logistic_regression.py    # ─┐
│   ├── svm.py                    #  │ required models
│   ├── random_forest.py          #  │
│   ├── naive_bayes.py            #  │
│   ├── knn.py                    # ─┘
│   ├── extra_models.py           # optional classical models
│   ├── torch_utils.py            # shared training loop (neural models)
│   ├── lstm_model.py             # optional BiLSTM
│   └── bert_model.py             # optional BERT
├── results/
│   ├── figures/<notebook>/       # plots
│   ├── metrics/<notebook>/       # CSV tables
│   └── predictions/<notebook>/   # per-model predictions
├── models/                       # final_model.joblib, tfidf_vectorizer.joblib
├── tests/                        # unit tests
└── docs/                         # methodology, experiment log, report template
```

---

## 9. Reproducibility

* Fixed seed (`42`) for splits, sampling, models and shuffling.
* Notebook 05 stores every chosen hyperparameter in `results/metrics/05_model_training/best_hyperparameters.csv`.
* Notebook 06 **retrains** each model from those parameters and checks that it reproduces the validation scores of notebook 05 (`reproducibility_check.csv`). A mismatch means data, settings or library versions changed between runs.
* Notebook 07 must reproduce the same numbers as notebooks 05 and 06. Keep its code in sync with `src/` and `config.py`.
* Pin your library versions in `requirements.txt` (`pip freeze > requirements.txt`) before handing in the project.
* Results of neural models can differ slightly across hardware (CPU vs GPU) even with a fixed seed.

---

## 10. Optional models: PyTorch, BERT, GPU

The five required models and the classical extras need only scikit-learn. The **BiLSTM** and **BERT** need `torch` (and BERT also `transformers`). A GPU is strongly recommended for BERT.

### 10.1 Do I have a usable GPU?

* **NVIDIA GPU (Windows / Linux):** open a terminal and run
  ```bash
  nvidia-smi
  ```
  If it prints a table with your GPU name and a *CUDA Version*, the driver is installed. If the command is not found, install or update the driver from [nvidia.com/drivers](https://www.nvidia.com/drivers). **You do not need to install the CUDA Toolkit separately:** the pip wheels of PyTorch bundle the CUDA runtime; only the NVIDIA **driver** must be recent enough.
* **Apple Silicon Mac (M1/M2/M3…):** PyTorch uses the GPU through *MPS*; no extra install (see below).
* **AMD GPU:** supported on **Linux** through ROCm builds only (select *ROCm* on the PyTorch selector).
* **No GPU:** use the CPU build, or run on Google Colab (section 10.4).

### 10.2 Install PyTorch with GPU support (NVIDIA, pip)

Always do this **inside your virtual environment** (`.venv`), with Python 3.10 or newer.

1. **Get the exact command from the official selector:** <https://pytorch.org/get-started/locally/>
   Choose *Stable* · your OS · *Pip* · *Python* · the **CUDA** version shown (the list of CUDA versions changes with each release; the latest one your driver supports is usually the best choice). Copy the command it shows. It looks like this (the `cuXXX` part is the one that changes):
   ```bash
   # Windows or Linux, NVIDIA GPU. Replace cu128 by the CUDA version given by the selector (e.g. cu126, cu128)
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
   ```
   `torchvision` and `torchaudio` are not used by this project; `pip install torch --index-url https://download.pytorch.org/whl/cu128` is enough.

2. **If you already installed a CPU-only `torch`** (for example through `pip install torch` or from a requirements file), remove it first, otherwise pip keeps the CPU version:
   ```bash
   pip uninstall -y torch torchvision torchaudio
   pip install torch --index-url https://download.pytorch.org/whl/cu128
   ```

3. **Install the Hugging Face library** (BERT only):
   ```bash
   pip install transformers
   ```

4. **Verify** that PyTorch sees the GPU:
   ```bash
   python -c "import torch; print(torch.__version__, '| CUDA available:', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no GPU')"
   ```
   You should see `CUDA available: True` and the name of your GPU. A version string ending in `+cpu` means the CPU build was installed (go back to step 2).

5. **Make Jupyter use this environment.** If the notebook still says `torch present: False`, the kernel is not running from your `.venv`:
   ```bash
   pip install ipykernel
   python -m ipykernel install --user --name tweeteval --display-name "Python (tweeteval)"
   ```
   then choose the kernel **Python (tweeteval)** in Jupyter (*Kernel → Change kernel*), and restart it.

### 10.3 Other platforms

| Platform | Command | Notes |
|---|---|---|
| **macOS (Apple Silicon)** | `pip install torch` | GPU used through MPS; the code detects it automatically. Mixed precision is only used on NVIDIA GPUs, so training is a bit slower than on CUDA. |
| **Linux + AMD (ROCm)** | command from the selector (*ROCm*) | Same `torch.cuda` API; check with the verification line above. |
| **CPU only** | `pip install torch --index-url https://download.pytorch.org/whl/cpu` | Works everywhere; DistilBERT fine-tuning takes roughly 30–60 minutes for 3 epochs. |

### 10.4 No GPU? Use Google Colab (free)

1. Open [colab.research.google.com](https://colab.research.google.com) and upload `05_model_training.ipynb`.
2. *Runtime → Change runtime type → T4 GPU.*
3. Upload `src/` and `data/processed/` (or mount Google Drive), then in the first cell run `!pip install transformers` (PyTorch is already installed on Colab).
4. In the setup cell of the notebook set `RUN_BERT = True` (and `RUN_LSTM = True`).
5. Download `results/` when finished.

### 10.5 Using the GPU in the notebooks

* In notebook 05, `RUN_LSTM` and `RUN_BERT` are `"auto"`: they run only if `torch` is installed **and** a GPU is detected. The setup cell prints `torch present` and `GPU` so you can check it immediately. Set them to `True` to force a run (even on CPU) or `False` to skip.
* The training loop picks the device automatically (CUDA, then MPS, then CPU) and uses mixed precision (fp16) on NVIDIA GPUs.
* **Out of memory?** In `src/bert_model.py` lower `batch_size` (32 → 16 → 8) or `max_len` (64 → 48). DistilBERT with the default settings fits in about 4–6 GB of GPU memory.
* Default checkpoint: `distilbert-base-uncased`. Alternatives (edit `DEFAULT_CFG` in `src/bert_model.py`): `bert-base-uncased`, `roberta-base`, `cardiffnlp/twitter-roberta-base` (pre-trained on tweets). The first run needs internet access to download the model (a few hundred MB, cached afterwards).
* **Do not** use `cardiffnlp/twitter-roberta-base-offensive` as your model: it is already fine-tuned on this exact dataset. Mention it only as an external reference.
* If a neural model is skipped, say so explicitly in the report and explain why.

---

## 11. Team workflow

Suggested split for a team of four:

| Member | Responsibility | Main files |
|---|---|---|
| 1 | Dataset and exploration | `data_loader.py`, notebook 01 |
| 2 | Cleaning and features | `preprocessing.py`, `features.py`, notebooks 02–04 |
| 3 | Models | model files in `src/`, notebook 05 |
| 4 | Evaluation and report | `evaluation.py`, notebook 06, `docs/` |
| All | Final hand-in notebook | notebook 07 (assembled together, kept in sync with `src/`) |

Git rules:
* Work on **feature branches** and open **Pull Requests**; never commit directly to `main`.
* Keep the notebooks' **outputs cleared** unless you are committing a final version (large outputs make merges painful): `jupyter nbconvert --clear-output --inplace notebooks/*.ipynb`.
* Do not commit `data/raw/`, `.venv/`, or large model files unless the course requires it.
* Shared variable names are fixed across the project (`train_df`, `val_df`, `text_clean`, `X_train_tfidf`, `y_train`, …): do not rename them without telling the team.
* Before opening a Pull Request: run the notebook you changed from top to bottom (**Kernel → Restart & Run All**) and run the tests (`pytest -q`).

---

## 12. Troubleshooting

| Problem | Likely cause and fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Start Jupyter from the repository root (or `notebooks/`); notebooks add the root to the Python path. |
| `train_final.csv not found` | Run notebook 03 first; later notebooks depend on it. |
| TF-IDF fails with `NaN` | Empty cleaned tweets were read as missing values: load with `pd.read_csv(..., keep_default_na=False)`. |
| LSTM / BERT section skipped | `torch` not installed, no GPU in `"auto"` mode, or the CPU build of torch is installed (`+cpu`): see section 10. |
| `torch.cuda.is_available()` is `False` although I have an NVIDIA GPU | CPU build installed (uninstall and reinstall with the `--index-url …/cuXXX` command), outdated NVIDIA driver (`nvidia-smi`), or Jupyter using another Python environment (section 10.2, step 5). |
| BERT cannot download the model | No internet access or a proxy issue; or `transformers` is missing. |
| Out-of-memory error during BERT | Lower `batch_size` or `max_len` in `src/bert_model.py`. |
| Validation scores differ between notebooks 05 and 06 | See `reproducibility_check.csv`: data, TF-IDF settings, seed or library version changed. |
| Notebook 07 gives different numbers from 05/06 | The code copied into 07 differs from `src/` (settings, seed, TF-IDF parameters). Keep 07 in sync with `src/` and `config.py`, then check `reproducibility_check.csv`. |
| Results look too good to be true | Check for leakage: TF-IDF fitted on all data, test used for tuning, or duplicated tweets across splits. |

---

## 13. References

* Barbieri, F., Camacho-Collados, J., Espinosa Anke, L., Neves, L. (2020). **TweetEval: Unified Benchmark and Comparative Evaluation for Tweet Classification.** *Findings of EMNLP 2020.*
* Zampieri, M., Malmasi, S., Nakov, P., Rosenthal, S., Farra, N., Kumar, R. (2019). **SemEval-2019 Task 6: Identifying and Categorizing Offensive Language in Social Media (OffensEval).** *SemEval 2019.*
* Devlin, J., Chang, M.-W., Lee, K., Toutanova, K. (2019). **BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding.** *NAACL 2019.*
* Pedregosa, F. et al. (2011). **Scikit-learn: Machine Learning in Python.** *JMLR 12.*