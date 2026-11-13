# =====================================================================================
# NOTEBOOK 05  --  add right after each training cell (adapt the variable names)
# =====================================================================================
from src import bert_model, lstm_model

# after:  res_lstm = lstm_model.train_lstm(...)
lstm_model.save_lstm(res_lstm, MODEL_DIR / "bilstm.pt")        # best-epoch weights + vocab + cfg

# after:  res_bert = bert_model.train_bert(...)
bert_model.save_bert(res_bert, MODEL_DIR / "bert")             # folder with model, tokenizer, cfg.json


# =====================================================================================
# NOTEBOOK 06  --  new cell, placed right AFTER the existing section 5 (final test + saving)
# Scores the best-on-validation model on the test set ONCE, even if it is a neural model.
# =====================================================================================
from src import bert_model, lstm_model

NEURAL = {
    # model name in results : (loader, predictor, column fed to the model)
    "BiLSTM": (lambda: lstm_model.load_lstm(MODEL_DIR / "bilstm.pt"), lstm_model.predict_lstm, "text_clean"),
    "BERT":   (lambda: bert_model.load_bert(MODEL_DIR / "bert"),       bert_model.predict_bert, "text"),
}

overall = results.iloc[0]["Model"]                       # best on validation among ALL models
print("Best on validation (all models):", overall)

if overall in NEURAL:
    load, predict, col = NEURAL[overall]
    res_nn = load()
    scores_test = predict(res_nn, test[col])             # BERT reads the ORIGINAL tweet, BiLSTM the cleaned one
    y_pred_nn = (scores_test >= 0.5).astype(int)         # same 0.5 threshold as used for validation
    mt_nn = compute_metrics(y_test, y_pred_nn)
    nn_table = pd.DataFrame([{"Model": overall, "Accuracy": mt_nn["accuracy"], "Precision": mt_nn["precision"],
                              "Recall": mt_nn["recall"], "F1": mt_nn["f1"]}]).round(4)
    display(pd.concat([nn_table, test_table], ignore_index=True))   # neural vs best classical, side by side

    fig, ax = plt.subplots(figsize=(4.6, 4))
    ConfusionMatrixDisplay(get_confusion_matrix(y_test, y_pred_nn),
                           display_labels=["non-off.", "offensive"]).plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{overall} - TEST set"); ax.grid(False)
    fig.tight_layout(); savefig(fig, P.figures / f"confusion_matrix_test_{slug(overall)}.png"); plt.show()

    nn_table.to_csv(P.metrics / "final_test_metrics_neural.csv", index=False)
    pd.DataFrame({"text": test["text"], "y_true": y_test, "y_pred": y_pred_nn, "score_offensive": scores_test}
                 ).to_csv(PRED06 / f"test_predictions_{slug(overall)}.csv", index=False)
    print(f"Validation F1 = {results.iloc[0]['F1']:.4f}  ->  Test F1 = {mt_nn['f1']:.4f}")
else:
    print("The best model is classical: the test result above already covers it.")
