"""
src/bert_model.py  --  OPTIONAL advanced model: BERT-family fine-tuning (Hugging Face)
======================================================================================
BERT is a Transformer pre-trained on huge text corpora; we FINE-TUNE it (all weights, small
learning rate) on our 2-class task. It understands context ("sick" in "that's sick" vs "I feel sick").
Important choices
* Input text: the ORIGINAL tweet (`text`), NOT the aggressively cleaned one: BERT's own
  tokenizer handles punctuation, emojis, casing and needs stop-words to understand context.
* Default checkpoint: distilbert-base-uncased (40% smaller / 2x faster than BERT, ~97% of its quality).
  Alternatives: "bert-base-uncased", "roberta-base", or "cardiffnlp/twitter-roberta-base"
  (pre-trained on tweets = usually best here).
  DO NOT use "cardiffnlp/twitter-roberta-base-offensive": it is ALREADY fine-tuned on this very
  dataset, so it would not be your own model (only mention it as an external reference).
* Needs: pip install torch transformers ; a GPU is strongly recommended (Google Colab is free).
  On CPU: ~30-60 minutes for 3 epochs on 12k tweets.

Workflow for the final test (notebooks 05 -> 06)
* notebook 05:  res = train_bert(...)   then   save_bert(res, "models/bert")
* notebook 06:  res = load_bert("models/bert");  scores = predict_bert(res, test["text"])
  The model that is saved is the BEST-EPOCH model selected on validation (weights restored by
  torch_utils.fit_classifier), so the test set is scored once with exactly the validated model.
"""
import json
from pathlib import Path

import numpy as np
from src import config
from src import torch_utils as tu

SEED = config.RANDOM_STATE
DEFAULT_CFG = dict(model_name="distilbert-base-uncased", max_len=64, batch_size=32,
                   epochs=3, lr=2e-5, weight_decay=0.01, warmup_ratio=0.1, patience=None)


def _forward(m, ids, mask):
    """Logits of a Hugging Face sequence-classification model."""
    return m(input_ids=ids, attention_mask=mask).logits


def _encode(tok, texts, max_len):
    """texts -> (input_ids, attention_mask) tensors, padded / truncated to max_len."""
    enc = tok(list(map(str, texts)), padding="max_length", truncation=True,
              max_length=max_len, return_tensors="pt")                     # tweets are short: 64 tokens is enough
    return enc["input_ids"], enc["attention_mask"]


def train_bert(train_texts, y_train, val_texts, y_val, cfg=None, verbose=True):
    """Fine-tune a Transformer.

    Returns dict(history, pred, scores, best_epoch, cfg, model, tokenizer):
      pred / scores = VALIDATION predictions of the best epoch,
      model         = the model with the best-epoch weights restored.
    """
    if not tu.TORCH_OK:
        raise ImportError("PyTorch is not installed: pip install torch transformers")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    tu.set_seed(SEED)

    tok = AutoTokenizer.from_pretrained(cfg["model_name"])                 # sub-word tokenizer of the checkpoint
    tr_ids, tr_mask = _encode(tok, train_texts, cfg["max_len"])
    va_ids, va_mask = _encode(tok, val_texts, cfg["max_len"])

    model = AutoModelForSequenceClassification.from_pretrained(cfg["model_name"], num_labels=2)
    history, best = tu.fit_classifier(
        model, (tr_ids, tr_mask), y_train, (va_ids, va_mask), y_val,
        forward=_forward,
        epochs=cfg["epochs"], lr=cfg["lr"], batch_size=cfg["batch_size"],
        weight_decay=cfg["weight_decay"], warmup_ratio=cfg["warmup_ratio"],
        patience=cfg["patience"], seed=SEED, verbose=verbose)
    return {"history": history, "pred": best["pred"], "scores": best["scores"],
            "best_epoch": best["epoch"], "cfg": cfg,
            "model": best["model"], "tokenizer": tok}


def predict_bert(res, texts, batch_size=128):
    """Probability of 'offensive' for each text. `res` comes from train_bert or load_bert."""
    ids, mask = _encode(res["tokenizer"], texts, res["cfg"]["max_len"])
    return tu.predict_scores(res["model"], (ids, mask), _forward, batch_size=batch_size)


def save_bert(res, path):
    """Save the fine-tuned model, its tokenizer and the config in the folder `path`."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    res["model"].save_pretrained(path)
    res["tokenizer"].save_pretrained(path)
    (path / "cfg.json").write_text(json.dumps(res["cfg"]), encoding="utf-8")
    return path


def load_bert(path):
    """Load what save_bert wrote. Returns dict(model, tokenizer, cfg) ready for predict_bert."""
    if not tu.TORCH_OK:
        raise ImportError("PyTorch is not installed: pip install torch transformers")
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    path = Path(path)
    tok = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSequenceClassification.from_pretrained(path).to(tu.pick_device()).eval()
    cfg = json.loads((path / "cfg.json").read_text(encoding="utf-8"))
    return {"model": model, "tokenizer": tok, "cfg": cfg}
