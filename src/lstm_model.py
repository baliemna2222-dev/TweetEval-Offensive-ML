"""
src/lstm_model.py  --  OPTIONAL advanced model: BiLSTM (PyTorch)
================================================================
Unlike TF-IDF models, a recurrent network reads the tweet WORD BY WORD and keeps a memory,
so it can use word ORDER ("not good" != "good").
Architecture:  token ids -> Embedding -> bidirectional LSTM -> max-pooling over time
               -> dropout -> linear (2 classes).
Input text: `text_clean` (already cleaned in notebook 02/03). The vocabulary is built on
TRAIN only (no leakage); unknown words map to <unk>.
Embeddings are learned from scratch (about 12k tweets): expect results close to the TF-IDF
models, not better - pre-trained embeddings (GloVe) or BERT are what really help.

Workflow for the final test (notebooks 05 -> 06)
* notebook 05:  res = train_lstm(...)   then   save_lstm(res, "models/bilstm.pt")
* notebook 06:  res = load_lstm("models/bilstm.pt");  scores = predict_lstm(res, test["text_clean"])
"""
from collections import Counter
import numpy as np
from src import config
from src import torch_utils as tu

SEED = config.RANDOM_STATE
DEFAULT_CFG = dict(max_len=40, vocab_size=20000, min_freq=2, emb_dim=128, hidden=128,
                   dropout=0.4, epochs=8, lr=2e-3, batch_size=64, patience=3)
PAD, UNK = 0, 1                                        # reserved ids


def build_vocab(texts, max_size, min_freq):
    """word -> id, built from TRAIN texts only (most frequent words first)."""
    counts = Counter(w for t in texts for w in str(t).split())
    words = [w for w, c in counts.most_common(max_size - 2) if c >= min_freq]
    return {w: i + 2 for i, w in enumerate(words)}     # ids 0/1 are <pad>/<unk>


def encode(texts, vocab, max_len):
    """Texts -> integer matrix (n, max_len): truncated / zero-padded on the right."""
    out = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, t in enumerate(texts):
        ids = [vocab.get(w, UNK) for w in str(t).split()][:max_len]
        out[i, :len(ids)] = ids
    return out


def make_model(vocab_size, emb_dim, hidden, dropout):
    """Build the BiLSTM classifier (torch imported lazily)."""
    import torch
    import torch.nn as nn

    class BiLSTM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=PAD)   # word id -> vector
            self.lstm = nn.LSTM(emb_dim, hidden, batch_first=True, bidirectional=True)
            self.drop = nn.Dropout(dropout)                                  # regularisation
            self.out = nn.Linear(2 * hidden, 2)                              # 2 directions concatenated -> 2 classes

        def forward(self, ids):
            h, _ = self.lstm(self.emb(ids))                                  # (batch, len, 2*hidden)
            pad = (ids == PAD).unsqueeze(-1)                                 # (batch, len, 1) True on padding
            # Fill value taken from h's own dtype: -1e9 overflows fp16 under autocast (max 65504).
            h = h.masked_fill(pad, torch.finfo(h.dtype).min)                 # ignore padding positions
            pooled = h.max(dim=1).values                                     # max-pool over time
            pooled = pooled.masked_fill(pad.all(dim=1), 0.0)                 # all-pad rows (empty text) -> zeros, avoids -inf/NaN
            return self.out(self.drop(pooled))                               # logits
    return BiLSTM()


def _forward(m, ids):
    return m(ids)


def train_lstm(train_texts, y_train, val_texts, y_val, cfg=None, verbose=True):
    """Train the BiLSTM.

    Returns dict(history, pred, scores, best_epoch, cfg, vocab_size, vocab, model):
      pred / scores = VALIDATION predictions of the best epoch,
      model         = the model with the best-epoch weights restored.
    """
    if not tu.TORCH_OK:
        raise ImportError("PyTorch is not installed: pip install torch")
    import torch
    cfg = {**DEFAULT_CFG, **(cfg or {})}
    tu.set_seed(SEED)
    vocab = build_vocab(train_texts, cfg["vocab_size"], cfg["min_freq"])
    Xtr = torch.as_tensor(encode(train_texts, vocab, cfg["max_len"]))
    Xva = torch.as_tensor(encode(val_texts, vocab, cfg["max_len"]))
    model = make_model(len(vocab) + 2, cfg["emb_dim"], cfg["hidden"], cfg["dropout"])
    history, best = tu.fit_classifier(model, (Xtr,), y_train, (Xva,), y_val,
                                      forward=_forward, epochs=cfg["epochs"],
                                      lr=cfg["lr"], batch_size=cfg["batch_size"],
                                      patience=cfg["patience"], seed=SEED, verbose=verbose)
    return {"history": history, "pred": best["pred"], "scores": best["scores"],
            "best_epoch": best["epoch"], "cfg": cfg, "vocab_size": len(vocab) + 2,
            "vocab": vocab, "model": best["model"]}


def predict_lstm(res, texts, batch_size=256):
    """Probability of 'offensive' for each (cleaned) text. `res` comes from train_lstm or load_lstm."""
    import torch
    X = torch.as_tensor(encode(list(texts), res["vocab"], res["cfg"]["max_len"]))
    return tu.predict_scores(res["model"], (X,), _forward, batch_size=batch_size)


def save_lstm(res, path):
    """Save weights + vocabulary + config in one file (e.g. models/bilstm.pt)."""
    import torch
    from pathlib import Path
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v.detach().cpu() for k, v in res["model"].state_dict().items()}
    torch.save({"state": state, "vocab": res["vocab"], "cfg": res["cfg"]}, path)
    return path


def load_lstm(path):
    """Load what save_lstm wrote. Returns dict(model, vocab, cfg) ready for predict_lstm."""
    import torch
    ck = torch.load(path, map_location="cpu")
    cfg, vocab = ck["cfg"], ck["vocab"]
    model = make_model(len(vocab) + 2, cfg["emb_dim"], cfg["hidden"], cfg["dropout"])
    model.load_state_dict(ck["state"])
    model = model.to(tu.pick_device()).eval()
    return {"model": model, "vocab": vocab, "cfg": cfg}
