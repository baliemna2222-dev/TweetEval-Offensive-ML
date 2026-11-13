"""
src/torch_utils.py
=========================================================
Shared PyTorch training utilities for advanced models:
    - BiLSTM
    - BERT

Project: TweetEval Offensive Language Detection

Features:
    - Lazy PyTorch import (PyTorch remains optional)
    - Automatic device selection (CUDA / Apple MPS / CPU)
    - Reproducible training
    - Class-weighted CrossEntropyLoss
    - AdamW optimizer
    - Optional learning-rate warmup and decay
    - Automatic mixed precision (CUDA FP16)
    - Gradient scaling
    - Gradient clipping
    - Validation-based model selection
    - Early stopping
    - Best-epoch weights are restored and the fitted model is returned   (NEW)
    - predict_scores(): probabilities of "offensive" for any inputs      (NEW)

IMPORTANT:
    - Vocabulary and preprocessing are handled elsewhere.
    - The validation set selects the best epoch.
    - The test set is never used during training.
"""

import random
import numpy as np

from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
)


# =========================================================
# 1. OPTIONAL PYTORCH IMPORT
# =========================================================

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    TORCH_OK = True

except ImportError:
    TORCH_OK = False


# =========================================================
# 2. DEVICE SELECTION
# =========================================================

def pick_device():
    """
    Select the best available computation device.

    Priority:
        1. NVIDIA CUDA GPU
        2. Apple MPS
        3. CPU
    """

    if not TORCH_OK:
        raise ImportError(
            "PyTorch is not installed. "
            "Install it using: pip install torch"
        )

    if torch.cuda.is_available():
        return torch.device("cuda")

    if (
        hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        return torch.device("mps")

    return torch.device("cpu")


# =========================================================
# 3. REPRODUCIBILITY
# =========================================================

def set_seed(seed=42):
    """
    Set random seeds for reproducibility.

    Controls:
        - Python random
        - NumPy
        - PyTorch CPU
        - PyTorch CUDA
    """

    random.seed(seed)
    np.random.seed(seed)

    if not TORCH_OK:
        return

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =========================================================
# 4. CLASS WEIGHT CALCULATION
# =========================================================

def class_weights_from(y):
    """
    Calculate balanced class weights.

    Formula:
        weight_c = n / (number_of_classes * count_c)

    This gives more importance to the minority class.

    For binary classification:
        Class 0 = Not offensive
        Class 1 = Offensive

    The weights are calculated from TRAINING labels only.
    """

    if not TORCH_OK:
        raise ImportError("PyTorch is required.")

    y = np.asarray(y, dtype=np.int64)

    counts = np.bincount(y, minlength=2)

    if np.any(counts == 0):
        raise ValueError(
            "Both classes must be present in the training data."
        )

    weights = len(y) / (2.0 * counts)

    return torch.tensor(
        weights,
        dtype=torch.float32
    )


# =========================================================
# 5. PREDICTION (NEW)
# =========================================================

def predict_scores(model, inputs, forward, batch_size=128):
    """
    Probability of class 1 (offensive) for every sample.

    Parameters
    ----------
    model:
        Trained PyTorch model (already on its device).

    inputs:
        Tuple of tensors, e.g. (ids,) or (ids, attention_mask).

    forward:
        Same callable as in fit_classifier: forward(model, *batch_inputs) -> logits.

    Returns
    -------
    numpy array of shape (n_samples,).

    Runs in FP32 (no autocast), exactly like the validation step of fit_classifier.
    """

    if not TORCH_OK:
        raise ImportError("PyTorch is required.")

    device = next(model.parameters()).device

    loader = DataLoader(
        TensorDataset(*inputs),
        batch_size=batch_size,
        shuffle=False,
    )

    model.eval()

    out = []

    with torch.no_grad():

        for batch in loader:

            xb = [tensor.to(device) for tensor in batch]

            logits = forward(model, *xb)

            out.append(
                torch.softmax(logits.float(), dim=1)[:, 1].cpu()
            )

    return torch.cat(out).numpy()


# =========================================================
# 6. TRAINING FUNCTION
# =========================================================

def fit_classifier(
    model,
    train_inputs,
    y_train,
    val_inputs,
    y_val,
    forward,
    *,
    epochs,
    lr,
    batch_size,
    weight_decay=0.0,
    warmup_ratio=0.0,
    patience=None,
    seed=42,
    verbose=True,
):
    """
    Train a binary PyTorch classifier.

    Parameters
    ----------
    model:
        PyTorch classification model.

    train_inputs:
        Tuple of training tensors.
        Example: (X_train,)

    y_train:
        Training labels.

    val_inputs:
        Tuple of validation tensors.

    y_val:
        Validation labels.

    forward:
        Function receiving:
            forward(model, *batch_inputs)

        Must return logits of shape (batch_size, 2).

    epochs:
        Maximum training epochs.

    lr:
        Learning rate.

    batch_size:
        Number of samples per batch.

    weight_decay:
        AdamW weight decay.

    warmup_ratio:
        Fraction of training steps used for warmup.

    patience:
        Number of epochs without F1 improvement
        before early stopping.

    seed:
        Random seed.

    Returns
    -------
    history:
        Metrics for every completed epoch.

    best:
        Dictionary containing:
            epoch
            f1
            pred
            scores
            model    <- the model with the BEST-EPOCH weights restored (NEW)
    """

    if not TORCH_OK:
        raise ImportError(
            "PyTorch is not installed. "
            "Install it using: pip install torch"
        )

    # -----------------------------------------------------
    # 6.1. Reproducibility
    # -----------------------------------------------------

    set_seed(seed)

    # -----------------------------------------------------
    # 6.2. Select computation device
    # -----------------------------------------------------

    device = pick_device()

    if verbose:
        print(f"Training device: {device}")

        if device.type == "cuda":
            print(
                "GPU:",
                torch.cuda.get_device_name(0)
            )

    # Move model to GPU or CPU
    model = model.to(device)

    # -----------------------------------------------------
    # 6.3. Prepare labels
    # -----------------------------------------------------

    y_train = np.asarray(y_train, dtype=np.int64)
    y_val = np.asarray(y_val, dtype=np.int64)

    y_tr = torch.as_tensor(
        y_train,
        dtype=torch.long
    )

    # -----------------------------------------------------
    # 6.4. Create DataLoaders
    # -----------------------------------------------------

    generator = torch.Generator()
    generator.manual_seed(seed)

    train_dataset = TensorDataset(
        *train_inputs,
        y_tr
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        pin_memory=(device.type == "cuda"),
    )

    val_dataset = TensorDataset(
        *val_inputs
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        pin_memory=(device.type == "cuda"),
    )

    # -----------------------------------------------------
    # 6.5. Optimizer
    # -----------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )

    # -----------------------------------------------------
    # 6.6. Learning-rate scheduler
    # -----------------------------------------------------

    total_steps = len(train_loader) * epochs

    warmup_steps = int(
        warmup_ratio * total_steps
    )

    def lr_lambda(step):
        if warmup_steps > 0 and step < warmup_steps:
            return (step + 1) / warmup_steps

        return max(
            0.0,
            (total_steps - step) /
            max(1, total_steps - warmup_steps)
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lr_lambda,
    )

    # -----------------------------------------------------
    # 6.7. Weighted loss function
    # -----------------------------------------------------

    class_weights = class_weights_from(
        y_train
    ).to(device)

    loss_fn = torch.nn.CrossEntropyLoss(
        weight=class_weights
    )

    if verbose:
        print(
            "Class weights:",
            class_weights.detach().cpu().numpy()
        )

    # -----------------------------------------------------
    # 6.8. Automatic Mixed Precision (AMP)
    # -----------------------------------------------------

    # FP16 AMP is enabled only for CUDA.
    use_amp = device.type == "cuda"

    # Modern GradScaler API.
    scaler = torch.amp.GradScaler(
        "cuda",
        enabled=use_amp,
    )

    if verbose:
        print(
            "Mixed precision:",
            "FP16 enabled" if use_amp else "disabled"
        )

    # -----------------------------------------------------
    # 6.9. Initialize training history
    # -----------------------------------------------------

    history = []

    best = {
        "f1": -1.0,
        "epoch": 0,
        "pred": None,
        "scores": None,
    }

    best_state = None          # CPU copy of the best-epoch weights (NEW)

    bad_epochs = 0

    # =====================================================
    # 7. EPOCH LOOP
    # =====================================================

    for epoch in range(1, epochs + 1):

        # -------------------------------------------------
        # 7.1. TRAINING
        # -------------------------------------------------

        model.train()

        running_loss = 0.0

        for batch in train_loader:

            # Last element is the target label.
            *xb, yb = batch

            # Transfer data to selected device.
            xb = [
                tensor.to(
                    device,
                    non_blocking=(device.type == "cuda")
                )
                for tensor in xb
            ]

            yb = yb.to(
                device,
                non_blocking=(device.type == "cuda")
            )

            # Reset gradients.
            optimizer.zero_grad(set_to_none=True)

            # Forward pass with optional AMP.
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):

                logits = forward(
                    model,
                    *xb
                )

            # Compute loss in FP32 for numerical stability.
            loss = loss_fn(
                logits.float(),
                yb
            )

            # Backpropagation with gradient scaling.
            scaler.scale(loss).backward()

            # Unscale before gradient clipping.
            scaler.unscale_(optimizer)

            # Prevent exploding gradients.
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
            )

            # Optimizer update.
            scaler.step(optimizer)

            scaler.update()

            # Update learning rate.
            scheduler.step()

            running_loss += loss.item()

        # -------------------------------------------------
        # 7.2. VALIDATION
        # -------------------------------------------------

        model.eval()

        probabilities = []

        with torch.no_grad():

            for batch in val_loader:

                # val_loader contains only input tensors.
                xb = [
                    tensor.to(
                        device,
                        non_blocking=(device.type == "cuda")
                    )
                    for tensor in batch
                ]

                # Validation in FP32.
                # This avoids unnecessary mixed-precision
                # numerical issues during evaluation.
                logits = forward(
                    model,
                    *xb
                )

                # Probability of class 1 (offensive).
                probs = torch.softmax(
                    logits.float(),
                    dim=1
                )[:, 1]

                probabilities.append(
                    probs.cpu()
                )

        # Combine validation probabilities.
        scores = torch.cat(
            probabilities
        ).numpy()

        # Default classification threshold.
        predictions = (
            scores >= 0.5
        ).astype(int)

        # -------------------------------------------------
        # 7.3. METRICS
        # -------------------------------------------------

        val_f1 = f1_score(
            y_val,
            predictions,
            zero_division=0,
        )

        val_precision = precision_score(
            y_val,
            predictions,
            zero_division=0,
        )

        val_recall = recall_score(
            y_val,
            predictions,
            zero_division=0,
        )

        val_accuracy = accuracy_score(
            y_val,
            predictions,
        )

        epoch_loss = (
            running_loss / len(train_loader)
        )

        row = {
            "epoch": epoch,
            "train_loss": epoch_loss,
            "accuracy": val_accuracy,
            "precision": val_precision,
            "recall": val_recall,
            "f1": val_f1,
        }

        history.append(row)

        # -------------------------------------------------
        # 7.4. DISPLAY PROGRESS
        # -------------------------------------------------

        if verbose:

            current_lr = optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch}/{epochs} "
                f"| Loss: {epoch_loss:.4f} "
                f"| Val F1: {val_f1:.4f} "
                f"| Precision: {val_precision:.3f} "
                f"| Recall: {val_recall:.3f} "
                f"| LR: {current_lr:.6f}"
            )

        # -------------------------------------------------
        # 7.5. BEST MODEL SELECTION
        # -------------------------------------------------

        if val_f1 > best["f1"]:

            best = {
                "epoch": epoch,
                "f1": val_f1,
                "pred": predictions.copy(),
                "scores": scores.copy(),
            }

            # Keep the weights of this epoch (CPU copy, detached).
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }

            bad_epochs = 0

        else:

            bad_epochs += 1

        # -------------------------------------------------
        # 7.6. EARLY STOPPING
        # -------------------------------------------------

        if patience is not None and patience > 0:

            if bad_epochs >= patience:

                if verbose:
                    print(
                        f"Early stopping at epoch {epoch}. "
                        f"Best epoch: {best['epoch']} "
                        f"(F1={best['f1']:.4f})"
                    )

                break

    # =====================================================
    # 8. FINAL RESULTS
    # =====================================================

    if best["pred"] is None:
        raise RuntimeError(
            "Training finished without producing a valid model."
        )

    # Restore the best-epoch weights: the model we return is the one that
    # produced best["pred"] / best["scores"], not the last-epoch model.
    model.load_state_dict(best_state)
    model.eval()
    best["model"] = model

    if verbose:

        print("\nTraining completed.")
        print(
            f"Best epoch: {best['epoch']}"
        )
        print(
            f"Best validation F1: {best['f1']:.4f}"
        )

    return history, best
