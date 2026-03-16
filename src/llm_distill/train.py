"""
llm_distill.train — Generalized training loop for LLM-distilled micro-NNs.

Extracted from MVEE's training pipeline and made application-agnostic so that
any downstream project can import and reuse the same training utilities without
depending on MVEE-specific feature extractors or action vocabularies.

Theoretical grounding
---------------------
- Policy Distillation (Rusu et al., 2015) — arXiv:1511.06295
  CrossEntropyLoss on hard teacher labels (DAgger-style) as used here.

- DAgger (Ross et al., 2011) — arXiv:1011.0686
  Training from teacher-generated trajectories; we follow the same offline
  variant: collect LLM labels once, then train to convergence.
"""

from __future__ import annotations

import time
from copy import deepcopy
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float = 1e-3,
    patience: int = 20,
    device: str = "cpu",
    class_weights: Optional[torch.Tensor] = None,
) -> list[dict]:
    """Train *model* and return per-epoch metrics.

    Optimiser: Adam with weight_decay=1e-4.
    Scheduler: CosineAnnealingLR over *epochs* steps.
    Loss:      CrossEntropyLoss, optionally with *class_weights*.
    Clipping:  Gradient norm clipped to 1.0 per step.
    Early stopping: halts when val_loss does not improve for *patience*
                    consecutive epochs; restores the best checkpoint.

    Parameters
    ----------
    model:
        Any ``nn.Module`` that accepts ``(batch, input_dim)`` tensors and
        returns raw logits of shape ``(batch, num_classes)``.
    train_loader:
        DataLoader yielding ``(features, labels)`` tuples.
    val_loader:
        DataLoader yielding ``(features, labels)`` tuples.
    epochs:
        Maximum number of training epochs.
    lr:
        Initial learning rate for Adam.
    patience:
        Number of consecutive non-improving validation epochs before early
        stopping is triggered.
    device:
        PyTorch device string, e.g. ``"cpu"``, ``"cuda"``, or ``"mps"``.
    class_weights:
        Optional 1-D tensor of length ``num_classes`` passed to
        ``CrossEntropyLoss(weight=...)``.  Use
        ``EpisodeDataset.compute_class_weights`` to generate inverse-frequency
        weights from an imbalanced dataset.

    Returns
    -------
    list[dict]
        One dict per completed epoch with keys:
        ``epoch``, ``train_loss``, ``train_acc``, ``val_loss``, ``val_acc``.
        The list may be shorter than *epochs* when early stopping fires.
    """
    _device = torch.device(device)
    model = model.to(_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    weights = class_weights.to(_device) if class_weights is not None else None
    criterion = nn.CrossEntropyLoss(weight=weights)

    best_val_loss = float("inf")
    best_state: dict | None = None
    patience_counter = 0
    metrics: list[dict] = []

    for epoch in range(1, epochs + 1):
        # ------------------------------------------------------------------ #
        # Training pass
        # ------------------------------------------------------------------ #
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for features, labels in train_loader:
            features = features.to(_device)
            labels = labels.to(_device)

            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * len(labels)
            preds = logits.argmax(dim=-1)
            train_correct += (preds == labels).sum().item()
            train_total += len(labels)

        scheduler.step()

        # ------------------------------------------------------------------ #
        # Validation pass
        # ------------------------------------------------------------------ #
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for features, labels in val_loader:
                features = features.to(_device)
                labels = labels.to(_device)
                logits = model(features)
                loss = criterion(logits, labels)
                val_loss += loss.item() * len(labels)
                preds = logits.argmax(dim=-1)
                val_correct += (preds == labels).sum().item()
                val_total += len(labels)

        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss / max(train_total, 1),
            "train_acc": train_correct / max(train_total, 1),
            "val_loss": val_loss / max(val_total, 1),
            "val_acc": val_correct / max(val_total, 1),
        }
        metrics.append(epoch_metrics)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}: "
                f"train_loss={epoch_metrics['train_loss']:.4f} "
                f"train_acc={epoch_metrics['train_acc']:.3f} "
                f"val_loss={epoch_metrics['val_loss']:.4f} "
                f"val_acc={epoch_metrics['val_acc']:.3f}"
            )

        # ------------------------------------------------------------------ #
        # Early stopping
        # ------------------------------------------------------------------ #
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch} (patience={patience})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return metrics


# ---------------------------------------------------------------------------
# Inference benchmark
# ---------------------------------------------------------------------------

def benchmark_inference(
    model: nn.Module,
    batch_size: int = 50,
    feature_dim: int = 40,
    n_iters: int = 1000,
) -> float:
    """Measure per-batch inference latency in milliseconds.

    Runs *n_iters* forward passes on a random ``(batch_size, feature_dim)``
    tensor after a 10-iteration warm-up, then returns the mean wall-clock time
    per iteration in milliseconds.

    All inference is performed on CPU; the model is not moved.

    Parameters
    ----------
    model:
        The trained ``nn.Module`` to benchmark.
    batch_size:
        Number of samples per batch (default: 50, matching a typical
        50-agent game tick).
    feature_dim:
        Width of the synthetic feature vector (must match ``model.INPUT_DIM``
        if present, otherwise any value that produces valid input).
    n_iters:
        Number of timed iterations.

    Returns
    -------
    float
        Mean inference time per batch in milliseconds.
    """
    model.eval()
    x = torch.randn(batch_size, feature_dim)

    with torch.no_grad():
        # Warm-up
        for _ in range(10):
            _ = model(x)

        t0 = time.perf_counter()
        for _ in range(n_iters):
            _ = model(x)
        elapsed_ms = (time.perf_counter() - t0) / n_iters * 1000.0

    return elapsed_ms
