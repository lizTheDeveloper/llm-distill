"""
Example: Distilling the Precursors limbic urgency weights network with llm-distill.

The Precursors limbic system maps an agent's internal chemical state to a set of
urgency weights that drive moment-to-moment behavioural prioritisation.  Unlike
the MVEE Talker/Executor networks — which perform multi-class classification over
a discrete action vocabulary — the limbic network is a multi-label regression
problem: each output weight is an independent scalar in [0, 1], and all 13
weights can be simultaneously elevated.  This requires MSELoss (not
CrossEntropyLoss) and sigmoid output activation (not softmax).

Architecture
------------
Input layer  : 140 features
               - 97  normalised chemical concentrations
               -  10  species one-hot encoding
               -   1  normalised tier
               -  32  imprint dimensions
Hidden layers: 256 → 512 → 512 → 128  (LayerNorm + GELU each; Dropout after
               block index 1, i.e. after the second 512 layer, matching
               PolicyNN's dropout_after = (4-1)//2 = 1)
Output layer : 13 urgency weights  (sigmoid applied post-forward at inference)

Data pipeline
-------------
1. Run Precursors — the EpisodeCollector.ts logs chemical states and LLM-labeled
   ideal urgency weights to:
       training_data/episodes_limbic.jsonl
   Each JSONL line looks like:
       {"feature_vector": [0.1, ...], "label_vector": [0.8, 0.1, ...]}
   where feature_vector has 140 floats and label_vector has 13 floats.
2. Run this script to train LimbicNN and export JSON weights:
       python3 examples/precursors_example.py \\
           --episodes-dir training_data \\
           --output-dir   weights \\
           --epochs       200
3. The TypeScript runtime loads weights/limbic_nn.json at startup
   (see LimbicPolicyInference.ts).

Refs
----
- Policy Distillation (Rusu et al., 2015)       arXiv:1511.06295
  Distilling a teacher policy (here: LLM oracle) into a compact student
  network; the offline imitation variant is used — collect labels once, then
  train to convergence.

- LLM4Teach (Fu et al., 2023)                   arXiv:2311.13373
  LLMs as teacher oracles for agent training; demonstrates strong
  sample-efficiency gains when training on LLM-generated labels rather than
  environment reward alone.

- Creatures heritage: Grand, S. (2000). "Creation: Life and How to Make It."
  The biochemical urgency-weight architecture descends directly from the
  Creatures engine (Steve Grand, 1996).  Chemicals map to drives; drives
  compete via normalised weights.  Precursors extends this with LLM-labeled
  ideal weights as distillation targets.

- DAgger (Ross et al., 2011)                    arXiv:1011.0686
  Training from teacher-generated trajectories; we follow the offline variant:
  the LLM labels episodes during live play, the network trains on the
  accumulated dataset, and the cycle repeats between sessions.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split

from llm_distill import PolicyNN, export_weights_json

# ---------------------------------------------------------------------------
# Limbic output names — must stay in sync with LimbicPolicyInference.ts
# ---------------------------------------------------------------------------

WEIGHT_NAMES = [
    "hunger",
    "thirst",
    "pain",
    "fatigue",
    "fear",
    "anger",
    "loneliness",
    "boredom",
    "curiosity",
    "escape",
    "social",
    "rest",
    "limbicInfluence",
]

# ---------------------------------------------------------------------------
# Feature-vector layout constants
# ---------------------------------------------------------------------------

CHEMICAL_COUNT = 97   # normalised chemical concentrations
NUM_SPECIES    = 10   # species one-hot
# + 1 normalised tier
IMPRINT_DIM    = 32   # imprint embedding dimensions
FEATURE_DIM    = CHEMICAL_COUNT + NUM_SPECIES + 1 + IMPRINT_DIM  # 140
OUTPUT_DIM     = 13   # one urgency weight per WEIGHT_NAMES entry

# LimbicNN hidden layer widths — matches the Precursors architecture exactly.
# PolicyNN places dropout after block index (len-1)//2 = (4-1)//2 = 1,
# i.e. after the second hidden block (the second 512-wide layer).
HIDDEN_LAYERS = [256, 512, 512, 128]

# Minimum labelled samples before training is attempted.
MIN_SAMPLES = 50


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class RegressionDataset(Dataset):
    """JSONL episode loader for multi-label regression targets.

    Each line of the JSONL file must contain:
        {"feature_vector": [float, ...], "label_vector": [float, ...]}

    Lines whose ``feature_vector`` length does not equal ``feature_dim``, whose
    ``label_vector`` length does not equal ``output_dim``, or that cannot be
    parsed as JSON are silently skipped.  A load-stats summary is printed on
    construction.

    Args:
        filepath:   Path to the ``.jsonl`` file produced by EpisodeCollector.ts.
        feature_dim: Expected length of each ``feature_vector`` (default: 140).
        output_dim:  Expected length of each ``label_vector`` (default: 13).
    """

    def __init__(
        self,
        filepath: Path | str,
        feature_dim: int = FEATURE_DIM,
        output_dim: int = OUTPUT_DIM,
    ) -> None:
        filepath = Path(filepath)
        self.feature_dim = feature_dim
        self.output_dim = output_dim
        self.samples: list[tuple[list[float], list[float]]] = []

        loaded = 0
        skipped_fv_dim = 0
        skipped_lv_dim = 0
        skipped_parse = 0

        with open(filepath, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    ep = json.loads(line)
                except json.JSONDecodeError:
                    skipped_parse += 1
                    continue

                fv = ep.get("feature_vector", [])
                lv = ep.get("label_vector", [])

                if len(fv) != feature_dim:
                    skipped_fv_dim += 1
                    continue

                if len(lv) != output_dim:
                    skipped_lv_dim += 1
                    continue

                self.samples.append((fv, lv))
                loaded += 1

        total_skipped = skipped_fv_dim + skipped_lv_dim + skipped_parse
        print(
            f"  Loaded {loaded} samples, skipped {total_skipped} "
            f"(fv_dim_mismatch={skipped_fv_dim}, lv_dim_mismatch={skipped_lv_dim}, "
            f"parse_error={skipped_parse})"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        fv, lv = self.samples[idx]
        return (
            torch.tensor(fv, dtype=torch.float32),
            torch.tensor(lv, dtype=torch.float32),
        )


# ---------------------------------------------------------------------------
# Regression training loop
# ---------------------------------------------------------------------------

def train_regression(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float = 1e-3,
    patience: int = 20,
    device: str = "cpu",
) -> list[dict]:
    """Train *model* with MSELoss for multi-label urgency-weight regression.

    The training protocol matches ``llm_distill.train_model`` in all respects
    except the loss function: CrossEntropyLoss is replaced with MSELoss because
    the limbic network predicts 13 independent continuous values rather than a
    probability distribution over a discrete action set.

    Optimiser: Adam with weight_decay=1e-4.
    Scheduler: CosineAnnealingLR over *epochs* steps.
    Loss:      MSELoss (mean over all 13 output dimensions and all samples).
    Clipping:  Gradient norm clipped to 1.0 per step.
    Early stopping: halts when val_loss does not improve for *patience*
                    consecutive epochs; restores the best checkpoint.

    Parameters
    ----------
    model:
        Any ``nn.Module`` that accepts ``(batch, FEATURE_DIM)`` tensors and
        returns raw pre-sigmoid outputs of shape ``(batch, OUTPUT_DIM)``.
        Sigmoid is applied *inside* this function for loss computation so that
        the model stores weights without a baked-in activation (matching the
        PolicyNN convention for the classification case, where logits are
        returned and softmax is applied at call sites).
    train_loader:
        DataLoader yielding ``(features, targets)`` tuples where both tensors
        are ``float32``.
    val_loader:
        DataLoader yielding ``(features, targets)`` tuples.
    epochs:
        Maximum number of training epochs.
    lr:
        Initial learning rate for Adam.
    patience:
        Consecutive non-improving validation epochs before early stopping.
    device:
        PyTorch device string, e.g. ``"cpu"``, ``"cuda"``, or ``"mps"``.

    Returns
    -------
    list[dict]
        One dict per completed epoch with keys:
        ``epoch``, ``train_loss``, ``val_loss``.
        The list may be shorter than *epochs* when early stopping fires.
    """
    _device = torch.device(device)
    model = model.to(_device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.MSELoss()

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
        train_total = 0

        for features, targets in train_loader:
            features = features.to(_device)
            targets  = targets.to(_device)

            optimizer.zero_grad()
            # Apply sigmoid so predictions are in [0, 1], matching label_vector.
            preds = torch.sigmoid(model(features))
            loss  = criterion(preds, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss  += loss.item() * len(targets)
            train_total += len(targets)

        scheduler.step()

        # ------------------------------------------------------------------ #
        # Validation pass
        # ------------------------------------------------------------------ #
        model.eval()
        val_loss  = 0.0
        val_total = 0

        with torch.no_grad():
            for features, targets in val_loader:
                features = features.to(_device)
                targets  = targets.to(_device)
                preds    = torch.sigmoid(model(features))
                loss     = criterion(preds, targets)
                val_loss  += loss.item() * len(targets)
                val_total += len(targets)

        epoch_metrics = {
            "epoch":      epoch,
            "train_loss": train_loss / max(train_total, 1),
            "val_loss":   val_loss   / max(val_total,   1),
        }
        metrics.append(epoch_metrics)

        if epoch % 10 == 0 or epoch == 1:
            print(
                f"  Epoch {epoch:3d}: "
                f"train_loss={epoch_metrics['train_loss']:.5f}  "
                f"val_loss={epoch_metrics['val_loss']:.5f}"
            )

        # ------------------------------------------------------------------ #
        # Early stopping
        # ------------------------------------------------------------------ #
        if val_loss < best_val_loss:
            best_val_loss    = val_loss
            best_state       = {k: v.cpu().clone() for k, v in model.state_dict().items()}
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
# Export — LimbicPolicyInference.ts-compatible JSON
# ---------------------------------------------------------------------------

def export_limbic_weights_json(
    model: nn.Module,
    output_path: Path | str,
) -> None:
    """Serialize LimbicNN weights to JSON for the TypeScript runtime.

    Produces a superset of the standard ``export_weights_json`` schema, adding
    the ``weight_names`` field (in place of ``actions``) so that
    ``LimbicPolicyInference.ts`` can map output indices to named urgency weights
    without maintaining a separate lookup table.

    JSON schema
    -----------
    ::

        {
          "model":        "limbic",
          "input_dim":    140,
          "output_dim":   13,
          "activation":   "sigmoid",
          "weight_names": ["hunger", "thirst", ...],
          "weights": {
            "net.0.weight": [[...], ...],
            "net.0.bias":   [...],
            ...
          }
        }

    Args:
        model:       Trained ``PolicyNN`` instance (will not be mutated).
        output_path: Destination file path (e.g. ``weights/limbic_nn.json``).
    """
    output_path = Path(output_path)

    raw_weights: dict[str, list] = {}
    for name, param in model.named_parameters():
        tensor = param.detach().cpu().float()
        raw_weights[name] = tensor.tolist()

    payload = {
        "model":        "limbic",
        "input_dim":    FEATURE_DIM,
        "output_dim":   OUTPUT_DIM,
        "activation":   "sigmoid",
        "weight_names": WEIGHT_NAMES,
        "weights":      raw_weights,
    }

    with open(output_path, "w") as f:
        json.dump(payload, f)

    size_kb = output_path.stat().st_size / 1024
    print(f"  Exported {output_path} ({size_kb:.1f} KB)")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_loaders(
    dataset: RegressionDataset,
    val_fraction: float = 0.2,
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader]:
    """Split *dataset* into train/val DataLoaders."""
    val_size   = max(int(len(dataset) * val_fraction), 10)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size)
    print(f"  Split: {train_size} train / {val_size} val")
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# LimbicNN training
# ---------------------------------------------------------------------------

def train_limbic(
    episodes_dir: Path = Path("training_data"),
    output_dir:   Path = Path("weights"),
    epochs:       int  = 200,
) -> None:
    """Train LimbicNN from LLM-labeled urgency-weight episodes.

    Architecture: 140 → 256 → 512 → 512 → 128 → 13.
    PolicyNN with ``hidden_layers=[256, 512, 512, 128]`` and sigmoid applied
    after the forward pass (MSELoss training, not CrossEntropyLoss).

    Args:
        episodes_dir: Directory containing ``episodes_limbic.jsonl`` produced
                      by EpisodeCollector.ts + the Precursors LLM labeler.
        output_dir:   Destination for ``limbic_nn.json`` (loaded at startup
                      by LimbicPolicyInference.ts).
        epochs:       Maximum training epochs; early stopping (patience=20) may
                      halt sooner.
    """
    limbic_file = episodes_dir / "episodes_limbic.jsonl"
    if not limbic_file.exists():
        print(f"WARNING: {limbic_file} not found — run EpisodeCollector.ts first")
        return

    arch_str = " → ".join(str(d) for d in [FEATURE_DIM] + HIDDEN_LAYERS + [OUTPUT_DIM])
    print(f"\n=== LimbicNN ({arch_str}) ===")

    dataset = RegressionDataset(limbic_file, feature_dim=FEATURE_DIM, output_dim=OUTPUT_DIM)

    if len(dataset) < MIN_SAMPLES:
        print(f"  Only {len(dataset)} samples — need {MIN_SAMPLES}+, skipping")
        return

    train_loader, val_loader = _build_loaders(dataset)

    # PolicyNN builds: Linear→LayerNorm→GELU per hidden block, Dropout after
    # block index (4-1)//2 = 1 (i.e. after the second 512-wide block).
    # The final nn.Linear produces raw pre-sigmoid scores; sigmoid is applied
    # externally (in train_regression and at inference time).
    model = PolicyNN(
        input_dim=FEATURE_DIM,
        output_dim=OUTPUT_DIM,
        hidden_layers=HIDDEN_LAYERS,
        dropout=0.1,
    )

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Trainable parameters: {total_params:,}")

    metrics = train_regression(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    export_limbic_weights_json(model, output_dir / "limbic_nn.json")

    final = metrics[-1]
    print(
        f"  Final  train_loss={final['train_loss']:.5f}  "
        f"val_loss={final['val_loss']:.5f}  "
        f"(epoch {final['epoch']})"
    )

    # Sanity check: run inference on a batch of 50 agents.
    # All 13 urgency weights should be in [0, 1] after sigmoid.
    model.eval()
    with torch.no_grad():
        x      = torch.randn(50, FEATURE_DIM)
        raw    = model(x)
        preds  = torch.sigmoid(raw)  # (50, 13)

    assert preds.shape == (50, OUTPUT_DIM), "Unexpected output shape"
    assert preds.min().item() >= 0.0 and preds.max().item() <= 1.0, \
        "Urgency weights out of [0, 1] range after sigmoid"

    per_weight_means = preds.mean(dim=0)
    print("  Sanity check (50 agents) — mean urgency weights:")
    for name, val in zip(WEIGHT_NAMES, per_weight_means.tolist()):
        print(f"    {name:<18s} {val:.3f}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Train Precursors LimbicNN (chemical state → urgency weights) "
            "with llm-distill"
        )
    )
    parser.add_argument(
        "--episodes-dir",
        type=Path,
        default=Path("training_data"),
        help=(
            "Directory containing episodes_limbic.jsonl produced by "
            "EpisodeCollector.ts (default: training_data/)"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("weights"),
        help="Where to write limbic_nn.json (default: weights/)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Maximum training epochs (default: 200; early stopping may halt sooner)",
    )
    args = parser.parse_args()

    train_limbic(args.episodes_dir, args.output_dir, args.epochs)

    print("\nDone. Load weights in TypeScript via LimbicPolicyInference.ts.")
