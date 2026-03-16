"""
llm_distill.cli — Entry points for the llm-distill command-line tools.

Entry points registered in pyproject.toml
------------------------------------------
  llm-distill-train  →  llm_distill.cli:train_cli
  llm-distill-parse  →  llm_distill.cli:parse_cli
  llm-distill-label  →  llm_distill.cli:label_cli

The parse and label entry points are intentionally left as application-level
stubs: feature extraction and RLAIF labeling are game-specific concerns.
See ``examples/`` for the reference MVEE implementations.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from llm_distill.dataset import EpisodeDataset
from llm_distill.export import export_weights_json
from llm_distill.model import PolicyNN, count_params
from llm_distill.train import benchmark_inference, train_model


# ---------------------------------------------------------------------------
# train_cli
# ---------------------------------------------------------------------------

def train_cli() -> None:
    """Argparse CLI wrapping :func:`train_model`.

    Reads a JSONL episodes file, builds a :class:`PolicyNN`, trains it, exports
    weights to JSON, and prints an inference benchmark.

    Example
    -------
    ::

        llm-distill-train \\
            --episodes training_data/episodes.jsonl \\
            --actions talk,move,forage,build \\
            --feature-dim 40 \\
            --output weights/policy.json \\
            --epochs 150 \\
            --batch-size 64 \\
            --val-split 0.2
    """
    parser = argparse.ArgumentParser(
        prog="llm-distill-train",
        description="Train a PolicyNN from LLM-labelled episode data.",
    )
    parser.add_argument(
        "--episodes",
        required=True,
        metavar="PATH",
        help="Path to JSONL episodes file. Each line: {feature_vector: [...], action_type: '...'}",
    )
    parser.add_argument(
        "--actions",
        required=True,
        metavar="A,B,C",
        help="Comma-separated ordered list of action names (defines output classes).",
    )
    parser.add_argument(
        "--feature-dim",
        type=int,
        default=40,
        metavar="N",
        help="Length of each feature_vector (default: 40).",
    )
    parser.add_argument(
        "--hidden-layers",
        default=None,
        metavar="W1,W2,...",
        help=(
            "Comma-separated hidden layer widths, e.g. '128,256,128'. "
            "If omitted, widths are auto-selected based on output_dim."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="PATH",
        help="Destination path for the exported weights JSON file.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        metavar="N",
        help="Maximum training epochs (default: 150).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="Mini-batch size (default: 64).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        metavar="LR",
        help="Initial Adam learning rate (default: 1e-3).",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=20,
        metavar="N",
        help="Early-stopping patience in epochs (default: 20).",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        metavar="F",
        help="Fraction of data reserved for validation (default: 0.2).",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Parse action list
    # ------------------------------------------------------------------ #
    actions: list[str] = [a.strip() for a in args.actions.split(",") if a.strip()]
    if len(actions) < 2:
        print("error: --actions must contain at least two comma-separated names.", file=sys.stderr)
        sys.exit(1)
    action_index: dict[str, int] = {name: i for i, name in enumerate(actions)}

    # ------------------------------------------------------------------ #
    # Parse optional hidden layers
    # ------------------------------------------------------------------ #
    hidden_layers: list[int] | None = None
    if args.hidden_layers is not None:
        try:
            hidden_layers = [int(w.strip()) for w in args.hidden_layers.split(",") if w.strip()]
        except ValueError:
            print("error: --hidden-layers must be comma-separated integers.", file=sys.stderr)
            sys.exit(1)

    # ------------------------------------------------------------------ #
    # Load dataset
    # ------------------------------------------------------------------ #
    episodes_path = Path(args.episodes)
    if not episodes_path.exists():
        print(f"error: episodes file not found: {episodes_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading episodes from {episodes_path} ...")
    dataset = EpisodeDataset(
        filepath=episodes_path,
        action_index=action_index,
        feature_dim=args.feature_dim,
    )

    if len(dataset) < 50:
        print(
            f"error: only {len(dataset)} valid samples found — need at least 50 to train.",
            file=sys.stderr,
        )
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Train/val split
    # ------------------------------------------------------------------ #
    val_size = max(int(len(dataset) * args.val_split), 10)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    print(f"Training on {train_size} samples, validating on {val_size}")

    # ------------------------------------------------------------------ #
    # Build model
    # ------------------------------------------------------------------ #
    model = PolicyNN(
        input_dim=args.feature_dim,
        output_dim=len(actions),
        hidden_layers=hidden_layers,
    )
    print(f"PolicyNN: {count_params(model):,} trainable parameters")

    class_weights = dataset.compute_class_weights(len(actions))
    print(f"Class weights: {[f'{w:.2f}' for w in class_weights.tolist()]}")

    # ------------------------------------------------------------------ #
    # Select device (prefer MPS on Apple Silicon, then CUDA, else CPU)
    # ------------------------------------------------------------------ #
    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Device: {device}")

    # ------------------------------------------------------------------ #
    # Train
    # ------------------------------------------------------------------ #
    metrics = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        device=device,
        class_weights=class_weights,
    )

    final = metrics[-1]
    print(
        f"Final: val_acc={final['val_acc']:.3f}  val_loss={final['val_loss']:.4f}  "
        f"(epoch {final['epoch']})"
    )

    # ------------------------------------------------------------------ #
    # Export weights
    # ------------------------------------------------------------------ #
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Derive a model name from the output filename stem
    model_name = output_path.stem

    model.cpu()
    export_weights_json(
        model=model,
        model_name=model_name,
        actions=actions,
        output_path=output_path,
        feature_dim=args.feature_dim,
    )

    # Also save training metrics alongside the weights
    metrics_path = output_path.with_suffix(".metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved to {metrics_path}")

    # ------------------------------------------------------------------ #
    # Inference benchmark
    # ------------------------------------------------------------------ #
    ms = benchmark_inference(
        model=model,
        batch_size=50,
        feature_dim=args.feature_dim,
    )
    print(f"Inference: {ms:.3f} ms per batch (50 agents, CPU)")


# ---------------------------------------------------------------------------
# parse_cli — application-specific stub
# ---------------------------------------------------------------------------

def parse_cli() -> None:
    """Stub entry point for log parsing.

    Log parsing is application-specific.  See examples/ for MVEE implementation.
    """
    print("Log parsing is application-specific. See examples/ for MVEE implementation.")


# ---------------------------------------------------------------------------
# label_cli — application-specific stub
# ---------------------------------------------------------------------------

def label_cli() -> None:
    """Stub entry point for RLAIF labeling.

    RLAIF labeling is application-specific.  See examples/ for MVEE implementation.
    """
    print("RLAIF labeling is application-specific. See examples/ for MVEE implementation.")
