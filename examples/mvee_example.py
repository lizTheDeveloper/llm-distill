"""
Example: Distilling MVEE agent decisions with llm-distill.

MVEE uses three decision layers:
- Autonomic: moment-to-moment survival (highest frequency, rule-based, no LLM)
- Talker:    social interactions (LLM-driven, distilled into TalkerNN)
- Executor:  task planning      (LLM-driven, distilled into ExecutorNN)

This example shows how to configure and train a TalkerNN and ExecutorNN using
the llm-distill library, replacing the original hardcoded class definitions with
a single configurable PolicyNN.

Data pipeline recap
-------------------
1. Run the game — LLM decisions are logged to
   custom_game_engine/logs/llm-prompts/llm-prompts-*.jsonl
2. Run parse_prompt_logs.py to convert those logs to labelled JSONL episodes:
       python3 training/parse_prompt_logs.py
   Outputs: training_data/episodes_talker.jsonl
            training_data/episodes_executor.jsonl
3. Run this script (or train.py) to train the NNs and export JSON weights.
4. The TypeScript runtime loads weights/talker_nn.json and
   weights/executor_nn.json at startup (see MVEEPolicyInference.ts).

Refs
----
- Policy Distillation (Rusu et al., 2015)  arXiv:1511.06295
- DAgger            (Ross et al., 2011)    arXiv:1011.0686
- LLM4Teach         (Fu   et al., 2023)    arXiv:2311.13373
"""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from llm_distill import EpisodeDataset, PolicyNN, export_weights_json, train_model

# ---------------------------------------------------------------------------
# Action spaces — must stay in sync with MVEEPolicyInference.ts
# ---------------------------------------------------------------------------

TALKER_ACTIONS = [
    "talk",
    "call_meeting",
    "set_personal_goal",
    "set_medium_term_goal",
    "set_group_goal",
    "follow_agent",
]

EXECUTOR_ACTIONS = [
    "gather",
    "till",
    "plan_build",
    "build",
    "farm",
    "help",
    "deposit_items",
    "idle",
    "explore",
    "plant",
    "set_priorities",
    "pick",
    "wander",
]

# 40-dim feature vector extracted from LLM prompts by feature_extractor.py.
# Layout: skills[0-5], priorities[6-9], env_resources[10-13],
#         village_state[14-18], behavior[19], faith[20], health[21-22],
#         perception[23-24], mood/emotions[25-29], conversation[30],
#         goals[31], inventory[32-35], layer_flag[36], position[37-38],
#         has_memories[39].
FEATURE_DIM = 40

# Minimum sample threshold — skip training if we have fewer than this many
# labelled episodes (avoids fitting noise on a near-empty dataset).
MIN_SAMPLES = 50


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_loaders(
    dataset: EpisodeDataset,
    val_fraction: float = 0.2,
    batch_size: int = 64,
) -> tuple[DataLoader, DataLoader]:
    """Split dataset into train/val DataLoaders."""
    val_size = max(int(len(dataset) * val_fraction), 10)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size)
    print(f"  Split: {train_size} train / {val_size} val")
    return train_loader, val_loader


# ---------------------------------------------------------------------------
# TalkerNN training
# ---------------------------------------------------------------------------

def train_talker(
    episodes_dir: Path = Path("training_data"),
    output_dir: Path = Path("weights"),
    epochs: int = 150,
) -> None:
    """Train TalkerNN from labelled Talker episodes.

    Architecture mirrors the original TalkerNN: 40 → 128 → 256 → 128 → 6.
    PolicyNN auto-selects [128, 256, 128] hidden layers when output_dim <= 10.

    Args:
        episodes_dir: Directory containing episodes_talker.jsonl (produced by
                      parse_prompt_logs.py).
        output_dir:   Destination for talker_nn.json (loaded by the TS runtime).
        epochs:       Maximum training epochs; early stopping (patience=20) may
                      halt sooner.
    """
    talker_file = episodes_dir / "episodes_talker.jsonl"
    if not talker_file.exists():
        print(f"WARNING: {talker_file} not found — run parse_prompt_logs.py first")
        return

    action_index = {a: i for i, a in enumerate(TALKER_ACTIONS)}

    print(f"\n=== TalkerNN (40 → 128 → 256 → 128 → {len(TALKER_ACTIONS)}) ===")
    dataset = EpisodeDataset(talker_file, action_index, FEATURE_DIM)

    if len(dataset) < MIN_SAMPLES:
        print(f"  Only {len(dataset)} samples — need {MIN_SAMPLES}+, skipping")
        return

    train_loader, val_loader = _build_loaders(dataset)

    # PolicyNN with explicit hidden_layers replicates the original TalkerNN
    # exactly. Omitting hidden_layers lets the library auto-select the same
    # widths ([128, 256, 128]) because output_dim=6 <= 10.
    model = PolicyNN(
        input_dim=FEATURE_DIM,
        output_dim=len(TALKER_ACTIONS),
        hidden_layers=[128, 256, 128],
    )

    # Inverse-frequency weights compensate for rare actions like call_meeting
    # that appear far less often than talk in natural play logs.
    class_weights = dataset.compute_class_weights(len(TALKER_ACTIONS))
    print(f"  Class weights: {[f'{w:.2f}' for w in class_weights.tolist()]}")

    metrics = train_model(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        class_weights=class_weights,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    export_weights_json(
        model,
        model_name="talker",
        actions=TALKER_ACTIONS,
        output_path=output_dir / "talker_nn.json",
        feature_dim=FEATURE_DIM,
    )

    final = metrics[-1]
    print(f"  Final val_acc: {final['val_acc']:.3f}  val_loss: {final['val_loss']:.4f}")

    # Quick inference sanity-check: batch of 50 agents should run in <0.5 ms.
    model.eval()
    x = torch.randn(50, FEATURE_DIM)
    action_indices, confidences = model.predict(x)
    actions_taken = [TALKER_ACTIONS[i] for i in action_indices.tolist()]
    print(f"  Sanity check (50 agents): {set(actions_taken)}")


# ---------------------------------------------------------------------------
# ExecutorNN training
# ---------------------------------------------------------------------------

def train_executor(
    episodes_dir: Path = Path("training_data"),
    output_dir: Path = Path("weights"),
    epochs: int = 150,
) -> None:
    """Train ExecutorNN from labelled Executor episodes.

    Architecture mirrors the original ExecutorNN: 40 → 256 → 512 → 256 → 13.
    PolicyNN auto-selects [256, 512, 256] hidden layers when output_dim > 10.

    Args:
        episodes_dir: Directory containing episodes_executor.jsonl.
        output_dir:   Destination for executor_nn.json.
        epochs:       Maximum training epochs.
    """
    executor_file = episodes_dir / "episodes_executor.jsonl"
    if not executor_file.exists():
        print(f"WARNING: {executor_file} not found — run parse_prompt_logs.py first")
        return

    action_index = {a: i for i, a in enumerate(EXECUTOR_ACTIONS)}

    print(f"\n=== ExecutorNN (40 → 256 → 512 → 256 → {len(EXECUTOR_ACTIONS)}) ===")
    dataset = EpisodeDataset(executor_file, action_index, FEATURE_DIM)

    if len(dataset) < MIN_SAMPLES:
        print(f"  Only {len(dataset)} samples — need {MIN_SAMPLES}+, skipping")
        return

    train_loader, val_loader = _build_loaders(dataset)

    # Executor has 13 actions (output_dim > 10), so the library auto-selects
    # [256, 512, 256] — matching the original ExecutorNN. Passing hidden_layers
    # explicitly here makes the architecture unambiguous.
    model = PolicyNN(
        input_dim=FEATURE_DIM,
        output_dim=len(EXECUTOR_ACTIONS),
        hidden_layers=[256, 512, 256],
    )

    class_weights = dataset.compute_class_weights(len(EXECUTOR_ACTIONS))
    print(f"  Class weights: {[f'{w:.2f}' for w in class_weights.tolist()]}")

    metrics = train_model(
        model,
        train_loader,
        val_loader,
        epochs=epochs,
        class_weights=class_weights,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    export_weights_json(
        model,
        model_name="executor",
        actions=EXECUTOR_ACTIONS,
        output_path=output_dir / "executor_nn.json",
        feature_dim=FEATURE_DIM,
    )

    final = metrics[-1]
    print(f"  Final val_acc: {final['val_acc']:.3f}  val_loss: {final['val_loss']:.4f}")

    model.eval()
    x = torch.randn(50, FEATURE_DIM)
    action_indices, confidences = model.predict(x)
    actions_taken = [EXECUTOR_ACTIONS[i] for i in action_indices.tolist()]
    print(f"  Sanity check (50 agents): {set(actions_taken)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train MVEE TalkerNN and ExecutorNN with llm-distill"
    )
    parser.add_argument(
        "--episodes-dir",
        type=Path,
        default=Path("training_data"),
        help="Directory produced by parse_prompt_logs.py (default: training_data/)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("weights"),
        help="Where to write *_nn.json weight files (default: weights/)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=150,
        help="Maximum training epochs per model (default: 150)",
    )
    parser.add_argument(
        "--talker-only",
        action="store_true",
        help="Train only TalkerNN",
    )
    parser.add_argument(
        "--executor-only",
        action="store_true",
        help="Train only ExecutorNN",
    )
    args = parser.parse_args()

    if args.executor_only:
        train_executor(args.episodes_dir, args.output_dir, args.epochs)
    elif args.talker_only:
        train_talker(args.episodes_dir, args.output_dir, args.epochs)
    else:
        train_talker(args.episodes_dir, args.output_dir, args.epochs)
        train_executor(args.episodes_dir, args.output_dir, args.epochs)

    print("\nDone. Load weights in TypeScript via MVEEPolicyInference.ts.")
