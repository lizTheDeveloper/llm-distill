"""
Minimal test suite for the llm-distill package.

Run with:
    cd /path/to/llm-distill
    pytest tests/test_llm_distill.py -v
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from llm_distill import EpisodeDataset, PolicyNN, export_weights_json, train_model
from llm_distill.model import count_params


# ---------------------------------------------------------------------------
# PolicyNN — forward pass
# ---------------------------------------------------------------------------


def test_policy_nn_forward():
    """Output shape matches (batch, output_dim) for a default architecture."""
    model = PolicyNN(input_dim=40, output_dim=6)
    x = torch.randn(4, 40)
    out = model(x)
    assert out.shape == (4, 6), f"Expected (4, 6), got {out.shape}"


# ---------------------------------------------------------------------------
# PolicyNN — predict helper
# ---------------------------------------------------------------------------


def test_policy_nn_predict():
    """predict() returns action_indices of shape (batch,) and confidences in [0, 1]."""
    model = PolicyNN(40, 6)
    x = torch.randn(4, 40)
    action_indices, confidences = model.predict(x)

    assert action_indices.shape == (4,), (
        f"Expected action_indices shape (4,), got {action_indices.shape}"
    )
    assert confidences.shape == (4,), (
        f"Expected confidences shape (4,), got {confidences.shape}"
    )
    assert (confidences >= 0.0).all() and (confidences <= 1.0).all(), (
        f"Confidences must be in [0, 1], got min={confidences.min():.4f} "
        f"max={confidences.max():.4f}"
    )


# ---------------------------------------------------------------------------
# PolicyNN — custom hidden layers
# ---------------------------------------------------------------------------


def test_policy_nn_custom_hidden():
    """Custom hidden_layers=[64, 128, 64] produces correct output shape."""
    model = PolicyNN(40, 10, hidden_layers=[64, 128, 64])
    x = torch.randn(4, 40)
    out = model(x)
    assert out.shape == (4, 10), f"Expected (4, 10), got {out.shape}"


# ---------------------------------------------------------------------------
# count_params
# ---------------------------------------------------------------------------


def test_count_params():
    """count_params returns a positive integer for a standard PolicyNN."""
    model = PolicyNN(40, 6)
    n = count_params(model)
    assert isinstance(n, int), f"Expected int, got {type(n)}"
    assert n > 0, f"Expected positive param count, got {n}"


# ---------------------------------------------------------------------------
# EpisodeDataset — basic loading
# ---------------------------------------------------------------------------


_ACTION_INDEX = {
    "talk": 0,
    "move": 1,
    "forage": 2,
    "rest": 3,
}

_ACTIONS = list(_ACTION_INDEX.keys())


def _make_jsonl(path: Path, n: int = 20, feature_dim: int = 40) -> None:
    """Write *n* well-formed episode lines to *path*."""
    import random

    random.seed(42)
    with open(path, "w") as f:
        for i in range(n):
            record = {
                "feature_vector": [random.random() for _ in range(feature_dim)],
                "action_type": _ACTIONS[i % len(_ACTIONS)],
            }
            f.write(json.dumps(record) + "\n")


def test_episode_dataset():
    """EpisodeDataset loads 20 episodes; __getitem__ returns correct tensor shapes."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        _make_jsonl(tmp_path, n=20, feature_dim=40)

        ds = EpisodeDataset(
            filepath=tmp_path,
            action_index=_ACTION_INDEX,
            feature_dim=40,
        )

        assert len(ds) == 20, f"Expected 20 samples, got {len(ds)}"

        features, label = ds[0]
        assert features.shape == (40,), (
            f"Expected feature tensor shape (40,), got {features.shape}"
        )
        assert features.dtype == torch.float32, (
            f"Expected float32 features, got {features.dtype}"
        )
        assert label.shape == (), (
            f"Expected scalar label tensor, got shape {label.shape}"
        )
        assert label.dtype == torch.long, (
            f"Expected long label, got {label.dtype}"
        )
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# EpisodeDataset — class weights
# ---------------------------------------------------------------------------


def test_class_weights():
    """compute_class_weights returns a positive tensor of length num_classes."""
    with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Imbalanced: 10 "talk", 5 "move", 3 "forage", 2 "rest"
        records = (
            [{"feature_vector": [0.0] * 40, "action_type": "talk"}] * 10
            + [{"feature_vector": [0.0] * 40, "action_type": "move"}] * 5
            + [{"feature_vector": [0.0] * 40, "action_type": "forage"}] * 3
            + [{"feature_vector": [0.0] * 40, "action_type": "rest"}] * 2
        )
        with open(tmp_path, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

        ds = EpisodeDataset(
            filepath=tmp_path,
            action_index=_ACTION_INDEX,
            feature_dim=40,
        )

        num_classes = len(_ACTION_INDEX)
        weights = ds.compute_class_weights(num_classes)

        assert weights.shape == (num_classes,), (
            f"Expected weights shape ({num_classes},), got {weights.shape}"
        )
        assert (weights > 0).all(), (
            f"All class weights must be positive, got {weights}"
        )
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# export_weights_json
# ---------------------------------------------------------------------------


def test_export_weights_json():
    """export_weights_json writes a JSON file with the expected top-level keys."""
    model = PolicyNN(40, 6)
    actions = ["talk", "move", "forage", "rest", "build", "idle"]

    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        export_weights_json(model, "test_model", actions, tmp_path, 40)

        with open(tmp_path) as f:
            data = json.load(f)

        required_keys = {"model", "input_dim", "output_dim", "actions", "weights"}
        missing = required_keys - data.keys()
        assert not missing, f"JSON missing keys: {missing}"
    finally:
        tmp_path.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# train_model — smoke test
# ---------------------------------------------------------------------------


def test_train_model_smoke():
    """train_model runs for 5 epochs and returns per-epoch metric dicts."""
    input_dim, output_dim, n_samples = 10, 3, 100

    features = torch.randn(n_samples, input_dim)
    labels = torch.randint(0, output_dim, (n_samples,))

    train_ds = TensorDataset(features[:80], labels[:80])
    val_ds = TensorDataset(features[80:], labels[80:])
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16)

    model = PolicyNN(input_dim, output_dim, hidden_layers=[32, 64, 32])

    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=5,
    )

    assert isinstance(history, list), f"Expected list, got {type(history)}"
    assert len(history) == 5, f"Expected 5 epoch records, got {len(history)}"

    expected_keys = {"epoch", "train_loss", "train_acc", "val_loss", "val_acc"}
    for i, record in enumerate(history):
        missing = expected_keys - record.keys()
        assert not missing, f"Epoch {i} record missing keys: {missing}"
