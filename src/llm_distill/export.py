"""
Weight export utilities for llm-distill.

Exports trained PyTorch models to a JSON format that can be consumed directly
by JavaScript/TypeScript runtimes without requiring a Python runtime at
inference time.

JSON schema
-----------
{
  "model":      "<model_name>",
  "input_dim":  <int>,
  "output_dim": <int>,
  "actions":    ["action_a", "action_b", ...],
  "weights": {
    "layer.weight": [[...], ...],   // number[][] for 2-D tensors
    "layer.bias":   [...],          // number[]   for 1-D tensors
    ...
  }
}
"""

import json
from pathlib import Path

import torch.nn as nn


def export_weights_json(
    model: nn.Module,
    model_name: str,
    actions: list[str],
    output_path: Path | str,
    feature_dim: int,
) -> None:
    """
    Serialize all named parameters of *model* to a JSON file.

    Args:
        model:       Trained PyTorch model (will not be mutated).
        model_name:  Identifier string written into the JSON (e.g. "talker").
        actions:     Ordered list of action-type strings; determines output_dim
                     and the mapping from logit index back to action name.
        output_path: Destination file path (.json).
        feature_dim: Expected input dimensionality; written as input_dim.
    """
    output_path = Path(output_path)

    weights: dict[str, list] = {}
    for name, param in model.named_parameters():
        tensor = param.detach().cpu().float()
        weights[name] = tensor.tolist()

    payload = {
        "model": model_name,
        "input_dim": feature_dim,
        "output_dim": len(actions),
        "actions": actions,
        "weights": weights,
    }

    with open(output_path, "w") as f:
        json.dump(payload, f)

    size_kb = output_path.stat().st_size / 1024
    print(f"  Exported {output_path} ({size_kb:.1f} KB)")
