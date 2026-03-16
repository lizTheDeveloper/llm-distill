"""
EpisodeDataset — generalized JSONL episode loader for llm-distill.

Each line in the JSONL file must have at minimum:
  {"feature_vector": [...], "action_type": "gather", ...}

Unknown action types and lines whose feature_vector length does not match
feature_dim are silently skipped; a load-stats summary is printed on
construction.
"""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class EpisodeDataset(Dataset):
    """
    Loads parsed episode JSONL files into (feature_vector, action_class) tensors.

    Args:
        filepath:     Path to the .jsonl file.
        action_index: Mapping from action-type string to integer class label.
        feature_dim:  Expected length of each feature_vector. Lines that do not
                      match this dimension are skipped.
    """

    def __init__(
        self,
        filepath: Path | str,
        action_index: dict[str, int],
        feature_dim: int,
    ) -> None:
        filepath = Path(filepath)
        self.feature_dim = feature_dim
        self.samples: list[tuple[list[float], int]] = []
        self.label_counts: dict[int, int] = {}

        loaded = 0
        skipped_dim = 0
        skipped_action = 0
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
                at = ep.get("action_type", "")

                if len(fv) != feature_dim:
                    skipped_dim += 1
                    continue

                if at not in action_index:
                    skipped_action += 1
                    continue

                label = action_index[at]
                self.samples.append((fv, label))
                self.label_counts[label] = self.label_counts.get(label, 0) + 1
                loaded += 1

        total_skipped = skipped_dim + skipped_action + skipped_parse
        print(
            f"  Loaded {loaded} samples, skipped {total_skipped} "
            f"(dim_mismatch={skipped_dim}, unknown_action={skipped_action}, "
            f"parse_error={skipped_parse})"
        )

    def compute_class_weights(self, num_classes: int) -> torch.Tensor:
        """
        Return inverse-frequency class weights as a 1-D tensor of length
        num_classes.  Classes with zero samples receive weight 1.0 (same as
        a class with every sample) rather than inf.
        """
        counts = torch.zeros(num_classes)
        for label, count in self.label_counts.items():
            if label < num_classes:
                counts[label] = count
        counts = torch.clamp(counts, min=1.0)
        weights = counts.sum() / (num_classes * counts)
        return weights

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        fv, label = self.samples[idx]
        return (
            torch.tensor(fv, dtype=torch.float32),
            torch.tensor(label, dtype=torch.long),
        )
