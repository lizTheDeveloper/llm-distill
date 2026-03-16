"""
llm_distill.model — Configurable policy networks for LLM decision distillation.

This module is part of the llm-distill package, which distills LLM decisions into
lightweight micro-neural-networks suitable for real-time game AI inference.

The core idea: run an LLM oracle to label agent decisions (e.g. "talk", "move",
"forage"), train a small PolicyNN on those labels, and deploy the PolicyNN at
game speed without LLM latency or cost.

Theoretical grounding
---------------------
- Policy Distillation (Rusu et al., 2015) — arXiv:1511.06295
  Introduces distilling a teacher policy into a compact student network via
  soft-target KL divergence, which is the training objective used here.

- LLM4Teach (Fu et al., 2023) — arXiv:2311.13373
  Demonstrates LLMs as teacher oracles for RL agents, showing strong
  sample-efficiency gains when the student is trained on LLM-generated labels
  rather than environment reward alone.

Original architecture replaced
-------------------------------
The original MVEE codebase contained separate, hardcoded networks per agent
role, e.g.::

    class TalkerNN(nn.Module):          # 40 → 128 → 256 → 128 → 6
    class ExecutorNN(nn.Module):        # 64 → 256 → 512 → 256 → 12

``PolicyNN`` generalises both into a single configurable class so that new
agent roles require only a change in constructor arguments rather than a new
class definition.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Default hidden-layer heuristic
# ---------------------------------------------------------------------------

def _default_hidden_layers(input_dim: int, output_dim: int) -> list[int]:
    """Return a reasonable hidden-layer width sequence for the given I/O dims.

    Small models  (output_dim <= 10): input → 128 → 256 → 128 → output
    Larger models (output_dim  > 10): input → 256 → 512 → 256 → output
    """
    if output_dim > 10:
        return [256, 512, 256]
    return [128, 256, 128]


# ---------------------------------------------------------------------------
# PolicyNN
# ---------------------------------------------------------------------------

class PolicyNN(nn.Module):
    """Configurable feed-forward policy network for LLM decision distillation.

    Architecture pattern (mirrors the original TalkerNN / ExecutorNN):
        For each hidden layer: Linear → LayerNorm → GELU
        Dropout(p) is inserted after the *middle* hidden block.
        Final layer: Linear (no activation — returns raw logits).

    Parameters
    ----------
    input_dim : int
        Dimensionality of the observation / feature vector.
    output_dim : int
        Number of action classes (logit outputs).
    hidden_layers : list[int] or None
        Explicit list of hidden-layer widths, e.g. ``[128, 256, 128]``.
        If ``None``, widths are auto-computed via :func:`_default_hidden_layers`.
    dropout : float
        Dropout probability applied after the middle hidden block.
        Set to ``0.0`` to disable dropout entirely.

    Class Attributes
    ----------------
    INPUT_DIM : int
        Stored value of ``input_dim`` for introspection.
    OUTPUT_DIM : int
        Stored value of ``output_dim`` for introspection.

    Examples
    --------
    Replicate the original TalkerNN (40 → 128 → 256 → 128 → 6)::

        talker = PolicyNN(input_dim=40, output_dim=6)

    Replicate the original ExecutorNN (64 → 256 → 512 → 256 → 12)::

        executor = PolicyNN(input_dim=64, output_dim=12)

    Custom architecture::

        custom = PolicyNN(input_dim=128, output_dim=4, hidden_layers=[64, 64], dropout=0.2)
    """

    INPUT_DIM: int
    OUTPUT_DIM: int

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_layers: Optional[list[int]] = None,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.INPUT_DIM = input_dim
        self.OUTPUT_DIM = output_dim

        if hidden_layers is None:
            hidden_layers = _default_hidden_layers(input_dim, output_dim)

        if len(hidden_layers) == 0:
            raise ValueError("hidden_layers must contain at least one width.")

        # Index of the middle block after which dropout is inserted.
        # For an odd-length sequence this is the true midpoint; for even-length
        # it falls on the lower-middle index (matches original architecture).
        dropout_after = (len(hidden_layers) - 1) // 2

        layers: list[nn.Module] = []
        prev_dim = input_dim

        for i, width in enumerate(hidden_layers):
            layers += [
                nn.Linear(prev_dim, width),
                nn.LayerNorm(width),
                nn.GELU(),
            ]
            if i == dropout_after and dropout > 0.0:
                layers.append(nn.Dropout(p=dropout))
            prev_dim = width

        layers.append(nn.Linear(prev_dim, output_dim))

        self.net = nn.Sequential(*layers)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute raw logits for input ``x``.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(..., INPUT_DIM)``.

        Returns
        -------
        torch.Tensor
            Raw logits of shape ``(..., OUTPUT_DIM)``.
        """
        return self.net(x)

    # ------------------------------------------------------------------
    # Inference helper
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return greedy action indices and their softmax confidences.

        Runs under ``torch.no_grad()`` — intended for inference, not training.

        Parameters
        ----------
        x : torch.Tensor
            Shape ``(..., INPUT_DIM)``.

        Returns
        -------
        action_indices : torch.Tensor
            Integer tensor of shape ``(...)`` with the argmax class per sample.
        confidences : torch.Tensor
            Float tensor of shape ``(...)`` with the softmax probability of the
            chosen action (i.e. ``softmax(logits).max(-1)``).
        """
        logits = self.forward(x)
        probs = F.softmax(logits, dim=-1)
        confidences, action_indices = probs.max(dim=-1)
        return action_indices, confidences


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def count_params(model: nn.Module) -> int:
    """Return the total number of trainable parameters in *model*.

    Parameters
    ----------
    model : nn.Module
        Any PyTorch module.

    Returns
    -------
    int
        Sum of ``p.numel()`` for all parameters where ``p.requires_grad``.

    Examples
    --------
    ::

        net = PolicyNN(40, 6)
        print(f"{count_params(net):,} trainable parameters")
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
