"""
llm-distill: Distill LLM agent decisions into fast micro-NNs for real-time inference.
"""

__version__ = "0.1.0"

from llm_distill.model import PolicyNN
from llm_distill.dataset import EpisodeDataset
from llm_distill.train import train_model
from llm_distill.export import export_weights_json

__all__ = [
    "PolicyNN",
    "EpisodeDataset",
    "train_model",
    "export_weights_json",
    "__version__",
]
