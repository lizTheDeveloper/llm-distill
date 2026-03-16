# llm-distill

Distill LLM agent decisions into fast micro neural networks.

## Overview

**Problem**: LLM calls take 200-3000ms. Game agents need decisions in under 1ms.

**Solution**: Record LLM teacher decisions as training episodes, train small policy networks offline, then deploy the micro-NNs for real-time inference — no LLM required at runtime.

**Based on**:
- Policy Distillation (arXiv:1511.06295)
- LLM4Teach (arXiv:2311.13373)
- Talker-Reasoner dual-process architecture (arXiv:2410.08328)

## Install

```
pip install llm-distill
```

Or from source:

```
pip install -e .
```

## Quick Start

```python
from llm_distill import PolicyNN, EpisodeDataset, train_model, export_weights_json

# Define the action space
actions = ["gather", "build", "farm", "idle", "explore"]

# Create the policy network
model = PolicyNN(input_dim=40, output_dim=len(actions))

# Load recorded LLM decisions
dataset = EpisodeDataset("data/episodes.jsonl", actions=actions)

# Train
train_model(model, dataset, epochs=50)

# Export for JS/TS runtime
export_weights_json(model, actions, path="weights.json")
```

## CLI Usage

```
llm-distill-train \
  --episodes data/episodes.jsonl \
  --actions "gather,build,farm,idle,explore" \
  --output weights.json
```

## Training Data Format

JSONL file, one record per decision:

```json
{"feature_vector": [0.1, 0.2, 0.7, ...], "action_type": "gather"}
{"feature_vector": [0.4, 0.9, 0.1, ...], "action_type": "build"}
```

Each `feature_vector` should have the same dimensionality as the `input_dim` configured for your `PolicyNN`. The `action_type` must be one of the actions passed to the dataset loader.

## Architecture

PolicyNN uses configurable Linear + LayerNorm + GELU blocks:

```
input (40-dim)
  → Linear(40, 128) + LayerNorm + GELU
  → Linear(128, 256) + LayerNorm + GELU
  → Linear(256, 128) + LayerNorm + GELU
  → Linear(128, N_actions)
  → softmax
```

- Typical parameter count: 27K-280K depending on hidden dims and depth
- Inference time: under 0.5ms for 50 concurrent agents on CPU
- No external dependencies at inference time once weights are exported

## RLAIF Labeling (Advanced)

The package supports Constitutional AI / RLAIF labeling patterns. A teacher LLM (e.g., via Ollama) evaluates agent state observations and produces labeled decisions suitable for distillation training. This allows you to generate training data without a live production LLM at scale.

See `examples/` for a reference implementation using the RLAIF labeler and validator scripts.

## Weight Export

Weights are exported as JSON compatible with TypeScript and JavaScript runtimes:

```json
{
  "actions": ["gather", "build", "farm", "idle", "explore"],
  "layers": [
    {"weight": [...], "bias": [...], "norm_weight": [...], "norm_bias": [...]},
    ...
  ]
}
```

The companion TypeScript runtime uses pre-allocated `Float32Array` buffers for GC-free inference, keeping frame times stable during gameplay.

## License

MIT

## Credits

Extracted from the [Multiverse Games MVEE](https://github.com/multiverse-games/mvee) project. Inspired by the emergent creature AI of the Creatures franchise.
