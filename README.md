# llm-distill

Distill LLM agent decisions into fast micro neural networks.

## Overview

**Problem**: LLM calls take 200-3000ms. Game agents need decisions in under 1ms.

**Solution**: Record LLM teacher decisions as training episodes, train small policy networks offline, then deploy the micro-NNs for real-time inference — no LLM required at runtime.

The full pipeline:

1. **Instrument** — wrap LLM calls with `EpisodeLogger` to collect `(feature_vector, action, confidence)` tuples
2. **Train** — feed the JSONL episodes to `PolicyNN` + `train_model` (PyTorch, offline)
3. **Export** — serialize weights to JSON with `export_weights_json`
4. **Deploy** — load JSON in `NNPolicy` (TypeScript, zero dependencies, browser-safe)

## Install

### Python (training)

```
pip install llm-distill
```

Or from source:

```
pip install -e .
```

### TypeScript (runtime inference)

Copy `ts/NNPolicy.ts` and/or `ts/EpisodeLogger.ts` into your project. Both are self-contained with zero dependencies.

## Quick Start

### 1. Collect episodes (TypeScript)

```typescript
import { EpisodeLogger } from './EpisodeLogger';

const logger = new EpisodeLogger({ flushEndpoint: '/api/training/episodes' });
logger.setEnabled(true);

// After each LLM call:
const features = extractFeatures(agentState);  // your feature extractor
const action = llmResponse.action;              // LLM-chosen action
logger.log(features, action, llmResponse.confidence);
```

### 2. Train (Python)

```python
from llm_distill import PolicyNN, EpisodeDataset, train_model, export_weights_json
from torch.utils.data import DataLoader, random_split

actions = ["gather", "build", "farm", "idle", "explore"]
action_index = {a: i for i, a in enumerate(actions)}

dataset = EpisodeDataset("episodes.jsonl", action_index=action_index, feature_dim=40)

val_size = max(int(len(dataset) * 0.2), 10)
train_ds, val_ds = random_split(dataset, [len(dataset) - val_size, val_size])
train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=64)

model = PolicyNN(input_dim=40, output_dim=len(actions))
train_model(model, train_loader, val_loader, epochs=150)

export_weights_json(model, "policy", actions, "weights.json", feature_dim=40)
```

### 3. Deploy (TypeScript)

```typescript
import { NNPolicy } from './NNPolicy';

const policy = await NNPolicy.fromURL('/models/weights.json');

// Every game tick — zero allocation, <0.5ms for 100 agents:
const result = policy.predict(featureVector);
console.log(result.action, result.confidence);

// Use confidence for LLM fallback (Talker-Reasoner pattern):
if (result.confidence < 0.85) {
  // Fall back to LLM for this decision
}
```

## CLI Usage

```
llm-distill-train \
  --episodes data/episodes.jsonl \
  --actions "gather,build,farm,idle,explore" \
  --feature-dim 40 \
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
  -> Linear(40, 128) + LayerNorm + GELU
  -> Linear(128, 256) + LayerNorm + GELU
  -> Linear(256, 128) + LayerNorm + GELU
  -> Linear(128, N_actions)
  -> softmax (at inference)
```

- Typical parameter count: 27K-280K depending on hidden dims and depth
- Inference time: under 0.5ms for 50 concurrent agents on CPU
- TypeScript runtime: zero dependencies, pre-allocated Float32Array scratch buffers

## TypeScript Runtime

The `ts/` directory contains two self-contained TypeScript files with zero dependencies:

### NNPolicy — inference runtime

Loads exported JSON weights and runs a forward pass using pre-allocated `Float32Array` scratch buffers. Architecture is auto-detected from weight keys — works with any `PolicyNN` configuration.

```typescript
import { NNPolicy } from './NNPolicy';

// Load from URL (browser) or file (Node)
const policy = await NNPolicy.fromURL('/models/talker.json');

// Classification (softmax + argmax)
const { action, actionIndex, confidence } = policy.predict(features);

// Raw logits (for custom post-processing)
const logits = policy.forward(features);
```

### EpisodeLogger — data collection middleware

Framework-agnostic middleware for logging LLM decisions as JSONL training episodes. Works in browsers and Node.

```typescript
import { EpisodeLogger } from './EpisodeLogger';

const logger = new EpisodeLogger({
  flushThreshold: 50,
  flushEndpoint: '/api/training/episodes',
});
logger.setEnabled(true);

// Log after each LLM decision:
logger.log(featureVector, actionType, confidence, { agentId: 'norn-42' });

// Manual export (when no endpoint configured):
const jsonl = logger.exportJSONL();
```

## RLAIF Labeling (Advanced)

The package supports Constitutional AI / RLAIF labeling patterns. A teacher LLM (e.g., via Ollama) evaluates agent state observations and produces labeled decisions suitable for distillation training. This allows you to generate training data without a live production LLM at scale.

See `examples/` for reference implementations.

## Weight Export

Weights are exported as JSON compatible with the TypeScript `NNPolicy` runtime:

```json
{
  "model": "talker",
  "input_dim": 40,
  "output_dim": 6,
  "actions": ["talk", "call_meeting", "set_personal_goal", ...],
  "weights": {
    "net.0.weight": [[...], ...],
    "net.0.bias": [...],
    "net.1.weight": [...],
    "net.1.bias": [...],
    ...
  }
}
```

Weight keys follow PyTorch `nn.Sequential` naming. The `NNPolicy` runtime auto-detects Linear vs LayerNorm layers from the weight tensor dimensionality.

## Examples

### MVEE — multi-class action classification

Two decision layers (Talker and Executor) trained on LLM-labeled episodes. See `examples/mvee_example.py`.

```
python3 examples/mvee_example.py --episodes-dir training_data --output-dir weights
```

### Precursors — multi-label regression (limbic urgency weights)

Maps 140-dim chemical/species/imprint state to 13 urgency weights via MSELoss + sigmoid. See `examples/precursors_example.py`.

```
python3 examples/precursors_example.py --episodes-dir training_data --output-dir weights
```

The Precursors limbic system is the reference case study: in production, this replaced 80% of LLM calls with a 50KB neural net running at <0.5ms for 100 concurrent agents.

## License

MIT

## Credits

Extracted from the [Multiverse Games](https://github.com/Multiverse-Games) project.

### Research

- **Policy Distillation** — Rusu et al., 2015 (arXiv:1511.06295)
- **DAgger** — Ross et al., 2011 (arXiv:1011.0686)
- **LLM4Teach** — Fu et al., 2023 (arXiv:2311.13373)
- **Talker-Reasoner** dual-process architecture — arXiv:2410.08328
- **Constitutional AI** — Anthropic (RLAIF labeling pattern)

### Heritage

The biochemical urgency-weight architecture descends from the Creatures engine (Steve Grand, 1996). Chemicals map to drives; drives compete via normalised weights.

- Grand, S. (2000). *Creation: Life and How to Make It.* Harvard University Press.
- Grand, S. (1997). "Creatures: An Exercise in Creation." *IEEE Intelligent Systems.*
