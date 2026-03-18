/**
 * NNPolicy.ts — Generic TypeScript inference runtime for the llm-distill package.
 *
 * Loads JSON weights exported by `export_weights_json()` from the llm-distill
 * Python package and runs a forward pass through a PolicyNN-compatible
 * architecture without any external dependencies.
 *
 * Architecture pattern (mirrors PolicyNN in llm_distill/model.py):
 *   For each hidden block: Linear → LayerNorm → GELU
 *   Dropout layers are skipped at inference (no weights present in export).
 *   Final layer: Linear only (raw logits).
 *
 * Theoretical grounding:
 *   Policy Distillation (Rusu et al., 2015) — arXiv:1511.06295
 *   LLM4Teach (Fu et al., 2023) — arXiv:2311.13373
 *
 * Game-specific usage: see LimbicPolicyInference.ts in Precursors for an
 * example of wrapping this class with domain-specific feature encoding.
 *
 * MIT License
 * Copyright (c) 2026 Multiverse Games
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

// ---------------------------------------------------------------------------
// JSON weight format
// ---------------------------------------------------------------------------

/**
 * Shape of the JSON file produced by `export_weights_json()` in the
 * llm-distill Python package.
 *
 * Weight key naming follows PyTorch nn.Sequential conventions:
 *   - Hidden block i: "net.{linearIdx}.weight" / "net.{linearIdx}.bias"   (Linear)
 *                     "net.{normIdx}.weight"   / "net.{normIdx}.bias"      (LayerNorm gamma/beta)
 *   - Final layer:    "net.{lastIdx}.weight"   / "net.{lastIdx}.bias"      (Linear, raw logits)
 * Dropout layers carry no parameters and are therefore absent from the export.
 */
export interface ModelWeightsJSON {
  /** Identifier string set at export time (e.g. "talker", "executor"). */
  model: string;
  /** Expected dimensionality of the input feature vector. */
  input_dim: number;
  /** Number of output classes / action logits. */
  output_dim: number;
  /** Ordered list of action names; index i corresponds to logit i. */
  actions: string[];
  /**
   * Named parameter tensors.
   * - 2-D tensors (weight matrices): number[][]
   * - 1-D tensors (bias / gamma / beta): number[]
   */
  weights: Record<string, number[] | number[][]>;
}

// ---------------------------------------------------------------------------
// Inference result
// ---------------------------------------------------------------------------

/** Result returned by `NNPolicy.predict()`. */
export interface PredictResult {
  /** Name of the selected action (from the `actions` list in the JSON). */
  action: string;
  /** Zero-based index of the selected action. */
  actionIndex: number;
  /**
   * Softmax probability of the selected action in [0, 1].
   * This is the model's confidence, not a raw logit.
   */
  confidence: number;
}

// ---------------------------------------------------------------------------
// Detected layer plan
// ---------------------------------------------------------------------------

/**
 * Describes a single layer as parsed from the weight keys.
 * Used to build the pre-allocated scratch buffer plan.
 */
interface LayerPlan {
  type: 'linear' | 'layernorm';
  /** Sequential index in the nn.Sequential (e.g. 0, 1, 3, 4, …). */
  netIdx: number;
  /** Output dimensionality of this layer. */
  outDim: number;
}

// ---------------------------------------------------------------------------
// Math primitives  (all in-place, all zero-allocation)
// ---------------------------------------------------------------------------

/**
 * In-place linear layer: out[i] = bias[i] + dot(weight[i], x)
 *
 * @param x      Input buffer (length = in_dim).
 * @param weight Row-major weight matrix (out_dim × in_dim).
 * @param bias   Bias vector (length = out_dim).
 * @param out    Output buffer (length = out_dim). May alias x only if safe.
 */
function linearLayerInPlace(
  x: Float32Array,
  weight: number[][],
  bias: number[],
  out: Float32Array,
): void {
  const outDim = weight.length;
  const inDim = x.length;
  for (let i = 0; i < outDim; i++) {
    let sum = bias[i];
    const row = weight[i];
    for (let j = 0; j < inDim; j++) {
      sum += row[j] * x[j];
    }
    out[i] = sum;
  }
}

/**
 * In-place layer normalization.
 *
 * Normalizes `x` to zero mean / unit variance, then applies the learned
 * affine transform: out[i] = gamma[i] * ((x[i] - mean) / sqrt(var + eps)) + beta[i]
 *
 * eps = 1e-5 (matches PyTorch default).
 *
 * @param x     Input buffer (length = dim).
 * @param gamma LayerNorm weight (gamma); exported as "net.{idx}.weight".
 * @param beta  LayerNorm bias (beta);  exported as "net.{idx}.bias".
 * @param out   Output buffer (length = dim). May safely alias x.
 */
function layerNormInPlace(
  x: Float32Array,
  gamma: number[],
  beta: number[],
  out: Float32Array,
): void {
  const len = x.length;
  const eps = 1e-5;

  let mean = 0;
  for (let i = 0; i < len; i++) mean += x[i];
  mean /= len;

  let variance = 0;
  for (let i = 0; i < len; i++) {
    const d = x[i] - mean;
    variance += d * d;
  }
  variance /= len;

  const std = Math.sqrt(variance + eps);
  for (let i = 0; i < len; i++) {
    out[i] = gamma[i] * ((x[i] - mean) / std) + beta[i];
  }
}

/**
 * In-place approximate GELU activation.
 *
 * Uses the tanh approximation from the original GELU paper (Hendrycks & Gimpel, 2016):
 *   GELU(v) ≈ v · 0.5 · (1 + tanh(√(2/π) · (v + 0.044715·v³)))
 * where √(2/π) ≈ 0.7978845608.
 *
 * This matches PyTorch's `nn.GELU(approximate='tanh')` and is the default in
 * the llm-distill PolicyNN.
 *
 * @param buf Buffer to activate in place.
 * @param len Number of elements to process (may be less than buf.length).
 */
function geluInPlace(buf: Float32Array, len: number): void {
  for (let i = 0; i < len; i++) {
    const v = buf[i];
    const cdf = 0.5 * (1 + Math.tanh(0.7978845608 * (v + 0.044715 * v * v * v)));
    buf[i] = v * cdf;
  }
}

/**
 * In-place numerically-stable softmax.
 *
 * Subtracts max before exponentiation to prevent overflow.
 *
 * @param buf Buffer to transform in place.
 * @param len Number of elements to process.
 */
function softmaxInPlace(buf: Float32Array, len: number): void {
  let max = buf[0];
  for (let i = 1; i < len; i++) {
    if (buf[i] > max) max = buf[i];
  }

  let sumExp = 0;
  for (let i = 0; i < len; i++) {
    buf[i] = Math.exp(buf[i] - max);
    sumExp += buf[i];
  }

  for (let i = 0; i < len; i++) {
    buf[i] /= sumExp;
  }
}

// ---------------------------------------------------------------------------
// Architecture detection
// ---------------------------------------------------------------------------

/**
 * Parse the weight key set to reconstruct the layer sequence.
 *
 * The export uses nn.Sequential indexing. We walk the indices in order,
 * identifying Linear layers (have a 2-D weight) and LayerNorm layers (have a
 * 1-D weight). GELU and Dropout carry no parameters, so they don't appear.
 *
 * Returns an ordered list of LayerPlan entries suitable for building scratch
 * buffers and driving the forward pass loop.
 */
function detectLayerPlan(weights: Record<string, number[] | number[][]>): LayerPlan[] {
  // Collect all net.{idx}.weight keys and sort by idx.
  const idxSet = new Set<number>();
  for (const key of Object.keys(weights)) {
    const m = key.match(/^net\.(\d+)\.weight$/);
    if (m) idxSet.add(parseInt(m[1], 10));
  }

  const indices = Array.from(idxSet).sort((a, b) => a - b);
  const plan: LayerPlan[] = [];

  for (const idx of indices) {
    const w = weights[`net.${idx}.weight`];
    // 2-D → Linear; 1-D → LayerNorm (gamma).
    if (Array.isArray(w) && Array.isArray(w[0])) {
      // Linear: outDim = number of rows.
      const outDim = (w as number[][]).length;
      plan.push({ type: 'linear', netIdx: idx, outDim });
    } else {
      // LayerNorm: outDim = length of gamma vector.
      const outDim = (w as number[]).length;
      plan.push({ type: 'layernorm', netIdx: idx, outDim });
    }
  }

  return plan;
}

// ---------------------------------------------------------------------------
// Scratch buffer allocation
// ---------------------------------------------------------------------------

/**
 * Pre-allocate one Float32Array per unique activation size seen in the layer
 * plan.  Hidden layers share a buffer when they have the same width (since
 * activations are consumed before the next use of that size).  We keep two
 * separate "ping/pong" slots of each size to allow safe in-place reads from
 * one and writes to the other, even when adjacent layers share a width.
 *
 * Returns a map from outDim → [ping, pong] pair.
 */
function allocateScratch(
  inputDim: number,
  plan: LayerPlan[],
): {
  input: Float32Array;
  buffers: Map<number, [Float32Array, Float32Array]>;
} {
  const sizes = new Set<number>();
  sizes.add(inputDim);
  for (const layer of plan) sizes.add(layer.outDim);

  const buffers = new Map<number, [Float32Array, Float32Array]>();
  for (const size of sizes) {
    buffers.set(size, [new Float32Array(size), new Float32Array(size)]);
  }

  return {
    input: buffers.get(inputDim)![0],
    buffers,
  };
}

// ---------------------------------------------------------------------------
// NNPolicy
// ---------------------------------------------------------------------------

/**
 * Generic inference runtime for any model trained and exported by llm-distill.
 *
 * Construction parses the JSON weight format, detects the layer architecture
 * automatically, and pre-allocates all Float32Array scratch buffers.
 * Every subsequent `forward()` / `predict()` call is allocation-free.
 *
 * @example
 * ```ts
 * const policy = await NNPolicy.fromURL('/models/talker.json');
 * const result = policy.predict(featureVector);
 * console.log(result.action, result.confidence);
 * ```
 */
export class NNPolicy {
  // -------------------------------------------------------------------------
  // Private fields
  // -------------------------------------------------------------------------

  private readonly _modelName: string;
  private readonly _inputDim: number;
  private readonly _outputDim: number;
  private readonly _actions: string[];
  private readonly _weights: Record<string, number[] | number[][]>;

  /** Ordered layer plan derived from weight keys at construction time. */
  private readonly _plan: LayerPlan[];

  /** Pre-allocated scratch buffers. */
  private readonly _scratch: {
    input: Float32Array;
    buffers: Map<number, [Float32Array, Float32Array]>;
  };

  // -------------------------------------------------------------------------
  // Constructor
  // -------------------------------------------------------------------------

  /**
   * Create an NNPolicy from a parsed JSON weight object.
   *
   * @param weightsJson Parsed JSON from a file produced by `export_weights_json()`.
   * @throws Error if `weightsJson` is missing required fields or has no weight keys.
   */
  constructor(weightsJson: ModelWeightsJSON) {
    if (typeof weightsJson.model !== 'string') {
      throw new Error('NNPolicy: weightsJson.model must be a string.');
    }
    if (typeof weightsJson.input_dim !== 'number' || weightsJson.input_dim <= 0) {
      throw new Error('NNPolicy: weightsJson.input_dim must be a positive integer.');
    }
    if (typeof weightsJson.output_dim !== 'number' || weightsJson.output_dim <= 0) {
      throw new Error('NNPolicy: weightsJson.output_dim must be a positive integer.');
    }
    if (!Array.isArray(weightsJson.actions) || weightsJson.actions.length !== weightsJson.output_dim) {
      throw new Error(
        `NNPolicy: weightsJson.actions length (${weightsJson.actions?.length}) ` +
        `must equal output_dim (${weightsJson.output_dim}).`,
      );
    }

    this._modelName = weightsJson.model;
    this._inputDim = weightsJson.input_dim;
    this._outputDim = weightsJson.output_dim;
    this._actions = weightsJson.actions.slice();
    this._weights = weightsJson.weights;

    this._plan = detectLayerPlan(weightsJson.weights);

    if (this._plan.length === 0) {
      throw new Error('NNPolicy: no weight keys found matching "net.{n}.weight" pattern.');
    }

    // Validate that the final linear layer's output dimension matches output_dim.
    const lastLinear = [...this._plan].reverse().find(l => l.type === 'linear');
    if (!lastLinear || lastLinear.outDim !== this._outputDim) {
      throw new Error(
        `NNPolicy: final linear layer output dim (${lastLinear?.outDim}) ` +
        `does not match output_dim (${this._outputDim}).`,
      );
    }

    this._scratch = allocateScratch(this._inputDim, this._plan);
  }

  // -------------------------------------------------------------------------
  // Static factory
  // -------------------------------------------------------------------------

  /**
   * Fetch a JSON weights file from `url` and construct an NNPolicy.
   *
   * Works in any environment that exposes the global `fetch` API
   * (modern browsers, Node 18+, Deno, Bun).
   *
   * @param url Absolute or relative URL to a `.json` weights file.
   * @throws Error if the fetch fails or the response is not valid JSON.
   */
  static async fromURL(url: string): Promise<NNPolicy> {
    const res = await fetch(url);
    if (!res.ok) {
      throw new Error(`NNPolicy.fromURL: HTTP ${res.status} fetching "${url}".`);
    }
    const json = (await res.json()) as ModelWeightsJSON;
    return new NNPolicy(json);
  }

  // -------------------------------------------------------------------------
  // Public accessors
  // -------------------------------------------------------------------------

  /** The model identifier string from the JSON (e.g. "talker"). */
  get modelName(): string { return this._modelName; }

  /** Expected input feature dimensionality. */
  get inputDim(): number { return this._inputDim; }

  /** Number of output action classes. */
  get outputDim(): number { return this._outputDim; }

  /** Ordered list of action names; index i corresponds to logit i. */
  get actions(): string[] { return this._actions; }

  // -------------------------------------------------------------------------
  // Inference
  // -------------------------------------------------------------------------

  /**
   * Run a full forward pass and return raw logits.
   *
   * The returned Float32Array is a view into a pre-allocated scratch buffer.
   * It remains valid until the next call to `forward()` or `predict()`.
   * Copy it with `.slice()` if you need to retain the values.
   *
   * @param features Input feature vector. Must have exactly `inputDim` elements.
   * @returns Float32Array of length `outputDim` containing raw (un-normalised) logits.
   */
  forward(features: number[] | Float32Array): Float32Array {
    // Copy features into the input scratch buffer.
    const input = this._scratch.input;
    if (features.length !== this._inputDim) {
      throw new Error(
        `NNPolicy.forward: features length ${features.length} does not match ` +
        `input_dim ${this._inputDim}.`,
      );
    }
    for (let i = 0; i < this._inputDim; i++) {
      input[i] = features[i];
    }

    // Execute the layer plan.
    // We maintain a pointer to the "current activation" buffer (src) and a
    // "destination" buffer (dst).  After each layer the roles may swap.
    let src: Float32Array = input;
    let srcDim = this._inputDim;

    // Ping/pong slot selector per unique size (0 = ping, 1 = pong).
    const slotOf = new Map<number, 0 | 1>();

    // Helper: get the next write slot for a given output dimension.
    const getDst = (outDim: number): Float32Array => {
      const pair = this._scratch.buffers.get(outDim)!;
      const slot = slotOf.get(outDim) ?? 0;
      slotOf.set(outDim, slot === 0 ? 1 : 0);
      return pair[slot];
    };

    // Iterate layer plan entries.
    // LayerNorm entries always immediately follow the Linear they normalize,
    // so `src` going into a LayerNorm has already been written by the Linear.
    let i = 0;
    while (i < this._plan.length) {
      const layer = this._plan[i];

      if (layer.type === 'linear') {
        const weight = this._weights[`net.${layer.netIdx}.weight`] as number[][];
        const bias   = this._weights[`net.${layer.netIdx}.bias`]   as number[];
        const dst    = getDst(layer.outDim);

        linearLayerInPlace(src, weight, bias, dst);

        src    = dst;
        srcDim = layer.outDim;
        i++;

        // Check whether the next entry is a LayerNorm for this same activation.
        if (i < this._plan.length && this._plan[i].type === 'layernorm') {
          const norm    = this._plan[i];
          const gamma   = this._weights[`net.${norm.netIdx}.weight`] as number[];
          const beta    = this._weights[`net.${norm.netIdx}.bias`]   as number[];
          // LayerNorm output dim must equal input dim.
          const normDst = getDst(norm.outDim);

          layerNormInPlace(src, gamma, beta, normDst);

          src    = normDst;
          srcDim = norm.outDim;
          i++;

          // GELU always follows a hidden LayerNorm (not the final Linear).
          // We detect this heuristically: if the next layer is another Linear
          // (or we're at the end but this is not the last linear), apply GELU.
          // Simpler rule: apply GELU after every LayerNorm (the final layer
          // has no LayerNorm, so this branch is only reached for hidden blocks).
          geluInPlace(src, srcDim);
        }
        // If the next entry is NOT a LayerNorm, we just finished the final
        // Linear and output raw logits — no activation applied.

      } else {
        // Standalone LayerNorm (unusual but handled for robustness).
        const gamma   = this._weights[`net.${layer.netIdx}.weight`] as number[];
        const beta    = this._weights[`net.${layer.netIdx}.bias`]   as number[];
        const normDst = getDst(layer.outDim);

        layerNormInPlace(src, gamma, beta, normDst);

        src    = normDst;
        srcDim = layer.outDim;
        i++;
      }
    }

    return src;
  }

  /**
   * Run inference and return the greedy-argmax action with its softmax confidence.
   *
   * Internally calls `forward()`, applies softmax, then picks the argmax.
   * This does NOT mutate the caller's `features` array.
   *
   * @param features Input feature vector. Must have exactly `inputDim` elements.
   * @returns `{ action, actionIndex, confidence }`.
   */
  predict(features: number[] | Float32Array): PredictResult {
    // forward() returns a view into scratch; copy before softmax mutates it.
    const logits = this.forward(features);

    // Use a dedicated softmax buffer (re-use the output pair's unused slot).
    const pair = this._scratch.buffers.get(this._outputDim)!;
    // The last forward() wrote to one slot; read from the other for softmax.
    // Actually: `logits` already IS one of the pair slots. Pick the other.
    const probsBuf = logits === pair[0] ? pair[1] : pair[0];
    probsBuf.set(logits.subarray(0, this._outputDim));
    softmaxInPlace(probsBuf, this._outputDim);

    let bestIdx = 0;
    let bestProb = probsBuf[0];
    for (let i = 1; i < this._outputDim; i++) {
      if (probsBuf[i] > bestProb) {
        bestProb = probsBuf[i];
        bestIdx  = i;
      }
    }

    return {
      action:      this._actions[bestIdx],
      actionIndex: bestIdx,
      confidence:  bestProb,
    };
  }
}
