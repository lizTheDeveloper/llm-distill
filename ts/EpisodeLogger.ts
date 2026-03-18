// MIT License
//
// Copyright (c) 2026 Multiverse Games
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

/**
 * EpisodeLogger — generic middleware for logging LLM decisions as training
 * episodes for the llm-distill package.
 *
 * Captures (feature_vector, action_type, confidence) tuples after each LLM
 * decision and buffers them for later distillation training. Episodes are
 * output in JSONL format compatible with the Python `EpisodeDataset` loader.
 *
 * This is the framework-agnostic version of the game-specific EpisodeCollector
 * from Precursors. It imposes no dependencies on game types, rendering engines,
 * or backend frameworks — the caller supplies feature vectors and action labels.
 *
 * Typical usage:
 *
 *   const logger = new EpisodeLogger({ flushEndpoint: '/api/training/episodes' });
 *   logger.setEnabled(true);
 *
 *   // After each LLM call:
 *   logger.log(featureVector, actionType, confidence);
 *
 *   // On page unload or session end:
 *   await logger.flush();
 *   // or: const jsonl = logger.exportJSONL();
 *
 * Ref: LLM4Teach (arXiv:2311.13373) episode collection pattern.
 *
 * @module llm-distill/EpisodeLogger
 */

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

/**
 * A single training episode record. Matches the schema expected by the Python
 * `EpisodeDataset` loader in llm-distill.
 */
export interface EpisodeRecord {
  /** Numeric feature vector encoding the decision context. */
  feature_vector: number[];
  /** Categorical label for the action chosen. */
  action_type: string;
  /** Model's self-reported confidence in [0, 1]. Optional. */
  confidence?: number;
  /** ISO 8601 timestamp of when the episode was logged. */
  timestamp: string;
  /** Identifies the session this episode belongs to. */
  session_id: string;
  /** Arbitrary caller-supplied key/value pairs for debugging or slicing. */
  metadata?: Record<string, unknown>;
}

/**
 * Configuration for {@link EpisodeLogger}.
 *
 * All fields are optional; the logger works out-of-the-box with defaults.
 */
export interface EpisodeLoggerConfig {
  /**
   * Number of buffered episodes that triggers an automatic flush.
   * Set to `Infinity` to disable auto-flush entirely.
   * @default 50
   */
  flushThreshold?: number;

  /**
   * HTTP POST endpoint to send batched episodes to.
   * When omitted, episodes accumulate in memory and must be exported manually
   * via {@link EpisodeLogger.exportJSONL} or {@link EpisodeLogger.exportJSON}.
   */
  flushEndpoint?: string;

  /**
   * Session identifier shared across all episodes logged in this instance.
   * Auto-generated from timestamp + random suffix when not provided.
   */
  sessionId?: string;
}

// ---------------------------------------------------------------------------
// EpisodeLogger
// ---------------------------------------------------------------------------

export class EpisodeLogger {
  private _enabled = false;
  private _sessionId: string;
  private _buffer: EpisodeRecord[] = [];
  private _totalFlushed = 0;
  private readonly _flushThreshold: number;
  private readonly _flushEndpoint: string | undefined;

  constructor(config: EpisodeLoggerConfig = {}) {
    this._flushThreshold = config.flushThreshold ?? 50;
    this._flushEndpoint = config.flushEndpoint;
    this._sessionId =
      config.sessionId ??
      `ses_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  }

  // -------------------------------------------------------------------------
  // Core logging
  // -------------------------------------------------------------------------

  /**
   * Log a single episode. Call this after each LLM decision.
   *
   * When the internal buffer reaches `flushThreshold` and a `flushEndpoint`
   * is configured, an automatic flush is triggered (fire-and-forget).
   *
   * @param featureVector - Numeric representation of the decision context.
   * @param actionType    - Categorical label for the action chosen.
   * @param confidence    - Model confidence in [0, 1]. Optional.
   * @param metadata      - Arbitrary extra fields for debugging or slicing.
   */
  log(
    featureVector: number[],
    actionType: string,
    confidence?: number,
    metadata?: Record<string, unknown>,
  ): void {
    if (!this._enabled) return;

    const record: EpisodeRecord = {
      feature_vector: featureVector,
      action_type: actionType,
      timestamp: new Date().toISOString(),
      session_id: this._sessionId,
    };

    if (confidence !== undefined) {
      record.confidence = confidence;
    }

    if (metadata !== undefined) {
      record.metadata = metadata;
    }

    this._buffer.push(record);

    if (
      this._flushEndpoint !== undefined &&
      this._buffer.length >= this._flushThreshold
    ) {
      // Fire-and-forget auto-flush; errors are logged to console.
      void this.flush();
    }
  }

  // -------------------------------------------------------------------------
  // Enable / disable
  // -------------------------------------------------------------------------

  /**
   * Enable or disable logging. While disabled, calls to {@link log} are
   * silently ignored. The buffer is preserved across toggles.
   */
  setEnabled(on: boolean): void {
    this._enabled = on;
    console.log(
      on
        ? `[EpisodeLogger] Enabled — session ${this._sessionId}`
        : `[EpisodeLogger] Disabled — ${this._buffer.length} episodes buffered`,
    );
  }

  /** Returns `true` when logging is active. */
  isEnabled(): boolean {
    return this._enabled;
  }

  // -------------------------------------------------------------------------
  // Counts
  // -------------------------------------------------------------------------

  /** Number of episodes currently held in the in-memory buffer. */
  get bufferedCount(): number {
    return this._buffer.length;
  }

  /**
   * Total episodes logged this session, including those already flushed to
   * the endpoint.
   */
  get totalCount(): number {
    return this._totalFlushed + this._buffer.length;
  }

  // -------------------------------------------------------------------------
  // Flush
  // -------------------------------------------------------------------------

  /**
   * POST all buffered episodes to `flushEndpoint` as a JSONL body.
   *
   * - If no endpoint is configured, this is a no-op and resolves immediately.
   * - On network failure the buffer is restored so no episodes are lost; the
   *   error is logged to console.
   * - On a non-2xx response the buffer is also restored.
   */
  async flush(): Promise<void> {
    if (this._buffer.length === 0) return;
    if (!this._flushEndpoint) return;

    const batch = [...this._buffer];
    const count = batch.length;
    // Clear before the await so concurrent log() calls don't duplicate.
    this._buffer = [];

    const body = batch.map((r) => JSON.stringify(r)).join('\n');

    try {
      const res = await fetch(this._flushEndpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/x-ndjson' },
        body,
      });

      if (res.ok) {
        this._totalFlushed += count;
        console.log(`[EpisodeLogger] Flushed ${count} episodes to ${this._flushEndpoint}`);
      } else {
        console.warn(
          `[EpisodeLogger] Flush failed (HTTP ${res.status}) — restoring ${count} episodes to buffer`,
        );
        this._buffer = [...batch, ...this._buffer];
      }
    } catch (err) {
      console.warn(
        `[EpisodeLogger] Flush error (${String(err)}) — restoring ${count} episodes to buffer`,
      );
      this._buffer = [...batch, ...this._buffer];
    }
  }

  // -------------------------------------------------------------------------
  // Export
  // -------------------------------------------------------------------------

  /**
   * Export all buffered episodes as a JSONL string (one JSON object per line).
   *
   * Compatible with the Python `EpisodeDataset` loader:
   * ```python
   * dataset = EpisodeDataset.from_jsonl("episodes.jsonl")
   * ```
   */
  exportJSONL(): string {
    return this._buffer.map((r) => JSON.stringify(r)).join('\n');
  }

  /**
   * Export all buffered episodes as a JSON array string.
   *
   * Useful for human inspection or tools that prefer a JSON array over JSONL.
   */
  exportJSON(): string {
    return JSON.stringify(this._buffer, null, 2);
  }

  // -------------------------------------------------------------------------
  // Utility
  // -------------------------------------------------------------------------

  /**
   * Discard all buffered episodes. Does not affect `totalCount` (which only
   * tracks flushed episodes) — use with care.
   */
  clear(): void {
    this._buffer = [];
  }

  /** The session ID shared by all episodes logged from this instance. */
  get sessionId(): string {
    return this._sessionId;
  }
}
