# Architecture

## Core Sections (Required)

### 1) Architectural Style

- Primary style: Two-stage encoder-decoder (Tokenizer → Predictor) with autoregressive sampling
- Why this classification: The system uses a hierarchical discrete tokenizer (BSQuantizer + Transformer encoder/decoder) to quantize continuous OHLCV data into discrete tokens, then a decoder-only Transformer to autoregressively predict future tokens, which are decoded back to price space.
- Primary constraints:
  1. `max_context` limits input sequence length (512 for Kronos-small/base, 2048 for Kronos-mini)
  2. All series in `predict_batch` must have identical `lookback` and `pred_len`
  3. GPU VRAM limits batch size and model variant selection

### 2) System Flow

```text
[OHLCV DataFrame + Timestamps] 
  → calc_time_stamps() extracts [minute, hour, weekday, day, month] 
  → Z-score normalization (per-window mean/std, clip=5) 
  → KronosTokenizer.encode() quantizes to discrete (s1, s2) tokens 
  → Kronos.decode_s1() + decode_s2() autoregressively sample next tokens 
  → KronosTokenizer.decode() reconstructs to price space 
  → Inverse normalization (x * std + mean) 
  → [Prediction DataFrame]
```

Detailed trace for `KronosPredictor.predict()` (`model/kronos.py:519`):
1. Validate required columns `['open', 'high', 'low', 'close']` exist, fill missing volume/amount
2. `calc_time_stamps()` derives 5 temporal features from timestamps
3. Z-score normalize: `x = (x - x_mean) / (x_std + 1e-5)`, clip to `[-5, 5]`
4. `auto_regressive_inference()` (`model/kronos.py:389`):
   - `tokenizer.encode(x, half=True)` → bipolar-quantized s1/s2 token pairs
   - Loop over `pred_len` steps: model decodes s1 logits → sample → conditionally decode s2 logits → sample
   - Ring buffer (size `max_context`) manages sliding context window
5. `tokenizer.decode(input_tokens, half=True)` reconstructs price features
6. Average over `sample_count` paths, inverse-normalize, return DataFrame

### 3) Layer/Module Responsibilities

| Layer or module | Owns | Must not own | Evidence |
|-----------------|------|--------------|----------|
| `KronosTokenizer` | Data compression: continuous OHLCV → discrete bipolar tokens via BSQuantizer | Token prediction / autoregressive logic | `model/kronos.py:13-177` |
| `Kronos` (predictor) | Hierarchical autoregressive token prediction (s1 first, then s2 conditioned on s1) | Tokenization, price normalization | `model/kronos.py:180-328` |
| `KronosPredictor` | Full inference pipeline: normalization → tokenization → prediction → denormalization | Model training, data fetching | `model/kronos.py:482-661` |
| `module.py` | Neural building blocks: BSQuantizer, TransformerBlock, attention, embeddings, DualHead | High-level prediction workflow | `model/module.py` |
| `finetune/config.py` | Qlib pipeline configuration (paths, hyperparameters, time ranges) | Inference logic | `finetune/config.py` |
| `finetune_csv/configs/*.yaml` | CSV pipeline YAML configuration | Training loop logic | `finetune_csv/configs/` |
| `webui/app.py` | Flask web server, Plotly chart rendering, model selection UI | Model logic (delegates to `model/`) | `webui/app.py` |

### 4) Reused Patterns

| Pattern | Where found | Why it exists |
|---------|-------------|---------------|
| **Two-stage hierarchical prediction** | `Kronos.forward()`, `auto_regressive_inference()` | Predict coarse s1 token first, then fine s2 conditioned on s1 — captures dependency structure |
| **Binary Spherical Quantization** | `BSQuantizer` (in `model/module.py:225-254`), `BinarySphericalQuantizer` (`module.py:39-222`) | Compresses continuous financial data to discrete codebook using bipolar {-1,+1} quantization |
| **Sliding window normalization** | `KronosPredictor.predict()`, `QlibDataset.__getitem__()` | Mean/std computed only on lookback window to prevent future data leakage |
| **Teacher forcing + autoregressive sampling** | `Kronos.forward(use_teacher_forcing=True/False)` | Training uses teacher forcing; inference uses sampling from s1 logits |
| **Ring buffer context management** | `auto_regressive_inference()` lines 408-454 | Maintains max_context-length sliding window during autoregressive generation |
| **Rotary Position Embedding (RoPE)** | `MultiHeadAttentionWithRoPE`, `RotaryPositionalEmbedding` (module.py:284-312) | Encodes positional information in attention without learned embeddings |
| **Dependency-Aware Layer** | `DependencyAwareLayer` (module.py:446-462) | Cross-attention where s1 embedding conditions s2 prediction |
| **ConfigLoader pattern** | CSV finetuning | YAML config → Python object with path resolution and template substitution |

### 5) Known Architectural Risks

- **Single time-step autoregressive sampling**: Each step depends on the previous sampled token; errors compound over long prediction horizons. This is inherent to autoregressive models, not a bug.
- **Naive mean/std normalization**: Per-window Z-score assumes stationarity; extreme market regime changes (crashes, gaps) may cause out-of-distribution inputs.
- **No explicit calendar/ holiday modeling**: `calc_time_stamps()` uses simple datetime fields (minute/hour/weekday/day/month). Holidays and non-trading days are not explicitly modeled — the model learns non-trading patterns from data gaps.
- **tokenizer/model revision pinning**: Tests pin exact git revisions (`MODEL_REVISION`, `TOKENIZER_REVISION` in `tests/test_kronos_regression.py:30-31`). Model updates require test expectation updates.

### 6) Evidence

- `model/kronos.py:13-177` — KronosTokenizer (encoder/decoder, encode/decode, indices_to_bits)
- `model/kronos.py:180-328` — Kronos model (forward, decode_s1, decode_s2, autoregressive inference)
- `model/kronos.py:389-469` — auto_regressive_inference (core generation loop)
- `model/kronos.py:519-559` — KronosPredictor.predict (entry point for single prediction)
- `model/kronos.py:562-661` — KronosPredictor.predict_batch (parallel batch prediction)
- `model/module.py:225-254` — BSQuantizer (composite s1+s2 quantization)
- `model/module.py:39-222` — BinarySphericalQuantizer
- `model/module.py:486-508` — DualHead with compute_loss
