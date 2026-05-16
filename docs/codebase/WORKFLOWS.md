# Workflow Blueprints

Document representative end-to-end workflows that serve as implementation templates for similar features.

## Core Sections (Required)

### 1) Workflow Catalog

| # | Workflow name | Trigger | Entry point | Layers traversed | Business purpose |
|---|---------------|---------|-------------|------------------|------------------|
| 1 | Single Stock Prediction | User script execution | `KronosPredictor.predict()` | Data validation → Normalization → Tokenization → Autoregressive inference → Denormalization → Response | Forecast OHLCV for one stock |
| 2 | Batch Stock Prediction | User script execution | `KronosPredictor.predict_batch()` | Same as #1 but stacked across multiple series in one GPU batch | Forecast OHLCV for multiple series in parallel |
| 3 | Qlib Finetuning Pipeline | CLI command | `finetune/qlib_data_preprocess.py` → `train_tokenizer.py` → `train_predictor.py` → `qlib_test.py` | Qlib data load → Pickle → Sliding window sampling → Tokenizer training → Predictor training → Backtest | Adapt Kronos to specific A-share domain |
| 4 | CSV Custom Finetuning | CLI command | `finetune_csv/train_sequential.py` | YAML config load → CSV data → Train split → Sequential tokenizer+predictor training | Finetune Kronos on custom CSV data (e.g., HK stocks) |
| 5 | WebUI Interactive Prediction | HTTP POST request | `webui/app.py` Flask route | File upload/selection → Model load → Predictor.predict() → Plotly chart → JSON response | Browser-based interactive forecasting |

### 2) Workflow Traces

#### Workflow 1: Single Stock Prediction

**Entry point:**
- File + method signature: `model/kronos.py:519` — `KronosPredictor.predict(df, x_timestamp, y_timestamp, pred_len, T, top_k, top_p, sample_count, verbose)`
- Trigger: Direct Python call from example scripts or WebUI
- Request shape: `pd.DataFrame` with columns `[open, high, low, close, volume?, amount?]` + `pd.Series` of timestamps + `pred_len: int`

**Processing flow (trace in order):**
```
[KronosPredictor.predict():519] 
  → validate columns/NaN/DataFrame type (:522-535) 
  → fill missing volume/amount (:527-532) 
  → calc_time_stamps(x_timestamp) extracts [minute, hour, weekday, day, month] (:537)
  → calc_time_stamps(y_timestamp) (:538) 
  → Z-score normalize: x = (x - mean) / std, clip to [-5,5] (:540-547) 
  → add batch dim (:549-551) 
  → self.generate() (:508-517) 
    → np→torch conversion (:510-512) 
    → auto_regressive_inference() (:389-469)
      → clip input (:391) 
      → expand for sample_count parallel paths (:394-396) 
      → tokenizer.encode(x, half=True) → (s1_ids, s2_ids) (:398)
      → init ring buffers (:408-413)
      → for step in range(pred_len): (:420-469)
        → model.decode_s1() → s1_logits (:436-438)
        → sample_from_logits(s1_logits, T, top_k, top_p) (:438)
        → model.decode_s2(context, sample_pre) → s2_logits (:440-442)
        → sample_from_logits(s2_logits) (:442)
        → ring buffer update (:447-454)
      → tokenizer.decode(input_tokens, half=True) → price tensor (:464)
      → average over sample_count paths (:466-468)
    → slice to pred_len (:516) 
  → inverse normalize: preds * std + mean (:555-556) 
  → pd.DataFrame with columns + index (:558)
```

**Response shape:**
- Success response: `pd.DataFrame` with columns `[open, high, low, close, volume, amount]`, indexed by `y_timestamp`
- Error response: `ValueError` for invalid input types, missing columns, NaN values

**Key decisions:**
- What is validated at each layer? Column presence + DataFrame type at entry; NaN check before normalization; sequence length consistency check in batch mode
- What transforms happen to the data? OHLCV continuous → Z-score normalization → bipolar quantization (-1,+1) → discrete token IDs → autoregressive token prediction → bipolar bits → continuous price → inverse normalization
- Where are side effects triggered? None — pure compute, no I/O beyond model weights (loaded once)

#### Workflow 2: Batch Stock Prediction

**Entry point:**
- File + method signature: `model/kronos.py:562` — `KronosPredictor.predict_batch(df_list, x_timestamp_list, y_timestamp_list, pred_len, ...)`
- Trigger: `examples/prediction_batch_example.py`

**Differences from Workflow 1:**
- Validates list types + consistent lengths across all series (:582-585)
- Independently normalizes each series (:629-631)
- Requires identical seq_len + pred_len across all series (:643-646) — raises `ValueError` if inconsistent
- Stacks into (B, seq_len, feat) tensors (:648-650) for single GPU forward pass
- Independently denormalizes each output (:656-658)

#### Workflow 3: Qlib Finetuning Pipeline

**Entry point:**
- File: `finetune/qlib_data_preprocess.py` → `finetune/train_tokenizer.py` → `finetune/train_predictor.py` → `finetune/qlib_test.py`
- Trigger: Sequential CLI commands per README

**Processing flow:**
```
[qlib_data_preprocess.py]
  → load Qlib data (csi300 instruments, 2011-2025 daily)
  → slide window (90d lookback, 10d predict)
  → save train/val/test .pkl files

[train_tokenizer.py] (torchrun for multi-GPU)
  → load pretrained KronosTokenizer from HuggingFace
  → QlibDataset (random sample per epoch) 
  → train tokenizer (BSQuantizer codebook adaptation)
  → save best checkpoint

[train_predictor.py] (torchrun for multi-GPU)
  → load finetuned tokenizer + pretrained Kronos predictor
  → train predictor (cross-entropy on s1+s2 logits)
  → save best checkpoint

[qlib_test.py]
  → load finetuned models
  → inference on test set → price change signals
  → top-K strategy backtest (hold 50, drop 5)
  → cumulative return plot vs benchmark (SH000300/SH000852/SH000906)
```

#### Workflow 4: CSV Custom Finetuning

**Entry point:**
- File: `finetune_csv/train_sequential.py`
- Config: `finetune_csv/configs/config_ali09988_candle-5min.yaml`

**Processing flow:**
```
[ConfigLoader loads YAML]
  → data_path: CSV with OHLCV columns + timestamps
  → lookback_window: 512, predict_window: 48
  → train_ratio: 0.9, val_ratio: 0.1

[train_sequential.py]
  → CustomFinetuneConfig from YAML
  → load CSV → pandas DataFrame → sliding window sampling
  → (optional) finetune tokenizer (if train_tokenizer: true)
  → (optional) finetune basemodel (if train_basemodel: true)
  → save to base_save_path/{exp_name}/tokenizer|basemodel/best_model
```

#### Workflow 5: WebUI Interactive Prediction

**Entry point:**
- File: `webui/app.py` — Flask routes
- Trigger: HTTP request from browser

**Processing flow:**
```
[Browser uploads CSV or selects server file]
  → Flask route receives file + parameters (model, lookback, pred_len, T, top_p)
  → load_data_file() — validate OHLCV columns, parse timestamps
  → KronosPredictor.predict()
  → Convert to Plotly OHLC candlestick chart + volume bar chart
  → Return JSON with chart data + prediction DataFrame
```

### 3) Layer Implementation Patterns

| Layer | Pattern | Example file | Reuse approach |
|-------|---------|-------------|----------------|
| Data loading | Standalone functions in example scripts, each with own data source logic | `examples/prediction_cn_markets_day.py:48-109` | Copy-paste pattern (no shared module) |
| Data validation | Inline checks in `predict()` / `predict_batch()` | `model/kronos.py:522-535, 582-587` | Modify `predict()` for new validation rules |
| Normalization | Z-score per call: `(x - mean) / std`, clip | `model/kronos.py:544-547` | Built into predict(); override per use case |
| Tokenization | `tokenizer.encode(x, half=True)` → (s1, s2) IDs | `model/kronos.py:142-159` | Always use half=True for predictor |
| Inference | `auto_regressive_inference()` — sampling loop | `model/kronos.py:389-469` | Wrapped by generate(); use T/top_p/top_k to control |
| Denormalization | `preds * std + mean` | `model/kronos.py:555-556` | Built into predict() |
| Training (finetune) | `torchrun` + epoch loop with `QlibDataset` / CSV dataset | `finetune/train_tokenizer.py` | Config-driven; add new dataset class |

### 4) Error Handling Patterns

| Pattern | Where used | Evidence |
|---------|------------|----------|
| `ValueError` for input validation | `KronosPredictor.predict()`, `predict_batch()` | `model/kronos.py:522-535, 582-587, 600-601, 643-646` |
| `sys.exit(1)` for unrecoverable data fetch | `examples/prediction_cn_markets_day.py:66-67` | `sys.exit(1)` on akshare failure |
| try/except at main() level | `examples/prediction_akshare_2024-2025.py:504-507` | `try/except Exception` with traceback |
| Fallback to simulated data | `webui/app.py:20-22` | `MODEL_AVAILABLE` flag |
| DualHead.compute_loss with padding mask | `model/module.py:494-507` | Masked cross-entropy for variable-length sequences |
| `torch.no_grad()` for inference | All predict() paths | `model/kronos.py:390` |

### 5) Implementation Templates

**Adding a new data source for prediction:**
```
1. Create data fetching function (e.g., fetch_yahoo_finance()) following pattern in examples/
2. Map columns to Kronos input: ['open','high','low','close','volume','amount']
3. Convert timestamps with pd.to_datetime()
4. Call predictor.predict(df, x_timestamp, y_timestamp, pred_len)
5. Handle rate limiting + retry following examples/prediction_cn_markets_day.py pattern
```

**Adding a new finetuning pipeline:**
```
1. Create config (Python class or YAML) defining data_path, lookback_window, predict_window
2. Create Dataset subclass (following finetune/dataset.py QlibDataset pattern)
3. Create training script (following finetune/train_tokenizer.py structure)
4. Register in finetune_csv/configs/ if using CSV pipeline
```

**Adding a new model variant:**
```
1. Add model entry to webui/app.py AVAILABLE_MODELS dict
2. Add HuggingFace repo with config matching Kronos.__init__ signature
3. Pin model revision in tests if needed
```

### 6) Common Pitfalls

| Pitfall | Why it happens | Where observed | How to avoid |
|---------|---------------|----------------|-------------|
| Future data leakage in normalization | Normalizing entire sequence (past+future) instead of lookback only | `QlibDataset.__getitem__():110-117` correctly uses past only | Always compute mean/std on historical window only |
| Forgetting volume/amount columns | Model expects 6 features but code paths handle optional fill | `predict():527-532` fills with 0.0 if missing | Always validate feature count = 6 before tokenizer.encode() |
| GPU OOM with large batch_size | All series in batch must fit in GPU memory simultaneously | `predict_batch():648-652` stacks all series | Check `torch.cuda.max_memory_allocated()` or reduce batch_size |
| Inconsistent timestamp handling | Different scripts use 'timestamps', 'timestamp', or 'date' column names | `webui/app.py:94-100` handles all three | Standardize to 'timestamps' in data pipeline |

### 7) Evidence

- `model/kronos.py:519-559` — predict() full trace
- `model/kronos.py:562-661` — predict_batch() full trace
- `model/kronos.py:389-469` — auto_regressive_inference() generation loop
- `model/kronos.py:472-479` — calc_time_stamps() temporal feature extraction
- `finetune/dataset.py` — QlibDataset sliding window data loader
- `finetune/config.py` — Qlib pipeline configuration
- `finetune_csv/configs/config_ali09988_candle-5min.yaml` — CSV pipeline config
- `finetune_csv/train_sequential.py` — CSV finetuning entry point
- `webui/app.py` — Flask WebUI with 3 model variants
- `tests/test_kronos_regression.py` — regression + MSE tests
