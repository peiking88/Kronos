# External Integrations

## Core Sections (Required)

### 1) Integration Inventory

| System               | Type (API/DB/Queue/etc)     | Purpose                                                                        | Auth model                               | Criticality | Evidence                                         |
| -------------------- | --------------------------- | ------------------------------------------------------------------------------ | ---------------------------------------- | ----------- | ------------------------------------------------ |
| HuggingFace Hub      | API (model registry)        | Download/upload pre-trained KronosTokenizer and Kronos models                  | Unauthenticated (read); token for upload | high        | `model/kronos.py:4` (PyTorchModelHubMixin)       |
| akshare              | Python library              | Fetch A-share daily OHLCV data (stock_zh_a_hist)                               | None (free)                              | medium      | `examples/prediction_cn_markets_day.py:28,57`    |
| 东方财富 (EastMoney) | HTTP API                    | Fetch A-share K-line data with 前复权 adjustment                               | None (push2his.eastmoney.com endpoint)   | medium      | `examples/get_akshare_date_2024-2025_x.py:46-58` |
| Qlib (Microsoft)     | Python library + local data | Financial data management, preparation, and backtesting for A-share finetuning | None (local data)                        | medium      | `finetune/config.py:13`, `finetune/dataset.py`   |
| Comet ML             | API (experiment tracking)   | Optional training metric logging                                               | API key in config                        | low         | `finetune/config.py:75-82`                       |

### 2) Data Stores

| Store                                         | Role                                                         | Access layer                                      | Key risk                                              | Evidence                                                                |
| --------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------- | ----------------------------------------------------- | ----------------------------------------------------------------------- |
| Qlib local data (`~/.qlib/qlib_data/cn_data`) | A-share daily OHLCV features for finetuning                  | `finetune/qlib_data_preprocess.py` → pickle files | Path must be configured manually; stale data possible | `finetune/config.py:13`                                                 |
| CSV files (local)                             | Custom market data for finetuning (e.g., HK stocks 5min)     | `finetune_csv/data/*.csv` → pandas                | Data format must match expected OHLCV columns         | `finetune_csv/data/HK_ali_09988_kline_5min_all.csv`                     |
| Test regression data (`tests/data/`)          | Pinned CSV input/output for deterministic regression testing | `tests/test_kronos_regression.py`                 | Tests break if model revisions change                 | `tests/data/regression_input.csv`, `tests/data/regression_output_*.csv` |
| Pickle datasets                               | Serialized preprocessed features (train/val/test splits)     | `finetune/dataset.py:42`                          | Large files not committed to git                      | `finetune/config.py:41`                                                 |

### 3) Secrets and Credentials Handling

- Credential sources: Environment variables for git (`GIT_USERNAME`, `GIT_TOKEN`). Comet ML API key hardcoded as placeholder `"YOUR_COMET_API_KEY"` in `finetune/config.py:79` — [ASK USER] whether to move to env var.
- Hardcoding checks: Comet ML API key is a placeholder, not a real secret. No real credentials found in committed code.
- Rotation or lifecycle notes: HuggingFace Hub token (if push needed) via `huggingface-cli login`. [TODO] for other services.

### 4) Reliability and Failure Behavior

- Retry/backoff behavior: `examples/prediction_cn_markets_day.py:55-62` — 3 retry attempts with 1.5s sleep for akshare fetch. `examples/get_akshare_date_2024-2025_x.py:67` — `time.sleep(random.uniform(1,2))` for rate limiting.
- Timeout policy: EastMoney HTTP requests use `timeout=10` (`examples/get_akshare_date_2024-2025_x.py:69`).
- Circuit-breaker or fallback behavior: WebUI falls back to simulated data when `MODEL_AVAILABLE = False` (`webui/app.py:21-22`). [TODO] for finetuning pipeline.

### 5) Observability for Integrations

- Logging around external calls: `print()` statements with emoji markers (✅/❌/📥/⚠️). Training uses `tqdm` progress bars.
- Metrics/tracing coverage: Comet ML (optional) for training metrics. None for inference.
- Missing visibility gaps: No structured logging; no error aggregation; no latency tracking for model inference; no alerting on data fetch failures.

### 6) Evidence

- `model/kronos.py:4` — HuggingFace Hub integration via PyTorchModelHubMixin
- `examples/prediction_cn_markets_day.py:28,57-61` — akshare data fetching with retry
- `examples/get_akshare_date_2024-2025_x.py:46-69` — EastMoney HTTP API
- `finetune/config.py:13,75-82` — Qlib data path, Comet ML config
- `webui/app.py:20-22` — WebUI fallback when model unavailable
