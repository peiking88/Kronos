# Codebase Structure

## Core Sections (Required)

### 1) Top-Level Map

| Path | Purpose | Evidence |
|------|---------|----------|
| `model/` | Core model: KronosTokenizer, Kronos (predictor), KronosPredictor + neural network modules | `model/__init__.py`, `model/kronos.py`, `model/module.py` |
| `examples/` | Usage examples: single/batch prediction, A-share data fetching, backtesting | `examples/prediction_example.py`, `examples/prediction_cn_markets_day.py` |
| `finetune/` | Qlib-based finetuning pipeline for A-share markets (tokenizer + predictor + backtest) | `finetune/config.py`, `finetune/dataset.py` |
| `finetune_csv/` | CSV-based finetuning pipeline for custom data (e.g., HK stocks) | `finetune_csv/config_loader.py`, `finetune_csv/configs/` |
| `webui/` | Flask web UI for interactive prediction with Plotly charts | `webui/app.py`, `webui/templates/index.html` |
| `tests/` | Regression tests for KronosPredictor | `tests/test_kronos_regression.py` |
| `figures/` | README images (logo, architecture overview, prediction examples) | `figures/overview.png` |
| `docs/` | Generated codebase documentation | `docs/codebase/` |

### 2) Entry Points

- Main runtime entry: Model import via `from model import Kronos, KronosTokenizer, KronosPredictor` — library-style, no `main()` dispatcher
- Secondary entry points:
  - `examples/prediction_example.py` — single-stock prediction demo
  - `examples/prediction_cn_markets_day.py --symbol CODE` — A-share daily prediction via akshare
  - `examples/prediction_akshare_2024-2025.py` — multi-stock A-share prediction with GUI charting
  - `examples/prediction_batch_example.py` — batch prediction demo (5 windows)
  - `webui/run.py` — web UI server
  - `finetune/train_tokenizer.py` — tokenizer finetuning (torchrun)
  - `finetune/train_predictor.py` — predictor finetuning (torchrun)
  - `finetune_csv/train_sequential.py` — CSV pipeline finetuning
- How entry is selected: Each script is standalone. Configuration via Python class (`finetune/config.py`) or YAML (`finetune_csv/configs/*.yaml`).

### 3) Module Boundaries

| Boundary | What belongs here | What must not be here |
|----------|-------------------|------------------------|
| `model/` | KronosTokenizer, Kronos predictor, KronosPredictor, neural network building blocks (TransformerBlock, BSQuantizer, embeddings) | Data fetching, UI logic, training loops |
| `examples/` | Standalone demonstration scripts with data fetching + inference + visualization | Shared library code (import from `model/` instead) |
| `finetune/` | Qlib-specific data preprocessing, training loops, backtesting for A-share daily data | General-purpose inference (use `model/`) |
| `finetune_csv/` | CSV-based custom data finetuning with YAML config | Qlib-specific logic (use `finetune/`) |
| `webui/` | Flask server, HTML templates, prediction result caching | Model logic (use `model/`) |
| `tests/` | Regression tests, test data fixtures | Production code |

### 4) Naming and Organization Rules

- File naming pattern: snake_case (`kronos.py`, `prediction_example.py`, `qlib_data_preprocess.py`)
- Directory organization pattern: functional — `model/` (core), `examples/` (demos), `finetune/` (training), `webui/` (UI), `tests/` (testing)
- Import aliasing or path conventions: Relative import for model package (`from model import Kronos...`); examples add parent to `sys.path` (`sys.path.append("../")`)

### 5) Evidence

- `model/__init__.py` — public API exports (KronosTokenizer, Kronos, KronosPredictor)
- `examples/` — 11 Python files covering prediction, data fetching, backtesting
- `finetune/config.py` — centralized config class
- `finetune_csv/configs/config_ali09988_candle-5min.yaml` — YAML config template
- `webui/app.py` — Flask app with 3 model variants
