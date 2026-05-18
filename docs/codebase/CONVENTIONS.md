# Coding Conventions

## Core Sections (Required)

### 1) Naming Rules

| Item              | Rule              | Example                                                                                                | Evidence                                                                               |
| ----------------- | ----------------- | ------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| Files             | snake_case        | `kronos.py`, `prediction_example.py`, `qlib_data_preprocess.py`, `config_loader.py`                    | file listings                                                                          |
| Functions/methods | snake_case        | `calc_time_stamps()`, `auto_regressive_inference()`, `top_k_top_p_filtering()`, `prepare_stock_data()` | `model/kronos.py:472`, `examples/prediction_akshare_2024-2025.py:28`                   |
| Classes           | PascalCase        | `KronosTokenizer`, `KronosPredictor`, `ConfigLoader`, `BinarySphericalQuantizer`                       | `model/kronos.py:13,180,482`, `model/module.py:39`                                     |
| Constants         | UPPER_SNAKE       | `TEST_CTX_LEN`, `MODEL_REVISION`, `TOKENIZER_PRETRAINED`, `MSE_TOLERANCE`                              | `tests/test_kronos_regression.py:17-28`, `examples/prediction_cn_markets_day.py:38-46` |
| Private methods   | underscore prefix | `_init_weights()`, `_load_config()`, `_update_cos_sin_cache()`, `_set_benchmark()`                     | `model/kronos.py:225`, `finetune_csv/config_loader.py:13`, `model/module.py:293`       |

### 2) Formatting and Linting

- Formatter: [TODO] — no formatter config found in project root
- Linter: [TODO] — no linter config found in project root
- Most relevant enforced rules: [TODO] — no automated enforcement detected
- Run commands: [TODO]

### 3) Import and Module Conventions

- Import grouping/order: Standard library → third-party → local. Examples adapt `sys.path.append("../")` to import `model` from sibling directories.
- Alias vs relative import policy: `from model import Kronos, KronosTokenizer, KronosPredictor` — absolute import from project-level `model/` package. Third-party aliases common: `import torch.nn as nn`, `import torch.nn.functional as F`, `import numpy as np`, `import pandas as pd`.
- Public exports/barrel policy: `model/__init__.py` re-exports `KronosTokenizer`, `Kronos`, `KronosPredictor` + `model_dict` registry.

### 4) Error and Logging Conventions

- Error strategy by layer:
  - Predictor layer: `ValueError` for invalid input (missing columns, NaN, type mismatch)
  - Data loading: `sys.exit(1)` on unrecoverable data fetch failures
  - Training: Let PyTorch exceptions propagate
  - WebUI: Graceful fallback to simulated data when model unavailable
- Logging style and required context fields: Uses `print()` with emoji prefixes (`✅`, `❌`, `📊`, `🎯`, `🔮`) for user-facing scripts. Training uses `tqdm.trange` for progress.
- Sensitive-data redaction rules: [TODO] — no explicit redaction found; Comet ML API key placeholder in config.

### 5) Testing Conventions

- Test file naming/location rule: `tests/test_*.py` (pytest convention discovery)
- Mocking strategy norm: Tests use real model inference with pinned HuggingFace revisions. No mocking of model components.
- Coverage expectation: > 80% target (per project CLAUDE rules). Current coverage: [TODO]

### 6) Evidence

- `model/__init__.py` — barrel exports, `model_dict` registry
- `model/kronos.py:1-12` — import structure
- `model/module.py:1-8` — import structure
- `tests/test_kronos_regression.py` — test conventions
- `finetune_csv/config_loader.py` — private method convention (`_load_config`, `_resolve_dynamic_paths`)
