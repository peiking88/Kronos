# Testing Patterns

## Core Sections (Required)

### 1) Test Stack and Commands

- Primary test framework: pytest (discovery-based, no config file)
- Assertion/mocking tools: `numpy.testing.assert_allclose`, standard `assert`. No mocking — tests use real model inference.
- Commands:

```bash
# Run all tests
cd /home/li/peiking88/Kronos && .venv/bin/python -m pytest tests/ -v

# Run specific test
.venv/bin/python -m pytest tests/test_kronos_regression.py::test_kronos_predictor_regression -v

# Run with coverage (pytest-cov not installed)
[TODO]
```

### 2) Test Layout

- Test file placement pattern: `tests/` directory at project root
- Naming convention: `test_*.py` (pytest auto-discovery)
- Setup files and where they run: None — test data loaded inline from `tests/data/`

### 3) Test Scope Matrix

| Scope       | Covered? | Typical target                       | Notes                                                                      |
| ----------- | -------- | ------------------------------------ | -------------------------------------------------------------------------- |
| Unit        | No       | [TODO]                               | No unit tests for individual modules (BSQuantizer, TransformerBlock, etc.) |
| Integration | Partial  | KronosPredictor end-to-end inference | Only regression tests (output comparison) + MSE validation exist           |
| E2E         | No       | [TODO]                               | No end-to-end test from data fetch → prediction → visualization            |

### 4) Mocking and Isolation Strategy

- Main mocking approach: No mocking. Tests load real models from HuggingFace Hub at pinned revisions.
- Isolation guarantees: `set_seed(123)` ensures deterministic random state. `torch.no_grad()` + `model.eval()` ensures deterministic inference.
- Common failure mode in tests: Model revision changes invalidate expected regression outputs (`regression_output_*.csv`). Network dependency on HuggingFace Hub.

### 5) Coverage and Quality Signals

- Coverage tool + threshold: [TODO] — pytest-cov not installed. Project rule requires >80%.
- Current reported coverage: [TODO] — not measured
- Known gaps/flaky areas:
  - No unit tests for `model/module.py` (BSQuantizer, TransformerBlock, RoPE, DualHead, etc.)
  - No tests for `KronosTokenizer.encode()` / `decode()` in isolation
  - No tests for `predict_batch()` method
  - No tests for finetuning training loops
  - Only 2 test functions total (`test_kronos_predictor_regression`, `test_kronos_predictor_mse`)
- Test data only covers `regression_input.csv` (90KB). No multi-market, multi-frequency test fixtures.

### 6) Evidence

- `tests/test_kronos_regression.py` — 2 test functions, 141 lines
- `tests/data/regression_input.csv` — test input (XSHG_5min data)
- `tests/data/regression_output_256.csv` — expected output for context_len=256
- `tests/data/regression_output_512.csv` — expected output for context_len=512
- `tests/data/generate_regression_output.py` — script to regenerate expected outputs
