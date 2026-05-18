# Technology Stack

## Core Sections (Required)

### 1) Runtime Summary

| Area                | Value                                         | Evidence                                                   |
| ------------------- | --------------------------------------------- | ---------------------------------------------------------- |
| Primary language    | Python 3                                      | `requirements.txt`                                         |
| Runtime + version   | Python 3.14.4 (dev), README states 3.10+      | `README.md:89` (3.10+ min), terminal (`python3 --version`) |
| Package manager     | pip (venv)                                    | `requirements.txt`, `.venv/`                               |
| Module/build system | none (no setup.py/pyproject.toml for the app) | `requirements.txt` only                                    |

### 2) Production Frameworks and Dependencies

| Dependency      | Version                     | Role in system                                                   | Evidence                                                 |
| --------------- | --------------------------- | ---------------------------------------------------------------- | -------------------------------------------------------- |
| PyTorch         | >=2.0.0 (2.11.0 installed)  | Deep learning framework — model definition, training, inference  | `requirements.txt:4`, `model/kronos.py:3`                |
| NumPy           | (unpinned, 2.4.4 installed) | Numerical array operations, normalization                        | `requirements.txt:1`, `model/kronos.py:1`                |
| pandas          | 3.0.2                       | Data loading, timestamp handling, feature DataFrame construction | `requirements.txt:8`, `model/kronos.py:2`                |
| einops          | 0.8.1                       | Tensor rearrangement (rearrange, reduce) for quantizer           | `requirements.txt:5`, `model/module.py:3`                |
| huggingface_hub | 0.33.1                      | Model/tokenizer download and sharing (PyTorchModelHubMixin)      | `requirements.txt:6`, `model/kronos.py:4`                |
| matplotlib      | 3.9.3                       | Visualization (prediction charts, backtest curves)               | `requirements.txt:7`, `examples/prediction_example.py:2` |
| tqdm            | 4.67.1                      | Progress bars for autoregressive inference                       | `requirements.txt:9`, `model/kronos.py:7`                |
| safetensors     | 0.6.2                       | Safe model weight serialization (HuggingFace Hub)                | `requirements.txt:10`                                    |

### 3) Development Toolchain

| Tool     | Purpose                                       | Evidence                            |
| -------- | --------------------------------------------- | ----------------------------------- |
| pytest   | Test runner (no config file; uses convention) | `tests/test_kronos_regression.py:7` |
| PyYAML   | CSV finetuning config parsing                 | `finetune_csv/config_loader.py:2`   |
| torchrun | Multi-GPU distributed training for finetuning | `README.md:271,282`                 |

### 4) Key Commands

```bash
# Create venv and install
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Single prediction example
python examples/prediction_example.py

# A-share daily prediction (uses akshare)
python examples/prediction_cn_markets_day.py --symbol 000001

# Finetuning (Qlib pipeline)
python finetune/qlib_data_preprocess.py
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_tokenizer.py
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_predictor.py
python finetune/qlib_test.py --device cuda:0

# WebUI
python webui/run.py
```

### 5) Environment and Config

- Config sources: `finetune/config.py` (Qlib pipeline), `finetune_csv/configs/*.yaml` (CSV pipeline), `webui/start.sh`
- Required env vars: `GIT_USERNAME`, `GIT_TOKEN` (for git push); Comet ML API key optional
- Deployment/runtime constraints: CUDA GPU recommended for inference (CPU fallback available). Model weights ~100MB-1GB from HuggingFace Hub.

### 6) Evidence

- `requirements.txt`
- `model/__init__.py`
- `model/kronos.py`
- `model/module.py`
- `webui/requirements.txt`
- Terminal: `python3 --version`, `nvidia-smi`
