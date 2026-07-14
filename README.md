<div align="center">
  <h2><b>Kronos: A Foundation Model for the Language of Financial Markets </b></h2>
</div>

<div align="center">

</a> 
<a href="https://huggingface.co/NeoQuasar"> 
<img src="https://img.shields.io/badge/🤗-Hugging_Face-yellow" alt="Hugging Face"> 
</a> 
<a href="https://shiyu-coder.github.io/Kronos-demo/"> <img src="https://img.shields.io/badge/🚀-Live_Demo-brightgreen" alt="Live Demo"> </a>
<a href="https://github.com/shiyu-coder/Kronos/graphs/commit-activity"> 
<img src="https://img.shields.io/github/last-commit/shiyu-coder/Kronos?color=blue" alt="Last Commit"> 
</a> 
<a href="https://github.com/shiyu-coder/Kronos/stargazers"> 
<img src="https://img.shields.io/github/stars/shiyu-coder/Kronos?color=lightblue" alt="GitHub Stars"> 
</a> 
<a href="https://github.com/shiyu-coder/Kronos/network/members"> 
<img src="https://img.shields.io/github/forks/shiyu-coder/Kronos?color=yellow" alt="GitHub Forks"> 
</a> 
<a href="./LICENSE"> 
<img src="https://img.shields.io/github/license/shiyu-coder/Kronos?color=green" alt="License"> 
</a>

</div>

<div align="center">
  <!-- Keep these links. Translations will automatically update with the README. -->
  <a href="https://zdoc.app/de/shiyu-coder/Kronos">Deutsch</a> | 
  <a href="https://zdoc.app/es/shiyu-coder/Kronos">Español</a> | 
  <a href="https://zdoc.app/fr/shiyu-coder/Kronos">Français</a> | 
  <a href="https://zdoc.app/ja/shiyu-coder/Kronos">日本語</a> | 
  <a href="https://zdoc.app/ko/shiyu-coder/Kronos">한국어</a> | 
  <a href="https://zdoc.app/pt/shiyu-coder/Kronos">Português</a> | 
  <a href="https://zdoc.app/ru/shiyu-coder/Kronos">Русский</a> | 
  <a href="https://zdoc.app/zh/shiyu-coder/Kronos">中文</a>
</div>

<p align="center">

<img src="./figures/logo.png" width="100">

</p>

> Kronos is the **first open-source foundation model** for financial candlesticks (K-lines),
> trained on data from over **45 global exchanges**.

</div>

## 📰 News

- 🚩 **[2025.11.10]** Kronos has been accpeted by AAAI 2026.
- 🚩 **[2025.08.17]** We have released the scripts for fine-tuning! Check them out to adapt Kronos to your own tasks.
- 🚩 **[2025.08.02]** Our paper is now available on [arXiv](https://arxiv.org/abs/2508.02739)!

<p align="center">

## 📜 Introduction

**Kronos** is a family of decoder-only foundation models, pre-trained specifically for the "language" of financial markets—K-line sequences. Unlike general-purpose TSFMs, Kronos is designed to handle the unique, high-noise characteristics of financial data. It leverages a novel two-stage framework:

1. A specialized tokenizer first quantizes continuous, multi-dimensional K-line data (OHLCV) into **hierarchical discrete tokens**.
2. A large, autoregressive Transformer is then pre-trained on these tokens, enabling it to serve as a unified model for diverse quantitative tasks.

<p align="center">
    <img src="figures/overview.png" alt="" align="center" width="700px" />
</p>

## ✨ Live Demo

We have set up a live demo to visualize Kronos's forecasting results. The webpage showcases a forecast for the **BTC/USDT** trading pair over the next 24 hours.

**👉 [Access the Live Demo Here](https://shiyu-coder.github.io/Kronos-demo/)**

## 📦 Model Zoo

We release a family of pre-trained models with varying capacities to suit different computational and application needs. All models are readily accessible from the Hugging Face Hub.

| Model        | Tokenizer                                                                       | Context length | Params | Open-source                                                                |
| ------------ | ------------------------------------------------------------------------------- | -------------- | ------ | -------------------------------------------------------------------------- |
| Kronos-mini  | [Kronos-Tokenizer-2k](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-2k)     | 2048           | 4.1M   | ✅ [NeoQuasar/Kronos-mini](https://huggingface.co/NeoQuasar/Kronos-mini)   |
| Kronos-small | [Kronos-Tokenizer-base](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base) | 512            | 24.7M  | ✅ [NeoQuasar/Kronos-small](https://huggingface.co/NeoQuasar/Kronos-small) |
| Kronos-base  | [Kronos-Tokenizer-base](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base) | 512            | 102.3M | ✅ [NeoQuasar/Kronos-base](https://huggingface.co/NeoQuasar/Kronos-base)   |
| Kronos-large | [Kronos-Tokenizer-base](https://huggingface.co/NeoQuasar/Kronos-Tokenizer-base) | 512            | 499.2M | ❌                                                                         |

## 🚀 Getting Started

### Installation

1. Install Python 3.10+, and then install the dependencies:

```shell
pip install -r requirements.txt
```

### 📈 Making Forecasts

Forecasting with Kronos is straightforward using the `KronosPredictor` class. It handles data preprocessing, normalization, prediction, and inverse normalization, allowing you to get from raw data to forecasts in just a few lines of code.

**Important Note**: The `max_context` for `Kronos-small` and `Kronos-base` is **512**. This is the maximum sequence length the model can process. For optimal performance, it is recommended that your input data length (i.e., `lookback`) does not exceed this limit. The `KronosPredictor` will automatically handle truncation for longer contexts.

Here is a step-by-step guide to making your first forecast.

#### 1. Load the Tokenizer and Model

First, load a pre-trained Kronos model and its corresponding tokenizer from the Hugging Face Hub.

```python
from model import Kronos, KronosTokenizer, KronosPredictor

# Load from Hugging Face Hub
tokenizer = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
model = Kronos.from_pretrained("NeoQuasar/Kronos-small")
```

#### 2. Instantiate the Predictor

Create an instance of `KronosPredictor`, passing the model, tokenizer, and desired device.

```python
# Initialize the predictor
predictor = KronosPredictor(model, tokenizer, max_context=512)
```

#### 3. Prepare Input Data

The `predict` method requires three main inputs:

- `df`: A pandas DataFrame containing the historical K-line data. It must include columns `['open', 'high', 'low', 'close']`. `volume` and `amount` are optional.
- `x_timestamp`: A pandas Series of timestamps corresponding to the historical data in `df`.
- `y_timestamp`: A pandas Series of timestamps for the future periods you want to predict.

```python
import pandas as pd

# Load your data
df = pd.read_csv("./data/XSHG_5min_600977.csv")
df['timestamps'] = pd.to_datetime(df['timestamps'])

# Define context window and prediction length
lookback = 400
pred_len = 120

# Prepare inputs for the predictor
x_df = df.loc[:lookback-1, ['open', 'high', 'low', 'close', 'volume', 'amount']]
x_timestamp = df.loc[:lookback-1, 'timestamps']
y_timestamp = df.loc[lookback:lookback+pred_len-1, 'timestamps']
```

#### 4. Generate Forecasts

Call the `predict` method to generate forecasts. You can control the sampling process with parameters like `T`, `top_p`, and `sample_count` for probabilistic forecasting.

```python
# Generate predictions
pred_df = predictor.predict(
    df=x_df,
    x_timestamp=x_timestamp,
    y_timestamp=y_timestamp,
    pred_len=pred_len,
    T=1.0,          # Temperature for sampling
    top_p=0.9,      # Nucleus sampling probability
    sample_count=1  # Number of forecast paths to generate and average
)

print("Forecasted Data Head:")
print(pred_df.head())
```

The `predict` method returns a pandas DataFrame containing the forecasted values for `open`, `high`, `low`, `close`, `volume`, and `amount`, indexed by the `y_timestamp` you provided.

For efficient processing of multiple time series, Kronos provides a `predict_batch` method that enables parallel prediction on multiple datasets simultaneously. This is particularly useful when you need to forecast multiple assets or time periods at once.

```python
# Prepare multiple datasets for batch prediction
df_list = [df1, df2, df3]  # List of DataFrames
x_timestamp_list = [x_ts1, x_ts2, x_ts3]  # List of historical timestamps
y_timestamp_list = [y_ts1, y_ts2, y_ts3]  # List of future timestamps

# Generate batch predictions
pred_df_list = predictor.predict_batch(
    df_list=df_list,
    x_timestamp_list=x_timestamp_list,
    y_timestamp_list=y_timestamp_list,
    pred_len=pred_len,
    T=1.0,
    top_p=0.9,
    sample_count=1,
    verbose=True
)

# pred_df_list contains prediction results in the same order as input
for i, pred_df in enumerate(pred_df_list):
    print(f"Predictions for series {i}:")
    print(pred_df.head())
```

**Important Requirements for Batch Prediction:**

- All series must have the same historical length (lookback window)
- All series must have the same prediction length (`pred_len`)
- Each DataFrame must contain the required columns: `['open', 'high', 'low', 'close']`
- `volume` and `amount` columns are optional and will be filled with zeros if missing

The `predict_batch` method leverages GPU parallelism for efficient processing and automatically handles normalization and denormalization for each series independently.

#### 5. Example and Visualization

For a complete, runnable script that includes data loading, prediction, and plotting, please see [`examples/prediction_example.py`](examples/prediction_example.py).

Running this script will generate a plot comparing the ground truth data against the model's forecast, similar to the one shown below:

<p align="center">
    <img src="figures/prediction_example.png" alt="Forecast Example" align="center" width="600px" />
</p>

Additionally, we provide a script that makes predictions without Volume and Amount data, which can be found in [`examples/prediction_wo_vol_example.py`](examples/prediction_wo_vol_example.py).

## 🔧 Finetuning on Your Own Data (A-Share Market Example)

We provide a complete pipeline for finetuning Kronos on your own datasets. As an example, we demonstrate how to use [Qlib](https://github.com/microsoft/qlib) to prepare data from the Chinese A-share market and conduct a simple backtest.

> **Disclaimer:** This pipeline is intended as a demonstration to illustrate the finetuning process. It is a simplified example and not a production-ready quantitative trading system. A robust quantitative strategy requires more sophisticated techniques, such as portfolio optimization and risk factor neutralization, to achieve stable alpha.

The finetuning process is divided into four main steps:

1.  **Configuration**: Set up paths and hyperparameters.
2.  **Data Preparation**: Process and split your data using Qlib.
3.  **Model Finetuning**: Finetune the Tokenizer and the Predictor models.
4.  **Backtesting**: Evaluate the finetuned model's performance.

### Prerequisites

1.  First, ensure you have all dependencies from `requirements.txt` installed.
2.  This pipeline relies on `qlib`. Please install it:
    ```shell
      pip install pyqlib
    ```
3.  You will need to prepare your Qlib data. Follow the [official Qlib guide](https://github.com/microsoft/qlib) to download and set up your data locally. The example scripts assume you are using daily frequency data.

### Step 1: Configure Your Experiment

All settings for data, training, and model paths are centralized in `finetune/config.py`. Before running any scripts, please **modify the following paths** according to your environment:

- `qlib_data_path`: Path to your local Qlib data directory.
- `dataset_path`: Directory where the processed train/validation/test pickle files will be saved.
- `save_path`: Base directory for saving model checkpoints.
- `backtest_result_path`: Directory for saving backtesting results.
- `pretrained_tokenizer_path` and `pretrained_predictor_path`: Paths to the pre-trained models you want to start from (can be local paths or Hugging Face model names).

You can also adjust other parameters like `instrument`, `train_time_range`, `epochs`, and `batch_size` to fit your specific task. If you don't use [Comet.ml](https://www.comet.com/), set `use_comet = False`.

### Step 2: Prepare the Dataset

Run the data preprocessing script. This script will load raw market data from your Qlib directory, process it, split it into training, validation, and test sets, and save them as pickle files.

```shell
python finetune/qlib_data_preprocess.py
```

After running, you will find `train_data.pkl`, `val_data.pkl`, and `test_data.pkl` in the directory specified by `dataset_path` in your config.

### Step 3: Run the Finetuning

The finetuning process consists of two stages: finetuning the tokenizer and then the predictor. Both training scripts are designed for multi-GPU training using `torchrun`.

#### 3.1 Finetune the Tokenizer

This step adjusts the tokenizer to the data distribution of your specific domain.

```shell
# Replace NUM_GPUS with the number of GPUs you want to use (e.g., 2)
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_tokenizer.py
```

The best tokenizer checkpoint will be saved to the path configured in `config.py` (derived from `save_path` and `tokenizer_save_folder_name`).

#### 3.2 Finetune the Predictor

This step finetunes the main Kronos model for the forecasting task.

```shell
# Replace NUM_GPUS with the number of GPUs you want to use (e.g., 2)
torchrun --standalone --nproc_per_node=NUM_GPUS finetune/train_predictor.py
```

The best predictor checkpoint will be saved to the path configured in `config.py`.

### Step 4: Evaluate with Backtesting

Finally, run the backtesting script to evaluate your finetuned model. This script loads the models, performs inference on the test set, generates prediction signals (e.g., forecasted price change), and runs a simple top-K strategy backtest.

```shell
# Specify the GPU for inference
python finetune/qlib_test.py --device cuda:0
```

The script will output a detailed performance analysis in your console and generate a plot showing the cumulative return curves of your strategy against the benchmark, similar to the one below:

<p align="center">
    <img src="figures/backtest_result_example.png" alt="Backtest Example" align="center" width="700px" />
</p>

## 🔌 TDX Local Data Fine-tuning (Single GPU)

We provide tools to fine-tune Kronos using local TDX (通达信) historical data — no cloud data download required.

### Quick Start

```bash
# Step 1: Import A-share data from local TDX (后复权)
python scripts/tdx_import.py \
  --dividend-type back --periods 1d \
  --output-dir ./data/tdx_import

# Step 2: Fine-tune Tokenizer (30 epochs, ~0.9h, single GPU)
python finetune/train_tokenizer_tdx.py \
  --data-dir ./data/tdx_import/1d --epochs 30

# Step 3: Fine-tune Predictor (30 epochs, ~5.4h, single GPU)
python finetune/train_predictor_tdx.py \
  --data-dir ./data/tdx_import/1d \
  --tokenizer-path ./outputs/tdx_finetune/tdx_tokenizer/checkpoints/best_model \
  --epochs 30

# Step 4: Verify with SSE Index prediction
python .claude/skills/finetune-kronos/scripts/predict_sse.py
```

### What's Included

| File                              | Purpose                                                                     |
| --------------------------------- | --------------------------------------------------------------------------- |
| `scripts/tdx_import.py`           | Import local TDX data with configurable price adjustment and factor caching |
| `finetune/config_tdx.py`          | Single-GPU config (后复权, RTX 4060 8GB tuned)                              |
| `finetune/train_tokenizer_tdx.py` | Tokenizer single-GPU training script                                        |
| `finetune/train_predictor_tdx.py` | Predictor single-GPU training script (AMP fp16)                             |
| `.claude/skills/finetune-kronos/scripts/predict_sse.py` | Prediction demo on SSE Composite Index (reads sh999999 from TDengine) |
| `summary.md`                      | Full workflow summary and results                                           |

### Hardware Requirements

- GPU with >= 8GB VRAM (tested on RTX 4060 Laptop)
- ~2GB disk for data, ~425MB for models, ~410MB for outputs
- Internet access for one-time factor fetching from Sina Finance

### 💡 From Demo to Production: Important Considerations

- **Raw Signals vs. Pure Alpha**: The signals generated by the model in this demo are raw predictions. In a real-world quantitative workflow, these signals would typically be fed into a portfolio optimization model. This model would apply constraints to neutralize exposure to common risk factors (e.g., market beta, style factors like size and value), thereby isolating the **"pure alpha"** and improving the strategy's robustness.
- **Data Handling**: The provided `QlibDataset` is an example. For different data sources or formats, you will need to adapt the data loading and preprocessing logic.
- **Strategy and Backtesting Complexity**: The simple top-K strategy used here is a basic starting point. Production-level strategies often incorporate more complex logic for portfolio construction, dynamic position sizing, and risk management (e.g., stop-loss/take-profit rules). Furthermore, a high-fidelity backtest should meticulously model transaction costs, slippage, and market impact to provide a more accurate estimate of real-world performance.

> **📝 AI-Generated Comments**: Please note that many of the code comments within the `finetune/` directory were generated by an AI assistant (Gemini 2.5 Pro) for explanatory purposes. While they aim to be helpful, they may contain inaccuracies. We recommend treating the code itself as the definitive source of logic.

## 🚀 TDX 数据微调与预测

基于通达信本地日线数据的 Kronos 模型微调与预测。

### 环境激活

```bash
# 激活虚拟环境（含 HF 国内镜像）
source .venv/bin/activate

# 或直接用 venv python（无需激活）
.venv/bin/python scripts/predict_stocks.py sz002741
```

### 一键预测

```bash
# 自动读取 TDX 自选股，输出 md 报告
.venv/bin/python scripts/predict_stocks.py

# 预测指定股票，输出 md 报告
.venv/bin/python scripts/predict_stocks.py sh600000 sz002741

# 控制台表格输出
.venv/bin/python scripts/predict_stocks.py --format console

# 指定输出路径
.venv/bin/python scripts/predict_stocks.py sz002741 -o outputs/my_pred.md
```

无参数时自动读取通达信自选股（zxg.blk）。报告含未来 10 日收盘价预测（实际市场价）、历史回测准确度、置信区间，按指数→看涨→看平→看跌分类排序。

### WebUI

```bash
bash start.sh -d        # 后台启动 (端口 7070)
bash start.sh stop      # 停止
bash start.sh status    # 状态
```

### 数据导入

```bash
# 生成股票列表
.venv/bin/python scripts/discover_stocks.py --output /tmp/stocks.txt

# 导入后复权日线
.venv/bin/python scripts/tdx_import.py \
  --symbol-file /tmp/stocks.txt \
  --dividend-type back \
  --periods 1d \
  --output-dir ./data/tdx_import \
  --train-range 2011-01-01 2025-10-31 \
  --val-range   2025-11-01 2026-02-14 \
  --test-range  2026-02-15 2026-05-17
```

### 微调训练

```bash
# Tokenizer (~0.9h)
.venv/bin/python finetune/train_tokenizer_tdx.py --data-dir ./data/tdx_import/1d --epochs 30

# Predictor (~5.4h)
.venv/bin/python finetune/train_predictor_tdx.py \
  --data-dir ./data/tdx_import/1d \
  --tokenizer-path ./outputs/tdx_finetune/tdx_tokenizer/checkpoints/best_model \
  --epochs 30
```

---

## 📖 Citation

If you use Kronos in your research, we would appreciate a citation to our [paper](https://arxiv.org/abs/2508.02739):

```
@misc{shi2025kronos,
      title={Kronos: A Foundation Model for the Language of Financial Markets},
      author={Yu Shi and Zongliang Fu and Shuo Chen and Bohan Zhao and Wei Xu and Changshui Zhang and Jian Li},
      year={2025},
      eprint={2508.02739},
      archivePrefix={arXiv},
      primaryClass={q-fin.ST},
      url={https://arxiv.org/abs/2508.02739},
}
```

## 📜 License

This project is licensed under the [MIT License](./LICENSE).

### 2026-05-05 09:35:17

```
 .gitignore                                   |    3 +
 docs/codebase/ARCHITECTURE.md                |   79 ++
 docs/codebase/CONCERNS.md                    |   68 ++
 docs/codebase/CONVENTIONS.md                 |   50 ++
 docs/codebase/INTEGRATIONS.md                |   48 ++
 docs/codebase/STACK.md                       |   74 ++
 docs/codebase/STRUCTURE.md                   |   55 ++
 docs/codebase/TESTING.md                     |   60 ++
 docs/codebase/WORKFLOWS.md                   |  214 ++++++
 requirements.txt                             |    2 +-
 summary.md                                   |  183 +----
 tdxdata/.gitignore                           |   14 +
 tdxdata/.trae/rules/project_rules.md         |   59 ++
 tdxdata/README.md                            |  367 ++++++++++
 tdxdata/docs/PRD.md                          |  209 ++++++
```

### 2026-05-16 18:08:36

```
 .claude/skills/check-kronos-env/SKILL.md           |  165 +
 .claude/skills/finetune-kronos/SKILL.md            |  341 ++
 .claude/skills/finetune-kronos/evals/evals.json    |   23 +
 .../finetune-kronos/evals/trigger_evals.json       |   30 +
 .../finetune-kronos/scripts/discover_stocks.py     |   97 +
 .../skills/finetune-kronos/scripts/predict_sse.py  |  129 +
 .../skills/finetune-kronos/scripts/tdx_import.py   |  926 +++++
 .gitignore                                         |    0
 LICENSE                                            |    0
 README.md                                          |   19 +
 data/tdx_import_sse/1d/data.pkl                    |  Bin 0 -> 27475 bytes
 docs/codebase/ARCHITECTURE.md                      |    0
 docs/codebase/CONCERNS.md                          |    0
 docs/codebase/CONVENTIONS.md                       |    0
 docs/codebase/INTEGRATIONS.md                      |    0
```

### 2026-05-16 18:18:17

```
 .gitignore                                         |    5 +
 data/tdx_import_sse/1d/data.pkl                    |  Bin 27475 -> 0 bytes
 outputs/pred_sh000001_20260506.csv                 |   31 -
 outputs/pred_sh000001_20260506_chart.html          | 3888 --------------------
 outputs/pred_sh600000_20260506.csv                 |   31 -
 outputs/pred_sh600000_20260506_chart.html          | 3888 --------------------
 outputs/pred_sh999999_20260506.csv                 |   31 -
 outputs/pred_sh999999_20260506_chart.html          | 3888 --------------------
 outputs/pred_sz002741_20260506.csv                 |   31 -
 outputs/pred_sz002741_20260506_chart.html          | 3888 --------------------
 .../tdx_predictor/checkpoints/best_model/README.md |   10 -
 .../checkpoints/best_model/config.json             |   13 -
 .../tdx_tokenizer/checkpoints/best_model/README.md |   10 -
 .../checkpoints/best_model/config.json             |   18 -
 14 files changed, 5 insertions(+), 15727 deletions(-)
```

### 2026-05-17 08:31:18

```
 README.md                 | 142 ++++++++++++++++---
 scripts/predict_stocks.py | 338 ++++++++++++++++++++++++++++++++++++++++++++++
 2 files changed, 461 insertions(+), 19 deletions(-)
```

### 2026-05-17 09:37:31

```
 .claude/skills/finetune-kronos/SKILL.md            |  19 +-
 .../finetune-kronos/evals/trigger_evals.json       |  48 +++---
 README.md                                          |   7 +
 finetune/config_tdx.py                             |  22 +--
 finetune/train_predictor_tdx.py                    |  13 +-
 finetune/train_tokenizer_tdx.py                    |  15 +-
 scripts/predict.py                                 |  63 ++++++-
 scripts/tdx_import.py                              |  14 +-
 tdxdata/pyproject.toml                             |   5 +-
 tdxdata/tdxdata/__init__.py                        |   5 +-
 tdxdata/tdxdata/api.py                             |  73 +++++---
 tdxdata/tdxdata/core/__init__.py                   |   3 +-
 tdxdata/tdxdata/core/connection.py                 |  60 ++++++-
 tdxdata/tdxdata/core/data_manager.py               |  89 ++++++++--
 tdxdata/tdxdata/core/registry.py                   |   3 -
```


### 2026-06-29 (预测脚本精简)

精简 `scripts/predict_stocks.py`：
- 删除 `fetch_stock_names()` 无操作函数，预测流程直接使用股票代码
- 移除 `name_map` 变量，消除 `sh600000 (sh600000)` 冗余输出
- 复权因子确认从 TDengine 获取，无需网络下载

### 2026-06-29 09:23:46
```
 .claude/skills/check-kronos-env/SKILL.md                     |   8 +-
 .claude/skills/finetune-kronos/SKILL.md                      | 158 +++++---------------
 .claude/skills/finetune-kronos/references/lessons-learned.md |  95 +++---------
 .claude/skills/finetune-kronos/scripts/tdx_import.py         | 648 ++++++++++++++++++++++++++++++++++++++-------------------------------------------
 CLAUDE.md                                                    |   6 +-
 README.md                                                    |  24 ---
 docs/fusion_czsc_plan.md                                     | 509 ----------------------------------------------------------------
 docs/prediction_data_files.md                                |  44 +++---
 docs/scripts_guide.md                                        |  19 ++-
 docs/two_phase_finetune_plan.md                              | 237 ------------------------------
 finetune/train_predictor_tdx.py                              | 147 ++++++++++++++++++-
 finetune/train_tokenizer_tdx.py                              |   7 +-
 model/__init__.py                                            |   2 +-
 pyproject.toml                                               |   6 +-
 requirements.txt                                             |   5 +-
```

### 2026-06-29

修复: 导出脚本补全 ETF/基金/指数代码前缀映射

- `scripts/tdx_export_from_tdengine.py` — `get_all_stocks()` 新增 sh(`5`), sz(`18`,`39`) 前缀
- 修复 sh520620, sh589960, sz159731 等 ETF/基金/指数无法导出问题
- 全部 5286 只 TDengine 代码覆盖验证通过

### 2026-07-01 15:16:15
```
 scripts/predict_stocks.py | 131 ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++----
 1 file changed, 127 insertions(+), 4 deletions(-)
```

### 2026-07-08

修复: 适配 tdx-cpp TDengine 子表市场前缀化（v0.13.6/7/8）

- 子表名改用带市场前缀 symbol 拼装：`k_{sh|sz|bj}{code}_1d` / `a_{sh|sz|bj}{code}`（旧 `k_000001_1d` / `a_000001` 已废弃）
- `tdx_export_from_tdengine.py` / `tdx_import.py` / `predict.py` / `realtime.py` / `predict_stocks.py` 全部停止剥离前缀
- `stock_name` 查询加 `market` 过滤，sh000001(上证指数)/sz000001(平安银行) 精确区分
- 顺带修复：上证指数（sh000001）此前因 code 前缀推断丢失，现两市 000001 分别导出
- 删除因此变死的 `_resolve_market`/`_code_to_symbol`/`_symbol_to_code`
- 验证：pytest 41 passed + 连真实库冒烟（15953 symbols，两市 000001 精确区分）

