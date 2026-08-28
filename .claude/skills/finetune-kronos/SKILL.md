---
name: finetune-kronos
description: >
  Kronos 模型 TDX（通达信）本地数据微调全流程：单阶段全参数微调、
  bf16 AMP、A股后复权日线。当用户提到 Kronos fine-tuning、TDX 数据导入、
  微调模型、后复权日线、训练 tokenizer/predictor、预测 A 股、续训/更新权重、
  模型训练时使用此技能。
  即使只提"微调"或"TDX数据"而不提 Kronos，也应触发。
---

# TDX本地数据 微调 Kronos（单阶段全参数微调）

基于 TDX（通达信）本地历史数据的 Kronos 模型领域自适应微调流程。
采用**单阶段全参数微调**策略，bf16 AMP 加速。

## 适用场景与前提仓库

本技能**不是**自包含的微调框架，而是一个 **Kronos 上游仓库 + TDX 适配脚本**协同使用的操作手册。冷启动时先确认宿主项目就绪：

```bash
# 1. 克隆 Kronos 上游
git clone https://github.com/shiyu-coder/Kronos.git
cd Kronos
```

2. 确保 `finetune/` 目录下存在以下 TDX 适配脚本（随宿主项目分发，**不在本技能包内**）：

| 文件                              | 作用                                 |
| --------------------------------- | ------------------------------------ |
| `finetune/config_tdx.py`          | 单卡微调配置（后复权、TDX 时间范围） |
| `finetune/train_tokenizer_tdx.py` | Tokenizer 单卡训练（bf16 AMP）       |
| `finetune/train_predictor_tdx.py` | Predictor 单阶段训练（bf16 AMP）     |
| `finetune/dataset.py`             | 数据集加载器                         |
| `model/kronos.py`                 | Kronos/KronosTokenizer 模型定义      |

## 前置条件

确认以下环境就绪后再开始：

| 条件          | 检查命令                                                          | 要求                                           |
| ------------- | ----------------------------------------------------------------- | ---------------------------------------------- |
| GPU           | `nvidia-smi`                                                      | >= 8GB VRAM（推荐 16GB 如 RTX 5080）           |
| PyTorch CUDA  | `python -c "import torch; print(torch.cuda.is_bf16_supported())"` | True（bf16 原生支持）                          |
| TDengine 连接 | `python -c "from taosws import connect; connect()"`               | 可连接                                         |
| 磁盘空间      | `df -h .`                                                         | >= 2GB（160MB 数据 + 425MB 模型 + 410MB 输出） |
| HF 镜像       | `curl -s --connect-timeout 5 https://hf-mirror.com`               | 可访问                                         |
| TDengine      | `python -c "from taosws import connect; connect()"`               | 已连接（数据源）                               |

### 首次环境初始化

```bash
# 1. 创建虚拟环境
python3 -m venv .venv

# 2. 写入 HF 国内镜像（必须在 source 之前）
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> .venv/bin/activate

# 3. 激活环境并安装依赖
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest pytest-timeout
```

**关键说明**：

- `HF_ENDPOINT` 必须写入 `activate` 脚本末尾，确保每次激活自动生效。
- RTX 5080 (Blackwell) 原生支持 bf16，不需要 GradScaler。

## 核心决策

这些决策已在流程中固定，不需要每次重新讨论：

- **复权方式**: 后复权 (hfq/back) — 匹配原始 Kronos Qlib 训练数据约定
- **模型**: Kronos-Tokenizer-base + Kronos-base（103M 参数）
- **数据周期**: 日线（1d），其他周期按需重采样
- **训练策略**: 单阶段全参数微调
- **精度**: bf16 AMP（RTX 5080 原生 bf16，不需要 GradScaler）

## 执行步骤

按顺序执行以下 4 步。每步完成后验证输出再进入下一步。

### Step 1: 数据导入

使用 `scripts/tdx_import.py` 从 TDengine 导入全量 A 股日线数据（后复权由 adjust 表事件实时计算）。

#### 切分原则（相对当前 TDX 数据末日，不要硬抄日期）

TDX 本地数据每天都在长，**不要把下面示例日期当永恒事实**。按以下原则在执行前**重新计算**：

- 设 `END = TDX 数据末日`（一般是今天或昨天交易日）
- **test**: `[END - 3 月, END]`
- **val**: `[END - 6 月, END - 3 月)` ← 与 test 不重叠
- **train**: `[数据起始（约 2011-01）, END - 6 月)` ← 与 val 不重叠

三段的**预测目标区间**严禁重叠。

```bash
# 生成股票列表
.venv/bin/python scripts/discover_stocks.py --output /tmp/tdx_all_stocks.txt

# 导入数据（后复权）
.venv/bin/python scripts/tdx_import.py \
  --symbol-file /tmp/tdx_all_stocks.txt \
  --dividend-type back \
  --periods 1d \
  --output-dir ./data/tdx_import \
  --train-range 2011-01-01 2025-10-31 \
  --val-range   2025-11-01 2026-02-14 \
  --test-range  2026-02-15 2026-05-16 \
  --no-continuity
```

**验证**:

```bash
ls -lh data/tdx_import/1d/train_data.pkl  # ~130MB
ls -lh data/tdx_import/1d/val_data.pkl    # ~65MB
ls -lh data/tdx_import/1d/test_data.pkl   # ~65MB
```

### Step 2: 模型下载

```bash
.venv/bin/python -c "
from huggingface_hub import hf_hub_download
for model_id in ['NeoQuasar/Kronos-Tokenizer-base', 'NeoQuasar/Kronos-base']:
    path = hf_hub_download(repo_id=model_id, filename='model.safetensors')
    print(f'{model_id}: {path}')
    hf_hub_download(repo_id=model_id, filename='config.json')
"
```

### Step 3: 微调训练

#### 3a. Tokenizer 微调 (30 epochs, ~16 分钟, bf16 AMP)

```bash
.venv/bin/python finetune/train_tokenizer_tdx.py \
  --data-dir ./data/tdx_import/1d \
  --epochs 30
```

#### 3b. Predictor 全参数微调 (10 epochs, ~35 分钟, bf16 AMP)

```bash
.venv/bin/python finetune/train_predictor_tdx.py \
  --data-dir ./data/tdx_import/1d
```

**验证**: Val Loss 应稳定下降（不出现过拟合）：

```bash
cat outputs/tdx_finetune/tdx_predictor/summary.json
# 期望: best_val_loss ~ 3.0x（全参数微调，稳定收敛）
```

### Step 4: 预测验证

技能内置 `scripts/predict_sse.py`，**直接从 TDengine 拉取上证指数**（表名 `k_sh999999_1d`，通达信别名 `sh999999`），无需先运行 `tdx_import.py`：

```bash
# 直接从 TDengine 拉取上证指数并预测（无需先 import）
python .claude/skills/finetune-kronos/scripts/predict_sse.py
```

> **说明**：`k_sh000001_1d` 表在 TDengine 中不存在，上证指数数据写入在 `k_sh999999_1d`（通达信别名）。
> 旧流程 `tdx_import.py --symbols sh000001` 会静默失败（表不存在被吞掉），`predict_stocks.py sh000001` 会回退到本地陈旧 pkl。
> 如需多股票预测 + md 报告，仍可使用 `scripts/predict_stocks.py sh999999`（走 `get_data` → `_fetch_index_from_tdengine` 直读 TDengine）。

## 关键文件清单

- `scripts/tdx_import.py` — TDX 数据导入工具
- `scripts/discover_stocks.py` — 股票代码枚举
- `finetune/config_tdx.py` — 微调配置
- `finetune/train_tokenizer_tdx.py` — Tokenizer bf16 AMP 训练
- `finetune/train_predictor_tdx.py` — Predictor 单阶段训练
- `finetune/dataset.py` — 数据集加载器
- `model/kronos.py` — Kronos/KronosTokenizer 模型定义

### `finetune/config_tdx.py` 关键字段速查

| 字段                      | 默认   | 说明                         |
| ------------------------- | ------ | ---------------------------- |
| `lookback_window`         | 90     | 模型可见的历史交易日数       |
| `predict_window`          | 10     | 训练时的预测窗口             |
| `dividend_type`           | "back" | 后复权                       |
| `batch_size`              | 128    | Tokenizer 批大小（bf16 AMP） |
| `predictor_batch_size`    | 64     | Predictor 批大小（bf16 AMP） |
| `tokenizer_learning_rate` | 2e-4   | Tokenizer 学习率             |
| `predictor_learning_rate` | 4e-5   | Predictor 学习率             |
| `epochs`                  | 10     | Predictor epoch 数           |
| `use_amp`                 | True   | bf16 AMP 开关                |

## 显存配置

在 RTX 5080 16GB 上实测：

| 配置       | Tokenizer  | Predictor |
| ---------- | ---------- | --------- |
| Batch size | 128 (bf16) | 64 (bf16) |
| 显存占用   | ~2.5 GB    | ~10.8 GB  |
| 每 Epoch   | 0.5 分钟   | 3.5 分钟  |

8GB GPU（RTX 4060）：降低 `batch_size` 至 32-50，Tokenizer 可保持 bs=64。

## 训练效果

| 方案               | Val Loss | 说明                 |
| ------------------ | -------- | -------------------- |
| 预训练 Kronos-base | ~4.2     | HuggingFace 预训练   |
| **全参数微调**     | **3.03** | ✅ 10 epoch 稳定下降 |

## 环境重建 / 项目迁移

```bash
rm -rf .venv
python3 -m venv .venv
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> .venv/bin/activate
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest pytest-timeout
```

## 常见陷阱

| 现象                                           | 根因                              | 修复                               |
| ---------------------------------------------- | --------------------------------- | ---------------------------------- |
| `from_pretrained` 报 positional arguments 错误 | `config.json` 下载失败，HF 不可达 | 确认 `HF_ENDPOINT` 已写入 activate |
| Val Loss 上升                                  | 学习率过高或 batch 太小           | lr=4e-5, bs=64                     |
| GPU OOM                                        | Batch size 超出显存               | 降低 bs 至 32-50                   |
| 复权因子缓存不一致                             | 行情修复导致因子漂移              | 删除缓存重导数据                   |
