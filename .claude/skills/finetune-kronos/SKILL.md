---
name: finetune-kronos
description: >
  Kronos 模型 TDX（通达信）本地数据微调全流程：单卡 8GB GPU 训练、A股后复权日线、
  从数据导入到预测验证。当用户提到 Kronos fine-tuning、TDX 数据导入、微调模型、
  后复权日线、训练 tokenizer/predictor、预测 A 股、续训/更新权重、模型训练时使用此技能。
  即使只提"微调"或"TDX数据"而不提 Kronos，也应触发。
---

# TDX本地数据 微调 Kronos

基于 TDX（通达信）本地历史数据的 Kronos 模型领域自适应微调流程。

## 适用场景与前提仓库

本技能**不是**自包含的微调框架，而是一个 **Kronos 上游仓库 + TDX 适配脚本**协同使用的操作手册。冷启动时先确认宿主项目就绪：

```bash
# 1. 克隆 Kronos 上游
git clone https://github.com/shiyu-coder/Kronos.git
cd Kronos
```

2. 确保 `finetune/` 目录下存在以下 TDX 适配脚本（随宿主项目分发，**不在本技能包内**）：

| 文件 | 作用 |
|------|------|
| `finetune/config_tdx.py` | 单卡微调配置（后复权、TDX 时间范围、显存参数） |
| `finetune/train_tokenizer_tdx.py` | Tokenizer 单卡训练 |
| `finetune/train_predictor_tdx.py` | Predictor 单卡训练（AMP fp16 + 梯度累积） |
| `finetune/dataset.py` | 数据集加载器（支持自定义 config） |

如果宿主项目缺少上述 `*_tdx.py`，本技能描述的 Step 3 训练步骤无法执行——需要先补齐这些脚本，或回退到上游 `finetune/train_tokenizer.py` 配 Qlib 数据走原始流程。

## 前置条件

确认以下环境就绪后再开始：

| 条件 | 检查命令 | 要求 |
|------|----------|------|
| GPU | `nvidia-smi` | >= 8GB VRAM（如 RTX 4060） |
| PyTorch CUDA | `python -c "import torch; print(torch.cuda.is_available())"` | True |
| TDX 数据目录 | `ls ~/.local/share/tdxcfv/drive_c/tc/vipdoc/sh/lday/` | 存在 `.day` 文件 |
| 磁盘空间 | `df -h .` | >= 2GB（160MB 数据 + 425MB 模型 + 410MB 输出） |
| HF 镜像 | `curl -s --connect-timeout 5 https://hf-mirror.com` | 可访问 |
| tdxdata 包 | `python -c "import tdxdata"` | 已安装 |
| pytest | `python -m pytest --version` | 已安装 |

TDX 数据目录默认路径：`~/.local/share/tdxcfv/drive_c/tc/`（wine 安装）。若路径不同，通过 `--tdxdir` 参数指定。

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
pip install -e tdxdata/
```

**关键说明**：
- `HF_ENDPOINT` 必须写入 `activate` 脚本末尾，确保每次激活自动生效。所有 HuggingFace 模型下载均走国内镜像。
- `tdxdata` 必须以 editable 模式安装，否则项目中 `from tdxdata import ...` 会失败。
- `pytest` 不在 `requirements.txt` 中，需单独安装。

## 核心决策

这些决策已在流程中固定，不需要每次重新讨论：

- **复权方式**: 后复权 (hfq/back) — 匹配原始 Kronos Qlib 训练数据约定
- **模型**: Kronos-Tokenizer-base + Kronos-base（102M 参数）
- **数据周期**: 日线（1d），其他周期按需重采样
- **单卡配置**: Tokenizer bs=50 fp32, Predictor bs=12 AMP fp16 + accumulation×4

## 执行步骤

按顺序执行以下 4 步。每步完成后验证输出再进入下一步。

### Step 1: 数据导入

使用 `scripts/tdx_import.py` 从 TDX 本地文件导入全量 A 股日线数据。

#### 切分原则（相对当前 TDX 数据末日，不要硬抄日期）

TDX 本地数据每天都在长，**不要把下面示例日期当永恒事实**。按以下原则在执行前**重新计算**：

- 设 `END = TDX 数据末日`（一般是今天或昨天交易日）
- **test**: `[END - 3 月, END]`
- **val**: `[END - 6 月, END - 3 月)` ← 与 test 不重叠
- **train**: `[数据起始（约 2024-06）, END - 6 月)` ← 与 val 不重叠

三段**严禁重叠**——重叠会让 val/test 指标偏低、丧失泛化判断意义；`lookback_window=90` 所需的历史会自动从更早的数据里取，不要靠"区间重叠"凑。

下面命令以 **2026-05-07 为当前日**给出示例日期，照抄前请按上面原则改：

```bash
# 生成股票列表（默认包含深沪主板，排除北交所 bj*；
# 如需排除 00 开头代码可加 --exclude-00）
.venv/bin/python scripts/discover_stocks.py --output /tmp/tdx_all_stocks.txt

# 导入数据（后复权，带因子缓存）
.venv/bin/python scripts/tdx_import.py \
  --symbol-file /tmp/tdx_all_stocks.txt \
  --dividend-type back \
  --periods 1d \
  --output-dir ./data/tdx_import \
  --train-range 2024-06-01 2025-10-31 \
  --val-range   2025-11-01 2026-01-31 \
  --test-range  2026-02-01 2026-04-30 \
  --no-continuity
```

**首次运行**: 需从新浪获取 ~5000 只股票的复权因子，约 25-30 分钟。因子缓存在 TDX 目录旁的 `.factor_cache/` 中，后续导入秒级完成。

**验证**: 检查输出文件存在且非空：
```bash
ls -lh data/tdx_import/1d/train_data.pkl  # ~100MB
ls -lh data/tdx_import/1d/val_data.pkl    # ~35MB
ls -lh data/tdx_import/1d/test_data.pkl   # ~25MB
```

**故障处理**:
- 部分股票复权因子获取失败 → 自动降级为不复权，不影响流程
- 股票数量显著少于 4900 → 检查 TDX 数据目录是否包含沪深两市数据
- 磁盘不足 → 可通过 `--limit N` 先导入少量股票测试

### Step 2: 模型下载

从 HuggingFace 镜像下载预训练权重（`HF_ENDPOINT` 已由 venv activate 自动设置）：

```bash
.venv/bin/python -c "
from huggingface_hub import hf_hub_download
for model_id in ['NeoQuasar/Kronos-Tokenizer-base', 'NeoQuasar/Kronos-base']:
    path = hf_hub_download(repo_id=model_id, filename='model.safetensors')
    print(f'{model_id}: {path}')
    # Also download config.json
    hf_hub_download(repo_id=model_id, filename='config.json')
"
```

模型大小：Tokenizer 15.8MB + Kronos-base 409MB，下载耗时约 5-10 秒（国内镜像）。

**故障处理**:
- `from_pretrained` 报 `missing N required positional arguments` → 本质是 `HF_ENDPOINT` 未生效，`config.json` 下载失败。检查 `echo $HF_ENDPOINT` 是否输出 `https://hf-mirror.com`，详见 [环境重建](#环境重建--项目迁移)
- `hf-mirror.com` 不可用 → 尝试 `https://huggingface.co`（直连可能较慢）
- 无网络 → 从其他有网络环境拷贝 `~/.cache/huggingface/hub/` 目录

### Step 3: 微调训练

分两个阶段：先微调 Tokenizer，再微调 Predictor。

**3a. Tokenizer 微调** (30 epochs, ~0.9 小时, ~5GB VRAM)

```bash
.venv/bin/python finetune/train_tokenizer_tdx.py \
  --data-dir ./data/tdx_import/1d \
  --epochs 30
```

**3b. Predictor 微调** (30 epochs, ~5.4 小时, ~6.3GB VRAM)

```bash
.venv/bin/python finetune/train_predictor_tdx.py \
  --data-dir ./data/tdx_import/1d \
  --tokenizer-path ./outputs/tdx_finetune/tdx_tokenizer/checkpoints/best_model \
  --epochs 30
```

**验证**: 检查模型输出文件：
```bash
ls -lh outputs/tdx_finetune/tdx_tokenizer/checkpoints/best_model/model.safetensors   # ~16MB
ls -lh outputs/tdx_finetune/tdx_predictor/checkpoints/best_model/model.safetensors  # ~391MB
```

**训练中断恢复**: Tokenizer 和 Predictor 的 checkpoint 在每个最佳 val_loss epoch 后保存。如果训练中断，Predictor 可以直接从已保存的 Tokenizer checkpoint 继续：

```bash
# 从中断的 Predictor 开始（需先完成 Tokenizer）
.venv/bin/python finetune/train_predictor_tdx.py \
  --data-dir ./data/tdx_import/1d \
  --tokenizer-path ./outputs/tdx_finetune/tdx_tokenizer/checkpoints/best_model \
  --epochs 30
```

**显存不足**: 若 OOM：
- Tokenizer: 减小 `batch_size`（修改 `config_tdx.py`）
- Predictor: 减小 `predictor_batch_size` 或增加 `predictor_accumulation`

### Step 4: 预测验证

使用 `scripts/predict_sse.py` 验证模型预测能力：

```bash
# 导入上证指数（不复权 — 指数不需要复权）
.venv/bin/python scripts/tdx_import.py \
  --symbols sh000001 \
  --dividend-type none \
  --periods 1d \
  --output-dir ./data/tdx_import_sse \
  --no-split --no-continuity

# 运行预测
.venv/bin/python scripts/predict_sse.py
```

预测脚本输出未来 20 个交易日的 OHLCV 点估计及涨跌幅统计。

**注意**:
- 指数预测仅作模型能力验证，不构成投资建议
- 交易日历未考虑 A 股节假日，生成的是自然周历
- 微调模型在个股数据上训练，对指数的适用性有限

## 关键文件清单

- `scripts/tdx_import.py` — TDX 数据导入工具（复权、连续性检测、因子缓存）
- `scripts/discover_stocks.py` — 从 TDX 日线目录枚举并过滤股票代码，输出 symbol-file
- `finetune/config_tdx.py` — 单卡微调配置（后复权、TDX 时间范围、显存参数）
- `finetune/train_tokenizer_tdx.py` — Tokenizer 单卡训练脚本
- `finetune/train_predictor_tdx.py` — Predictor 单卡训练脚本（AMP fp16 + 梯度累积）
- `finetune/dataset.py` — 数据集加载器（修改后支持自定义 config）
- `scripts/predict_sse.py` — 上证指数预测演示
- `.venv/bin/activate` — venv 激活脚本（末尾含 `HF_ENDPOINT` 国内镜像配置）
- `tdxdata/pyproject.toml` — tdxdata 包配置（需 `pip install -e` 安装）
- `requirements.txt` — Python 核心依赖（不含 pytest、tdxdata）

### `finetune/config_tdx.py` 关键字段速查

复制改造时重点关注以下字段（完整字段以宿主项目源文件为准）：

| 字段 | 默认 | 说明 |
|------|------|------|
| `lookback_window` | 90 | 模型可见的历史交易日数 |
| `predict_window` | 10 | 训练时的预测窗口 |
| `dividend_type` | "back" | 后复权（匹配上游 Qlib 训练约定） |
| `batch_size` | 50 | Tokenizer 批大小（fp32） |
| `predictor_batch_size` | 12 | Predictor 批大小（AMP fp16） |
| `predictor_accumulation` | 4 | 梯度累积步数（有效 bs = 12×4 = 48） |
| `tokenizer_learning_rate` | 2e-4 | Tokenizer 学习率 |
| `predictor_learning_rate` | 4e-5 | Predictor 学习率 |
| `epochs` | 30 | Tokenizer / Predictor 训练轮数 |
| `seed` | 100 | 复现种子 |
| `train_time_range` / `val_time_range` / `test_time_range` | 见 Step 1 | 三段切分日期 |

## 环境重建 / 项目迁移

当项目目录发生迁移（如 `~/financial/Kronos` → `~/peiking88/Kronos`）或 venv 损坏时的完整修复流程。

### venv 损坏症状

- `source .venv/bin/activate` 设置错误的 `VIRTUAL_ENV` 路径
- 所有 bin 脚本（pip, python, activate 等共 20+ 个）shebang 指向旧路径
- `pip install` 报 `No such file or directory` 或 `bad interpreter`

### 一键重建

```bash
# 1. 删除旧 venv
rm -rf .venv

# 2. 创建新 venv
python3 -m venv .venv

# 3. 写入 HF 国内镜像到 activate 脚本
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> .venv/bin/activate

# 4. 激活并安装
source .venv/bin/activate
pip install -r requirements.txt
pip install pytest pytest-timeout
pip install -e tdxdata/
```

### 测试验证

```bash
# 验证核心依赖
python -c "import torch; import numpy; import pandas; import tdxdata; print('OK')"

# 验证 CUDA
python -c "import torch; print(torch.cuda.is_available())"

# 运行测试套件
python -m pytest tdxdata/tests/ -v     # 184 项，全部通过
python -m pytest tests/ -v             # 4 项回归测试，全部通过
```

### 常见陷阱

| 现象 | 根因 | 修复 |
|------|------|------|
| `from_pretrained` 报 `missing N required positional arguments` | `config.json` 下载失败，`model_kwargs` 为空。本质是 HF 不可达 | 确认 `HF_ENDPOINT` 已写入 activate 脚本并 source |
| `ModuleNotFoundError: No module named 'tdxdata'` | tdxdata 未安装 | `pip install -e tdxdata/` |
| `No module named pytest` | pytest 不在 requirements.txt | `pip install pytest pytest-timeout` |
| 部分模型能加载、另一部分不行 | 能加载的模型已缓存，未缓存的模型因网络问题下载失败 | 同上 — 配置 HF 镜像后重新加载 |

## 技术细节

### 复权因子获取

复权因子从新浪财经 HTTP API 获取：
- URL: `https://finance.sina.com.cn/realstock/company/{market}{symbol}/{qfq|hfq}.js`
- 仅日线数据做复权调整（分钟线不改动）
- 仅调整 OHLC 四列，volume/amount 不变
- 因子获取失败 → 自动降级为不复权
- 因子缓存: `{tdxdir}/../.factor_cache/{code}.pkl`

### 显存配置依据

在 RTX 4060 Laptop 8GB 上实测：

| 配置 | Tokenizer | Predictor |
|------|-----------|-----------|
| Batch size | 50 (fp32) | 12 (AMP fp16) |
| 显存占用 | ~5.0 GB | ~6.3 GB |
| 最大 bs | 64 | 16 (AMP) |
| OOM | bs=100 | bs=20 (AMP) |

若使用其他 GPU，需要重新测显存并调整参数。

### 数据格式

Kronos 6 字段格式: `open, high, low, close, vol, amt`

- 输出为 pickle 文件: `{symbol: DataFrame(index=DatetimeIndex, columns=6)}`
- Amount 计算: 优先 TDX 原始数据，缺失时 `mean(OHLC) × vol`
- 索引列名兼容 `date` 和 `datetime`

### 数据时间范围

TDX 本地数据起始约为 2024-06，每日增长。切分务必**按当前数据末日动态计算**（参见 Step 1 「切分原则」），下方示例对应 2026-05 数据快照：

- 训练集: 约 17 个月（数据起始 ~ END-6M）
- 验证集: 约 3 个月（END-6M ~ END-3M）
- 测试集: 约 3 个月（END-3M ~ END）

三段不重叠；lookback_window=90 所需的历史会自动从早于训练起点的数据里取。早期版本的 SKILL.md 让 train 与 val 区间重叠以"适配 lookback"——这是一个错误，已修正：lookback 通过自然历史前向取数即可，无需区间交叉。

## 实践经验

微调实战中积累的常见陷阱和最佳实践，详见 `references/lessons-learned.md`：

| 条目 | 说明 |
|---|---|
| 后复权因子外推 | hfq 必须 `direction="backward"`，排查跳变方法 |
| val/test lookback 补齐 | 从 train 末尾接 120 天，否则 val 集为空 |
| 早停 | patience=5，节省 15-25% 训练时间 |
| 报告价格规范 | 只显示实际市场价，不显示后复权价 |
| 极端波动过滤 | 90 日回撤 >30% 或日波动 >8% 自动跳过 |
| 模型偏置认知 | 均值回复 + 空头偏置，方向准确率 ~50% |
| 依赖版本锁定 | mootdx≥2.0.3, opentdx≥0.5.10, tdxdata≥0.8.4 |
| 一键预测 | `predict_stocks.py` 多股票预测 + md 报告 |
