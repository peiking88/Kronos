---
name: check-kronos-env
description: "Kronos 项目环境完整性检查与测试运行。检查 Python venv 完整性（含符号链接修复）、验证依赖包、运行回归测试、启动 WebUI 并验证 HTTP 可访问性。当用户提到 Kronos、kronos 环境检查、环境验证、运行测试、启动前后端、venv 修复、依赖检查、项目初始化、健康检查、检查环境、跑测试、启动项目时使用此技能。"
---

# Kronos 环境完整性检查技能

本技能对 Kronos 项目（A股金融时序预测模型）进行完整的环境健康检查，覆盖从 Python 虚拟环境到模型推理验证的全链路。

## 项目上下文

- **项目根目录**: `/home/li/peiking88/Kronos`
- **虚拟环境**: `.venv`（Python 3.14）
- **模型包**: 项目根目录下的 `model/`（含 `__init__.py`）
- **测试**: `tests/test_kronos_regression.py`（pytest）
- **WebUI**: `webui/app.py`（Flask，端口 7070）
- **数据源**: TDX 本地数据 (`~/.local/share/tdxcfv/drive_c/tc/`)

## 执行流程

按照以下 5 个阶段依次执行。每个阶段结束后报告状态，遇到失败先尝试自动修复，修复失败则报告错误并停止。

### 阶段 1：Python 虚拟环境检查

这是最容易出问题的环节——项目迁移后 venv 中的 python 符号链接经常丢失。

1. 检查 `.venv/bin/python`、`.venv/bin/python3`、`.venv/bin/python3.14` 是否存在
2. 如果缺失，查找系统 python 路径（`/usr/bin/python3` 或 `/usr/bin/python3.14`），重建符号链接：
   ```bash
   ln -sf /usr/bin/python3.14 .venv/bin/python3.14
   ln -sf python3.14 .venv/bin/python3
   ln -sf python3 .venv/bin/python
   ```
3. 激活 venv 并确认 `python --version` 正常输出
4. 如果 venv 本身不存在，提示用户创建（不自动创建，避免覆盖）

**失败标志**: 激活 venv 后 `python --version` 无输出或报错。

### 阶段 2：依赖完整性验证

关键依赖列表（任一缺失会导致后续阶段失败）：

| 包名            | 用途         | 检查命令                 | 来源                                          |
| --------------- | ------------ | ------------------------ | --------------------------------------------- |
| torch           | 模型推理     | `import torch`           | requirements.txt                              |
| pandas          | 数据处理     | `import pandas`          | requirements.txt                              |
| numpy           | 数值计算     | `import numpy`           | requirements.txt                              |
| einops          | 模型张量操作 | `import einops`          | requirements.txt                              |
| safetensors     | 模型加载     | `import safetensors`     | requirements.txt                              |
| tqdm            | 进度条       | `import tqdm`            | requirements.txt                              |
| huggingface_hub | 模型下载     | `import huggingface_hub` | requirements.txt                              |
| flask           | WebUI 后端   | `import flask`           | webui/requirements.txt                        |
| flask_cors      | 跨域支持     | `import flask_cors`      | webui/requirements.txt                        |
| plotly          | 图表渲染     | `import plotly`          | webui/requirements.txt                        |
| taos-ws-py      | TDengine 连接 | `import taosws`          | requirements.txt                              |

**开发依赖**（仅运行测试需要，不在 requirements.txt 中）：

| 包名           | 用途                             | 检查命令                |
| -------------- | -------------------------------- | ----------------------- |
| pytest         | 测试运行                         | `import pytest`         |
| pytest-timeout | 测试超时控制（`--timeout` 参数） | `import pytest_timeout` |

检查方式：逐个 `python -c "import <pkg>"`，收集缺失列表。

如果存在缺失包，按来源分组安装：

```bash
# 核心依赖
pip install -i https://mirrors.aliyun.com/pypi/simple/ -r requirements.txt
# WebUI 依赖
pip install -i https://mirrors.aliyun.com/pypi/simple/ -r webui/requirements.txt
# 开发依赖
pip install -i https://mirrors.aliyun.com/pypi/simple/ pytest
```

**关于 matplotlib**：`requirements.txt` 中固定了 `matplotlib==3.9.3`，但该版本与 Python 3.14 不兼容（legend RecursionError）。如果安装时报错或 import 失败，可忽略——项目已全面使用 plotly 替代 matplotlib。后续建议从 requirements.txt 中移除 matplotlib。

**GPU/CUDA 可用性**：PyTorch 推理性能依赖 GPU，检查 CUDA 是否可用：

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"
```

如果 CUDA 不可用，报告"将使用 CPU 模式运行，推理速度较慢"。不影响后续阶段，但应在报告中标注。

### 阶段 3：回归测试

运行项目自带的回归测试，验证模型推理精度。

```bash
PYTHONPATH=. pytest tests/test_kronos_regression.py -v --timeout=550
```

**关键点**：

- 必须设置 `PYTHONPATH=.`，因为测试中 `from model import ...` 依赖项目根目录在搜索路径中
- 项目没有 `conftest.py` 或 `pytest.ini`，所以 `PYTHONPATH` 是必须的
- 测试会从 HuggingFace 下载模型（首次约 100MB），后续使用缓存
- 4 个测试用例应在 15 秒内全部通过（CPU 模式）

**预期结果**: 4 passed（2 个 regression + 2 个 MSE）

### 阶段 4：WebUI 启动与验证

1. 检查端口 7070 是否空闲：`ss -tlnp | grep 7070`
2. 如果被占用，报告占用进程，不自动 kill
3. 优先使用项目自带的启动脚本（支持 `-d` 后台、`-p 端口`、`stop`、`status`）：
   ```bash
   bash start.sh -d
   ```
   如果 `start.sh` 不可用，退回到手动启动：
   ```bash
   cd webui && PYTHONPATH=/home/li/peiking88/Kronos nohup python app.py > /tmp/kronos_webui.log 2>&1 &
   ```
   用户可通过 `-p` 参数指定其他端口，如 `bash start.sh -d -p 8080`。
4. 等待最多 10 秒，用 `curl -s -o /dev/null -w "%{http_code}" --connect-timeout 3 --max-time 5 http://127.0.0.1:7070/` 验证返回 200
5. 验证模型库可用性：检查日志中 `Model availability: True`

### 阶段 5：生成报告

汇总所有阶段的检查结果，输出结构化报告：

```
# Kronos 环境检查报告

## Python 环境
- 版本: Python 3.14.x
- venv: 正常 / 已修复 / 异常
- CUDA: 可用 (GPU 名称) / 不可用 (CPU 模式)

## 依赖 (N/M 通过)
- ✅ torch 2.x.x
- ⚠️ matplotlib (不兼容 Python 3.14，可忽略)
- ❌ flask (已安装)
...

## 测试
- 4/4 通过 (12.3s)

## WebUI
- 端口 7070: 正常
- HTTP: 200 OK
- 模型库: 可用
```

## 已知问题与修复

| 问题                                           | 原因                                   | 修复方式                                                     |
| ---------------------------------------------- | -------------------------------------- | ------------------------------------------------------------ |
| venv 中无 python 可执行文件                    | 项目迁移后绝对路径符号链接失效         | 重建指向系统 python 的链接                                   |
| `ModuleNotFoundError: No module named 'model'` | pytest 未设置 PYTHONPATH               | `PYTHONPATH=. pytest`                                        |
| matplotlib legend RecursionError               | Python 3.14 与 matplotlib 3.9.x 不兼容 | 使用 plotly 代替，可从 requirements.txt 移除 matplotlib      |
| kaleido PNG 导出失败                           | kaleido 1.x 需要浏览器引擎             | 改用 `fig.write_html()`                                      |
| plotly `add_vline` TypeError                   | pandas 3.x Timestamp 加法不兼容        | 使用 `fig.add_shape()` 代替                                  |
| TDengine 连接失败                              | TDengine 服务未启动或配置错误          | 检查 taosd 服务状态与连接参数                                |
| `import taosws` 失败                           | taos-ws-py 未安装                      | `pip install taos-ws-py`                                    |

## 边界情况

- 如果用户只想检查环境不想启动 WebUI，执行阶段 1-3 + 5 即可
- 如果用户只想跑测试，执行阶段 1-2 + 3 即可
- 如果用户说"全部检查"，执行完整 5 个阶段
- 如果没有 TDX 数据，跳过数据相关检查，不影响其他阶段
