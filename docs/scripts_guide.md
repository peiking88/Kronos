# Kronos 脚本使用说明书

本文档涵盖两个核心脚本的完整用法：一键启动脚本 `start.sh` 和一键预测脚本 `scripts/predict.py`。

---

## 目录

- [一、一键启动脚本 start.sh](#一一键启动脚本-startsh)
  - [基本用法](#基本用法)
  - [命令参考](#命令参考)
  - [选项参考](#选项参考)
  - [使用示例](#使用示例)
  - [启动检查流程](#启动检查流程)
  - [常见问题](#常见问题)
- [二、一键预测脚本 scripts/predict.py](#二一键预测脚本-scriptspredictpy)
  - [基本用法](#基本用法-1)
  - [参数参考](#参数参考)
  - [使用示例](#使用示例-1)
  - [输出文件说明](#输出文件说明)
  - [预测流程](#预测流程)
  - [常见问题](#常见问题-1)
- [三、配合使用](#三配合使用)

---

## 一、一键启动脚本 start.sh

Kronos WebUI 的启动/停止/管理脚本。WebUI 是前后端一体的 Flask 应用（前端 HTML/Plotly + 后端 API），端口 7070。

### 基本用法

```bash
cd /home/li/peiking88/Kronos

# 前台启动（推荐调试时使用，Ctrl+C 停止）
./start.sh

# 后台启动（推荐日常使用）
./start.sh -d
```

启动成功后访问 http://localhost:7070

### 命令参考

| 命令                 | 说明                                   |
| -------------------- | -------------------------------------- |
| `./start.sh`         | 前台启动，日志直接输出到终端           |
| `./start.sh -d`      | 后台启动（守护进程模式），日志写入文件 |
| `./start.sh stop`    | 停止后台服务                           |
| `./start.sh status`  | 查看运行状态、PID、最近日志            |
| `./start.sh restart` | 重启服务（先 stop 再启动）             |

### 选项参考

| 选项      | 默认值 | 说明         |
| --------- | ------ | ------------ |
| `-d`      | 无     | 后台启动模式 |
| `-p PORT` | 7070   | 指定端口号   |
| `-h`      | 无     | 显示帮助信息 |

### 使用示例

```bash
# 前台启动，默认端口 7070
./start.sh

# 后台启动，默认端口
./start.sh -d

# 后台启动，指定端口 8080
./start.sh -d -p 8080

# 查看当前状态
./start.sh status
# 输出:
#   [INFO] Kronos WebUI 运行中 (PID=12345)
#   [INFO] 访问: http://localhost:7070

# 停止后台服务
./start.sh stop

# 重启服务
./start.sh restart
```

### 启动检查流程

脚本启动时自动执行以下检查：

```
1. 虚拟环境检测
   └── 检查 .venv/bin/activate 是否存在
   └── 自动修复缺失的 python 符号链接（迁移后常见问题）

2. 依赖完整性检查
   └── 逐个验证 flask、pandas、numpy、plotly、torch 等
   └── 缺失的包自动从阿里云镜像安装

3. 模型库可用性
   └── 验证 from model import Kronos 能否成功

4. 端口冲突检测
   └── 检查 7070 端口是否被占用
   └── 被占用时提示具体进程，不自动 kill
```

### 常见问题

**Q: 启动报 "虚拟环境不存在"**

项目根目录下需要 `.venv` 目录。创建方式：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt -r webui/requirements.txt
```

**Q: 启动报 "端口 7070 已被占用"**

用 `-p` 指定其他端口，或先停止占用进程：

```bash
# 查看占用进程
ss -tlnp | grep 7070

# 方式一：用其他端口
./start.sh -d -p 8080

# 方式二：停止旧进程
./start.sh stop
./start.sh -d
```

**Q: 后台启动后怎么看日志？**

日志文件位于 `logs/webui.log`：

```bash
tail -f logs/webui.log    # 实时查看
tail -20 logs/webui.log   # 最近 20 行
```

**Q: 如何确认服务是否在运行？**

```bash
./start.sh status
# 或者直接 curl
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:7070/
# 返回 200 表示正常
```

---

## 二、一键预测脚本 scripts/predict.py

从 TDX 本地数据读取最新行情，使用 Kronos 模型预测未来 N 日走势，输出 CSV 数据文件和交互式 HTML 图表。

### 基本用法

```bash
cd /home/li/peiking88/Kronos
source .venv/bin/activate

# 预测单只股票
python scripts/predict.py 600000

# 预测多只股票
python scripts/predict.py 600000 002741 600519
```

### 参数参考

| 参数                   | 默认值                              | 说明                                                                                |
| ---------------------- | ----------------------------------- | ----------------------------------------------------------------------------------- |
| `symbols`              | (必填)                              | 股票代码，支持多个。格式：`600000`、`002741`、`sh000001`、`sz002741`                |
| `--pred-len`           | 30                                  | 预测交易日数                                                                        |
| `--model`              | small                               | 模型大小：`mini`（4.1M 参数，最快）/ `small`（24.7M，推荐）/ `base`（102.3M，最准） |
| `--device`             | cpu                                 | 计算设备：`cpu` / `cuda:0` / `mps`                                                  |
| `--temperature` / `-T` | 1.2                                 | 采样温度，越高越随机                                                                |
| `--top-p`              | 0.95                                | 核采样概率阈值                                                                      |
| `--samples`            | 2                                   | 采样次数，多次采样取平均可提升稳定性                                                |
| `--lookback`           | 400                                 | 回看历史天数                                                                        |
| `--no-limit`           | 关闭                                | 默认开启 ±10% 涨跌停约束，加此选项关闭                                              |
| `--output-dir`         | outputs                             | 输出目录                                                                            |
| `--tdxdir`             | `~/.local/share/tdxcfv/drive_c/tc/` | TDX 数据目录                                                                        |

### 使用示例

```bash
# 基础用法：预测浦发银行未来 30 日
python scripts/predict.py 600000

# 批量预测多只股票
python scripts/predict.py 600000 002741 000001 600519

# 使用 base 模型预测 60 日（更准但更慢）
python scripts/predict.py --model base --pred-len 60 600000

# 快速预测（mini 模型）
python scripts/predict.py --model mini 002741

# GPU 加速
python scripts/predict.py --device cuda:0 600000

# 高随机性预测（温度 1.5，多次采样）
python scripts/predict.py -T 1.5 --samples 5 600000

# 指数预测（6 开头自动识别为沪市）
python scripts/predict.py sh000001

# 不限制涨跌停（模型原始输出）
python scripts/predict.py --no-limit 600000
```

### 输出文件说明

每只股票生成两个文件，保存在 `outputs/` 目录下：

| 文件                            | 说明                                                        |
| ------------------------------- | ----------------------------------------------------------- |
| `pred_{代码}_{日期}.csv`        | 预测数据，包含 date、open、high、low、close、volume、amount |
| `pred_{代码}_{日期}_chart.html` | 交互式图表，用浏览器打开查看                                |

CSV 文件示例：

```
date,open,high,low,close,volume,amount
2026-05-07,9.02,9.15,8.95,9.08,12345678.00,123456789.00
2026-05-08,9.08,9.20,8.90,9.12,11234567.00,112345678.00
...
```

HTML 图表包含：

- 最近 60 日历史收盘价（蓝色实线）
- 预测收盘价（红色虚线）
- 鼠标悬停显示具体数据

### 预测流程

```
1. 数据读取
   └── 从 TDX 本地数据读取指定股票日线
   └── 自动识别最新交易日（以 sh000001 为基准）

2. 数据准备
   └── 取最近 400 根 K 线作为上下文
   └── 数据不足 400 根时跳过并报告

3. 模型推理
   └── 加载 Kronos 模型（首次运行需下载，后续使用缓存）
   └── 自回归逐日预测

4. 后处理
   └── 默认应用 ±10% 涨跌停约束
   └── 输出 CSV + HTML 图表

5. 摘要输出
   └── 首日/末日预测价及涨跌幅
   └── 预测区间（最高/最低）
```

### 股票代码格式

脚本自动识别市场，支持以下格式：

| 输入格式   | 识别结果 | 说明                   |
| ---------- | -------- | ---------------------- |
| `600000`   | sh600000 | 6 开头自动识别为沪市   |
| `000001`   | sz000001 | 0/3 开头自动识别为深市 |
| `sh600000` | sh600000 | 显式指定沪市           |
| `sz002741` | sz002741 | 显式指定深市           |
| `SH600000` | sh600000 | 大小写不敏感           |

### 常见问题

**Q: 提示 "TDX 中未找到 xxx 的数据"**

TDX 客户端未下载该股票数据。解决方式：

1. 打开 TDX 客户端，访问该股票页面触发数据下载
2. 确认 TDX 数据目录正确（默认 `~/.local/share/tdxcfv/drive_c/tc/`）

**Q: 提示 "数据不足: N 根 < 回看 400"**

该股票在 TDX 中的历史数据不足 400 个交易日。可降低回看天数：

```bash
python scripts/predict.py --lookback 200 600000
```

**Q: 预测结果看起来不合理**

- 尝试调整采样参数：`-T 1.0 --samples 3`（降低随机性，增加采样次数）
- 使用更大的模型：`--model base`
- 模型预测仅供参考，不构成投资建议

**Q: GPU 模式报错**

确认 PyTorch 支持 CUDA：

```bash
python -c "import torch; print(torch.cuda.is_available())"
# 应输出 True
```

**Q: 首次运行很慢**

首次运行需要从 HuggingFace 下载模型（约 100MB），后续使用缓存。如果网络不佳，可设置镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
python scripts/predict.py 600000
```

---

## 三、配合使用

### 典型工作流

```bash
# 1. 启动 WebUI
./start.sh -d

# 2. 命令行预测
python scripts/predict.py 600000 002741

# 3. 在 WebUI 中查看结果
# 浏览器打开 http://localhost:7070
# 上传 outputs/pred_sh600000_20260506.csv 查看详细图表

# 4. 停止服务
./start.sh stop
```

### 输出文件与 WebUI 联动

`predict.py` 生成的 CSV 文件可直接在 WebUI 中加载：

1. 将 CSV 文件复制到 `webui/data/` 目录
2. 在 WebUI 界面选择该文件
3. WebUI 会自动解析并展示 K 线图

### 目录结构

```
Kronos/
├── start.sh              # 一键启动脚本
├── scripts/
│   ├── predict.py         # 一键预测脚本
│   └── tdx_import.py      # TDX 数据导入工具
├── webui/
│   ├── app.py             # Flask 应用
│   ├── run.py             # Python 启动入口
│   └── data/              # WebUI 数据目录
├── outputs/               # 预测输出目录
│   ├── pred_sh600000_20260506.csv
│   └── pred_sh600000_20260506_chart.html
└── logs/                  # 运行日志
    ├── webui.log
    └── webui.pid
```
