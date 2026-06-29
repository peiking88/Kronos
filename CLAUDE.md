# 项目配置

- 报告输出目录为 `outputs/`

## 数据源

- **TDengine**：`tdx.kline` (OHLCV) + `tdx.adjust` (分红送转股/配股事件)
- 数据导出：`scripts/tdx_export_from_tdengine.py` → `data/tdx_import/1d/*.pkl`
- 后复权：导出时从 `adjust` 表原始事件实时计算，无需外部 API
- 格式：`{symbol: DataFrame(open/high/low/close/vol/amt float32, index=DatetimeIndex)}`

## 模型

- 上游纯净 Kronos：`model/kronos.py` 来自 `https://github.com/shiyu-coder/Kronos`
- 预训练权重：`NeoQuasar/Kronos-Tokenizer-base` + `NeoQuasar/Kronos-base`
- 下载镜像：`hf-mirror.com`（`HF_ENDPOINT` 写入 `.venv/bin/activate`）

## 微调

- 单阶段全参数微调
- Tokenizer: 30 epochs, bf16
- Predictor: 10 epochs, bf16, bs=128 可 OOM → 建议 bs=64
- `finetune/train_predictor_tdx.py` 已精简为仅 full 模式

## TDengine 连接

- 方式：`taos-ws-py` (WebSocket)，非 `taospy`（原生库有 `execstack` 兼容性问题）
- 连接：`from taosws import connect; c = connect()`
- 查询需带库名前缀：`tdx.k_000001_1d` / `tdx.a_000001`

## 已知陷阱

- HF 缓存 symlink 可能悬空 — 删除 `models--NeoQuasar--*` 后重下
- GPU 僵尸进程残留 — 训练前用 `nvidia-smi` 确认
