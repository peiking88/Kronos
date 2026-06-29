# 预测所需本地数据文件

微调后模型预测涉及两个入口脚本，数据来源和模型加载方式不同：

| 预测入口                    | 模型来源               | 行情数据来源         | 复权因子来源                |
| --------------------------- | ---------------------- | -------------------- | --------------------------- |
| `scripts/predict.py`        | HuggingFace 预训练模型 | TDengine 实时查询    | adjust 表事件本地计算       |
| `scripts/predict_stocks.py` | 本地微调模型           | 本地 pkl + TDengine 增量更新 | TDengine 原始收盘价推导 |

---

## 一、预置文件（必须提前存在）

### 1.1 模型权重

| 文件路径                                                                      | 格式        | 用途                                  | 读取位置                |
| ----------------------------------------------------------------------------- | ----------- | ------------------------------------- | ----------------------- |
| `outputs/tdx_finetune/tdx_tokenizer/checkpoints/best_model/model.safetensors` | safetensors | 微调后的 tokenizer 权重（行情编码器） | `predict_stocks.py:128` |
| `outputs/tdx_finetune/tdx_tokenizer/checkpoints/best_model/config.json`       | json        | tokenizer 模型结构配置                | 同上                    |
| `outputs/tdx_finetune/tdx_predictor/checkpoints/best_model/model.safetensors` | safetensors | 微调后的 predictor 权重（自回归预测） | `predict_stocks.py:131` |
| `outputs/tdx_finetune/tdx_predictor/checkpoints/best_model/config.json`       | json        | predictor 模型结构配置                | 同上                    |

> 由 `finetune/train_tokenizer_tdx.py` 和 `finetune/train_predictor_tdx.py` 训练生成。

### 1.2 行情数据

| 文件路径                            | 格式 | 用途                                | 读取位置                    |
| ----------------------------------- | ---- | ----------------------------------- | --------------------------- |
| `data/tdx_import/1d/train_data.pkl` | pkl  | 训练集后复权日线（2011~2025.10）    | `predict_stocks.py:116-123` |
| `data/tdx_import/1d/val_data.pkl`   | pkl  | 验证集后复权日线（2025.11~2026.02） | 同上                        |
| `data/tdx_import/1d/test_data.pkl`  | pkl  | 测试集后复权日线（2026.02~2026.05） | 同上                        |
| `data/tdx_import/1d/data.pkl`       | pkl  | 全量未分割数据（兜底）              | 同上                        |
| `data/tdx_import_sse/1d/data.pkl`   | pkl  | 上证指数后复权日线                  | `predict_stocks.py:109-113` |

> pkl 格式为 `dict[str, pd.DataFrame]`，key 为股票代码（如 `sh600000`），value 为 DataFrame（datetime index，列: open/high/low/close/vol/amt）。
> 由 `scripts/tdx_import.py --dividend-type back` 生成。

### 1.3 TDengine 数据源（predict.py / tdx_import.py 共用）

| 数据源                                      | 格式         | 用途         | 读取位置                  |
| ------------------------------------------- | ------------ | ------------ | ------------------------- |
| TDengine `tdx.k_{code}_1d`                  | TDengine 表  | 个股日线 OHLCV | `tdx_import.py:_query_daily_kline` |
| TDengine `tdx.a_{code}`                     | TDengine 表  | 分红/送转股/配股事件 | `tdx_import.py:_query_adjust_events` |

> 数据源为 TDengine，通过 `taosws` WebSocket 连接。后复权由 adjust 表事件实时计算，无需外部 API。

---

## 二、运行时生成文件

### 2.1 复权因子缓存

| 文件路径                                                    | 格式           | 用途                           | 生成位置                                       |
| ----------------------------------------------------------- | -------------- | ------------------------------ | ---------------------------------------------- |
| `outputs/.factor_1d.json`                                   | json           | 复权因子缓存（月度有效）        | `predict_stocks.py:prefetch_factors`            |

> 因子由 TDengine 原始收盘价与后复权收盘价推导（hfq_close / raw_close），缓存到 outputs/.factor_1d.json 按月有效。

### 2.2 预测输出

| 文件路径                                        | 格式        | 用途                            | 生成位置                |
| ----------------------------------------------- | ----------- | ------------------------------- | ----------------------- |
| `outputs/pred_{code}_{date}.csv`                | CSV         | predict.py 预测结果（实际价格） | `predict.py:300`        |
| `outputs/pred_{code}_{date}_chart.html`         | Plotly HTML | 可视化交互图表                  | `predict.py:308`        |
| `outputs/kronos_{codes}.md`                     | Markdown    | predict_stocks.py 预测+回测报告 | `predict_stocks.py:521` |
| `webui/prediction_results/prediction_{ts}.json` | JSON        | WebUI 预测结果                  | `webui/app.py:153`      |

---

## 三、关键路径常量

| 常量             | 值                                               | 定义位置                                 |
| ---------------- | ------------------------------------------------ | ---------------------------------------- |
| `MODEL_DIR`      | `outputs/tdx_finetune`                           | `predict_stocks.py:33`                   |
| `DATA_DIR`       | `data/tdx_import/1d`                             | `predict_stocks.py:31`                   |
| `SSE_DATA`       | `data/tdx_import_sse/1d/data.pkl`                | `predict_stocks.py:32`                   |
| `FACTOR_CACHE_1D` | `outputs/.factor_1d.json`               | `predict_stocks.py:118`                   |
| `LOOKBACK`        | 400 (predict.py) / 90 (predict_stocks.py) | `predict.py:41` / `predict_stocks.py:130` |
| `MAX_STALE_DAYS` | 5                                                | `predict_stocks.py:43`                   |

---

## 四、数据流

### 4.1 predict.py（预训练模型 + TDengine 实时导入）

```
TDengine tdx.k_{code}_1d
    ↓ taosws.connect().query()
原始 OHLCV DataFrame
    ↓ 加载复权因子（TDengine adjust 表 → 本地计算）
后复权 DataFrame + factor
    ↓ HuggingFace 下载 KronosTokenizer + Kronos 模型
    ↓ 标准化 → tokenizer 编码 → 自回归推理 → 解码 → 反标准化
后复权预测价格
    ↓ / factor → 实际价格
    ↓ 涨跌停校准 + 回测校准
→ outputs/pred_{code}_{date}.csv + chart.html
```

### 4.2 predict_stocks.py（微调模型 + 本地 pkl）

```
data/tdx_import_sse/1d/data.pkl（指数优先）
    或
data/tdx_import/1d/{test,val,train,data}.pkl（个股）
    ↓ pickle.load() → 查找 code 对应 DataFrame
后复权 DataFrame
    ↓ 若数据过期（>5天），自动从 TDX 增量导入并合并
    ↓ 推导复权因子（缓存/推导/在线获取）
factor 值
    ↓ 加载本地微调模型（outputs/tdx_finetune/.../best_model）
    ↓ 标准化 → 编码 → 自回归推理 → 解码 → 反标准化
后复权预测价格
    ↓ / factor → 实际价格
    ↓ 涨跌停校准 + 回测校准
→ outputs/kronos_{codes}.md
```

---

## 五、脚本间依赖关系

```
tdx_import.py ──生成──→ data/tdx_import/1d/*.pkl
                       data/tdx_import_sse/1d/data.pkl
                       .factor_cache/{code}.pkl

predict.py ──读取──→ TDengine tdx.k_{code}_1d
           ──下载──→ HuggingFace 预训练模型
           ──调用──→ calibrate.py（纯计算）

predict_stocks.py ──读取──→ data/tdx_import/*.pkl
                  ──读取──→ outputs/tdx_finetune/.../best_model/*
                  ──调用──→ calibrate.py（纯计算）
                  ──调用──→ tdx_import.py（数据过期时增量导入）

webui/app.py ──读取──→ data/**/*.pkl
             ──读取──→ outputs/tdx_finetune/.../best_model/*（可选）
             ──下载──→ HuggingFace 预训练模型（可选）
             ──调用──→ calibrate.py（纯计算）
```

---

## 六、数据准备步骤

使用微调模型预测前，需按以下顺序准备数据：

1. **确保 TDengine 服务运行**（数据源）
2. **运行 `scripts/tdx_import.py`** 生成 pkl 行情数据
3. **完成模型微调**（`finetune/train_tokenizer_tdx.py` + `train_predictor_tdx.py`），生成 `outputs/tdx_finetune/` 下的模型权重
4. **运行 `scripts/predict_stocks.py`** 进行预测
