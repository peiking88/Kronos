# TDX 本地数据微调 Kronos 工作总结

## 日期
2026-05-03 ~ 2026-05-04

## 目标
利用 TDX（通达信）本地历史数据对 Kronos-base 模型进行领域自适应微调，使其适配 A 股市场特征。

---

## 1. 基线调查

### 原始模型训练数据
| 项目 | 详情 |
|------|------|
| 预训练数据 | 45+ 全球交易所 K 线数据 |
| A 股微调推荐数据 | Qlib `cn_data`（2007-2020+，后复权） |
| 特征格式 | `open, high, low, close, vol, amt`（6 字段） |
| 复权方式 | **后复权 (hfq/back)** — Qlib 默认，与原始 Kronos 训练一致 |

### TDX 本地数据覆盖
| 指标 | 日线 | 5分钟 |
|------|------|-------|
| 股票数量 | 4,939 只（沪深 A 股） | 同左 |
| 时间范围 | 2024-06-03 ~ 2026-04-30 | 2024-06-24 ~ 2026-04-30 |
| K 线/股 | ~464 根 | ~21,600 根 |
| 历史跨度 | ~2 年 | ~22 个月 |

### 硬件环境
- GPU: NVIDIA RTX 4060 Laptop 8GB
- CUDA: 13.0, PyTorch 2.11.0
- 模型拉取: hf-mirror.com 镜像

---

## 2. 复权因子

复权因子不在 TDX 本地文件中，唯一来源是新浪财经 HTTP API：

| 项目 | 说明 |
|------|------|
| API | `https://finance.sina.com.cn/realstock/company/{market}{symbol}/{qfq\|hfq}.js` |
| 缓存 | `{tdxdir}/../.factor_cache/{code}.pkl`（首次获取后持久化） |
| 适用周期 | 仅日线（分钟线不调整） |
| 调整范围 | 仅 OHLC，volume/amount 不变 |
| 降级策略 | 获取失败 → 不复权 |
| 首次耗时 | ~25 分钟（~5,000 只 × 0.3s/次 HTTP） |
| 缓存命中 | ~3,749 只；~1,190 只板块/特殊品种获取失败 |

- **前复权 (qfq)**: 最新日期因子=1，历史价格被调高
- **后复权 (hfq)**: 最早日期因子=1，当前价格被调高

---

## 3. 模型下载

| 模型 | 参数量 | 大小 | 来源 | 耗时 |
|------|--------|------|------|------|
| Kronos-Tokenizer-base | 3,958,042 (4M) | 15.8 MB | hf-mirror.com | 1.5s |
| Kronos-base | 102,310,592 (102M) | 409.3 MB | hf-mirror.com | 3.9s |

通过 `HF_ENDPOINT=https://hf-mirror.com` 使用国内镜像。

---

## 4. 显存评估

**Tokenizer 微调 (fp32):**

| Batch Size | 显存 |
|-----------|------|
| 16 | 2.60 GB |
| 32 | 3.93 GB |
| 50 | ~5.0 GB (**默认**） |
| 64 | 6.58 GB |

**Predictor 微调 (AMP fp16):**

| Batch Size | 显存 |
|-----------|------|
| 8 | 3.05 GB |
| 12 | 5.12 GB (**默认**） |
| 16 | 6.30 GB |
| 20 | OOM |

> `batch_size=12 + gradient_accumulation=4` → 等效 batch size 48。

---

## 5. 交付文件

### 新增文件

| 文件 | 功能 |
|------|------|
| `scripts/tdx_import.py` | TDX 本地数据导入工具（复权、因子缓存、连续性检测） |
| `finetune/config_tdx.py` | 单卡微调配置（后复权、TDX 时间范围、显存参数） |
| `finetune/train_tokenizer_tdx.py` | Tokenizer 单卡微调脚本 |
| `finetune/train_predictor_tdx.py` | Predictor 单卡微调脚本（AMP fp16 + 梯度累积） |
| `scripts/predict_sse.py` | 上证指数预测演示脚本 |
| `summary.md` | 本文档 |

### 修改文件

| 文件 | 修改内容 |
|------|----------|
| `finetune/dataset.py` | 增加 `config` 参数支持自定义配置；兼容 `date`/`datetime` 列名 |

---

## 6. 全量微调结果

### 数据导入
- 4,939 只股票，160MB 数据（train 103MB, val 34MB, test 24MB）
- 1.37M 训练样本，7.7 万验证样本

### Tokenizer 微调
| 指标 | 值 |
|------|-----|
| 耗时 | 0.85 小时 |
| Epochs | 30 |
| val_loss 初始 | 0.0132 |
| val_loss 最终 | **0.0096**（-27%） |

### Predictor 微调
| 指标 | 值 |
|------|-----|
| 耗时 | 5.38 小时 |
| Epochs | 30 |
| train_loss | 3.36 → 2.25（-33%） |
| val_loss 最佳 | **3.3384**（-3.4%） |
| 显存峰值 | ~6.3 GB / 7.6 GB |

### 模型输出
```
outputs/tdx_finetune/
├── tdx_tokenizer/checkpoints/best_model/   # 16MB
└── tdx_predictor/checkpoints/best_model/   # 391MB
```

---

## 7. 预测验证

使用微调后模型预测上证指数（sh000001）未来 20 个交易日走势：

```
基准: 4112.16 (2026-04-30)
走势: 先扬后抑 — 4112 → 4255 (+3.5%) → 4046 (-1.6%)
区间: 4025 ~ 4271 (振幅 5.98%)
涨跌: 11涨 / 8跌 / 1平
```

**注意**: 模型在个股后复权数据上微调，对上证指数的适用性有限；交易日历未过滤 A 股节假日。

---

## 8. 技能导出

创建了 `kronos-finetune` 技能，封装完整微调流程，存放于：

- 安装路径: `~/.claude/skills/kronos-finetune/`
- 导出路径: `~/hot-skills/installed/kronos-finetune/`

技能包含:
- `SKILL.md` — 环境检查、数据导入、模型下载、训练、预测 4 步流程
- `scripts/tdx_import.py` — 数据导入工具
- `scripts/predict_sse.py` — 预测演示脚本
- `references/config_tdx.py` — 微调参数配置

---

## 9. 已知限制

1. **历史长度**: TDX 本地数据仅约 2 年，无法覆盖完整牛熊周期
2. **过拟合**: Predictor train/val loss 差距较大（2.25 vs 3.51），数据跨度过短是主因
3. **网络依赖**: 复权因子需从新浪获取，离线环境降级为不复权
4. **复权因子缺失**: ~1,190 只板块指数/特殊品种无有效复权因子
5. **北交所**: TDX 本地无北交所数据
6. **单卡训练**: 脚本适配单 GPU，不支持多卡 DDP
7. **指数预测**: 模型在个股数据上训练，对指数预测的适用性未经验证
