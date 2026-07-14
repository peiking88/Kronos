# 工作摘要

**时间:** 2026-07-14
**任务:** 预测脚本数据获取逻辑审查 + 禁用无表标的的本地 pickle 回退

## 审查发现

预测脚本 (`scripts/predict_stocks.py`) 通过 `ensure_fresh_data` 从 TDengine 拉取最新 1d 数据，但存在以下陈旧数据回退路径：

1. `main()` 中 `fresh_cache` 未命中的标的回退到 `get_data()` (读本地 pickle)
2. `process_single()` 中同样逻辑
3. `prefetch_factors._fetch_one()` 中同样逻辑（单测覆盖，保留）

DB 实测现状：当前 TDengine 仅 46 个 1d 子表，多数标的（如 sh600000、sz000001）无 1d 表。对这些标的原逻辑会静默回退到本地 pickle（末笔 ~2026-06-25），且 `ensure_fresh_data` 的 import 失败/缺表被 `_query_daily_kline` 的 `except Exception: pass` 吞掉。

## 修复

### 1. 禁止回退到无表标的的本地 pickle

- `ensure_fresh_data()` 改为返回 `(fresh_cache, missing_codes)`：
  - 过期的标的：能从 TDengine 导入并合并 → 进入 `fresh_cache`；缺表或导入失败 → 进入 `missing_codes`，**不再保留旧数据**
  - 未过期的标的：直接用本地数据填充 `fresh_cache`，避免再次回退读 pickle
- `main()` 在获取因子 / 实时行情 / 预测前剔除 `missing_codes` 并记入 `errors`，全不可用时 `sys.exit(1)`
- `process_single()` 移除 `get_data()` 回退，`fresh_cache` 缺失直接返回 error
- `all_data` 构建移除 `get_data()` 回退，仅从 `fresh_cache` 取数
- `--no-import` 模式保留原行为（用户显式选择使用本地数据），单独走 `get_data()` 分支

### 2. 修复跨边界异常检测的 34 年误报（附带发现）

`sh999999`（上证综指）因 `k_sh999999_1d` 是 1992-2026 的全历史表，跨边界校验把旧末笔(2026-07-13 close=3913.79) 与旧首笔(1992-01-02 close=293.75) 比较，触发 0.1x 跳变误判被丢弃。旧逻辑用保留本地数据掩盖了该问题。

修复：`_detect_close_anomaly()` 新增 `ref_date` 参数，跨边界校验仅比较 `ref_close` 与 df 中首条**严格晚于 ref_date** 的 bar，跳过全历史回填场景中早于 ref_date 的旧 bar。

## 验证

- `pytest tests/` 全套 **41 passed**
- 抽样冒烟（sh600549/sz399006/sh999999 有表，sh600000/sz000001/sz002741 无表）：
  - `sh600549, sz399006, sh999999, sz002741` 进入 `fresh_cache`，末笔 2026-07-13
  - `sh600000, sz000001` 进入 `missing_codes`
  - `sh999999`（上证综指）不再误判丢弃

## 变更文件

| 文件 | 改动 |
|---|---|
| `scripts/predict_stocks.py` | `ensure_fresh_data` 返回 `(fresh_cache, missing_codes)`，缺表/异常标的不保留旧数据且标记缺失；`main()` 剔除 missing_codes 并禁用一切 `get_data()` 回退；`process_single` 禁用回退；`--no-import` 保留原行为；`_detect_close_anomaly` 新增 `ref_date` 避免跨 34 年误判 |
| `summary.md` | 本文件 |

## 最近提交
```43da2f1 修复: 适配 tdx-cpp TDengine 子表市场前缀化（v0.13.6/7/8）
```
