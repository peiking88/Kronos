# 实践经验（2026-05 微调总结）

以下经验来自一次完整的 TDX 数据微调实战，耗时约 6.4h（含 bug 排查），模型 val_loss=3.0185。

## 复权因子缓存漂移（高危）

**问题**: 预测价格与实际市场价偏离 40%+，但模型本身没有问题。

**根因**: `compute_factor_from_xdxr` 依赖 kline 中的 `pre_close` 计算每步除权因子。不同时间获取的 kline 数据可能包含行情修正，导致同一股票的累积因子不同。训练数据 pkl 用旧因子生成，预测时用新因子转换，两者不一致。

**实例（sh600353 旭光电子）**:

| 指标            | 值                          |
| --------------- | --------------------------- |
| 实际市场价      | 21.09                       |
| pkl 隐含 factor | 10.71（训练时用的旧 cache） |
| 新 cache factor | 18.60（重新计算的）         |
| 旧 factor 换算  | 21.09 ✓                     |
| 新 factor 换算  | 12.14 ✗（偏低 42%）         |

**防护**: 预测时不信任 factor cache，从数据本身推导因子：

```python
def derive_factor(code, df_hfq):
    """从 hfq 数据与 TDX 原始数据对比，计算一致的复权因子。"""
    from mootdx.reader import Reader
    reader = Reader.factory(market="std", tdxdir=TDX_DIR)
    raw_df = reader.daily(symbol=code[2:])
    last_date = df_hfq.index[-1]
    raw_before = raw_df[raw_df.index <= pd.Timestamp(last_date)]
    raw_close = float(raw_before.iloc[-1]["close"])
    hfq_close = float(df_hfq.iloc[-1]["close"])
    return hfq_close / raw_close  # 保证与 pkl 数据一致
```

**关键认知**:

- 清空 factor cache 不能解决问题——重新算仍可能不同
- `derive_factor` 保证与当前 pkl 数据一致，是最可靠的防护
- 根本修复需要用正确 factor 重新导入数据 + 重新微调

**排查方法**:

```python
# 1. 检查 factor 一致性
factor_cache = load_factor(code)         # 从 cache 读
factor_derived = derive_factor(code, df)  # 从数据推导
if abs(factor_cache / factor_derived - 1) > 0.05:
    print(f"⚠️ {code} factor 不一致: cache={factor_cache:.4f}, derived={factor_derived:.4f}")

# 2. 检查 pkl 数据的 hfq 是否正确（与 TDX 原始数据对比）
raw_close = float(raw_df.iloc[-1]["close"])
correct_hfq = raw_close * factor_cache   # 用当前 cache 算"正确的" hfq
pkl_hfq = float(df.iloc[-1]["close"])
if abs(pkl_hfq / correct_hfq - 1) > 0.05:
    print(f"⚠️ {code} pkl hfq 偏离正确值 {100-pkl_hfq/correct_hfq*100:.1f}%")
```

## 后复权因子外推

**问题**: 因子缓存末次日期（如 2025-07-01）可能早于数据截止日。若因子未正确外推，末次除权日之后的数据为原始不复权价，导致价格序列出现数倍跳变。

**根因**: `merge_asof` 的 `direction` 参数影响。hfq 必须用 `direction="backward"`（向后找最近因子），qfq 用 `"forward"`。`_apply_factor` 方法已正确处理，但**临时手写因子应用代码时容易写死错误方向**。

**排查方法**:

```python
# 检查是否有异常跳变
ret = df['close'].pct_change().dropna()
big_jumps = ret[ret.abs() > 0.5]
print(f'跳变>50%: {len(big_jumps)}次')  # 应为 0
```

**修复**: 直接用 `scripts/tdx_import.py --dividend-type back` 重导，不要手写因子应用逻辑。

## val/test 需要 lookback 补齐

val_data.pkl 和 test_data.pkl 只含切分区间数据。模型推理需要 90 日上下文，需从 train_data.pkl 末尾接 ~120 天：

```python
train_tail = train_data[code].iloc[-120:]
merged = pd.concat([train_tail, val_data[code]])
```

否则 val 集 sample 数为 0，训练无法评估。

## 早停节省训练时间

30 轮训练中最佳轮次通常在 25-28，最后 2-5 轮已过拟合。设置 `early_stop_patience=5`：

- Predictor 实际运行 30 轮（本次持续改善至 27 轮，未触发）
- 典型场景下可节省 15-25% 训练时间
- 最佳模型自动保存，不受早停影响

## 预测报告价格规范

**所有报告只显示实际市场价，不显示后复权价。** 换算方式必须用 `derive_factor`（从数据推导），不要用 `load_factor`（从 cache 读）：

```python
# ✓ 正确：从数据推导（保证与训练数据一致）
factor = derive_factor(code, df_hfq)
actual_price = hfq_price / factor

# ✗ 危险：从 cache 读（可能与训练数据不一致）
factor = pd.read_pickle(f'.factor_cache/{code}.pkl')['factor'].iloc[-1]
actual_price = hfq_price / factor
```

指数（sh000001/sz399006）因子为 1.0，无需换算。涨跌幅不变——复权只改变绝对价格。

## 极端波动股票自动过滤

90 日回撤 >30% 或日波动率异常（>8%）的股票，Tokenizer 码本映射可能崩溃，产生无效预测（指数暴涨、高低价倒置）。

前置过滤规则：

```python
lookback_ret = df['close'].pct_change(90).iloc[-1]
daily_vol = df['close'].pct_change().iloc[-20:].std()
if abs(lookback_ret) > 0.30 or daily_vol > 0.08:
    skip("预测不可靠")
```

## 模型偏置认知

微调后模型存在两个固有偏置：

1. **均值回复偏置**: 预测走势通常"先延续短期方向，再向长期均值回归"（N 型走势）
2. **空头偏置**: 训练数据含多轮熊市，模型倾向于低估上涨幅度

方向准确率约 50-55%，不优于随机。**模型的正确用法是价格区间估计而非方向博弈。**

## 依赖版本锁定

| 包      | 版本     | 关键修复                       |
| ------- | -------- | ------------------------------ |
| mootdx  | >=2.0.3  | `_clean_code` 统一处理市场前缀 |
| opentdx | >=0.5.10 | mootdx 适配层依赖              |
| tdxdata | >=0.8.4  | errors 模块导出补全            |

升级顺序: opentdx → mootdx → tdxdata（按依赖链）。

## 一键预测

```bash
.venv/bin/python scripts/predict_stocks.py sh600000 sz002741 sz300450
```

输出 md 报告含预测结果（实际市场价）、历史回测准确度、置信区间。
