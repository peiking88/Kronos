# 实践经验（2026-06 微调总结）

以下经验来自完整的 TDX 数据全参数微调实战，总耗时约 2.3h。

## 单阶段全参数微调

全参数微调让模型适配 A 股数据分布，Val Loss 从预训练的 ~4.2 降至 ~3.03。

## bf16 不需要 GradScaler

**问题**: Predictor 训练中使用 `GradScaler(enabled=use_amp)`，但 bf16 的动态范围与 fp32 相同，scaler 是无效操作。

**修复**: bf16 AMP 仅需 `torch.amp.autocast(dtype=torch.bfloat16)`，不需要 `GradScaler`。直接 `loss.backward()` + `optimizer.step()`。

## 复权因子缓存漂移（高危）

**问题**: 预测价格与实际市场价偏离 40%+，但模型本身没有问题。

**根因**: `compute_factor_from_xdxr` 依赖 kline 中的 `pre_close` 计算每步除权因子。不同时间获取的 kline 数据可能包含行情修正，导致同一股票的累积因子不同。

**防护**: 预测时不信任 factor cache，从数据本身推导因子：

```python
def derive_factor(code, df_hfq):
    """从 hfq 数据与 TDengine 原始数据对比，计算一致的复权因子。"""
    from taosws import connect
    conn = connect()
    try:
        r = conn.query(
            f"select ts, close from tdx.k_{code[2:]}_1d order by ts desc limit 1"
        )
        raw_close = float(list(r)[0][1])
    finally:
        conn.close()
    last_date = df_hfq.index[-1]
    hfq_close = float(df_hfq.loc[last_date]["close"])
    return hfq_close / raw_close
```

## 后复权因子外推

hfq 必须用 `direction="backward"`（向后找最近因子），qfq 用 `"forward"`。

排查方法：检查价格跳变：

```python
ret = df['close'].pct_change().dropna()
big_jumps = ret[ret.abs() > 0.5]
print(f'跳变>50%: {len(big_jumps)}次')  # 应为 0
```

## val/test 需要 lookback 补齐

val_data.pkl 和 test_data.pkl 只含切分区间数据。模型推理需要 90 日上下文，需从 train_data.pkl 末尾接 ~120 天。

## 早停节省训练时间

设置 `early_stop_patience=5`，全参数微调通常在 10 epoch 内收敛。

## 预测报告价格规范

**所有报告只显示实际市场价，不显示后复权价。** 换算用 `derive_factor`（从数据推导），不用 `load_factor`（从 cache 读）。

## 极端波动股票自动过滤

90 日回撤 >30% 或日波动率异常（>8%）的股票，预测不可靠，应自动跳过。

## 模型偏置认知

微调后模型存在两个固有偏置：

1. **均值回复偏置**: 预测走势通常"先延续短期方向，再向长期均值回归"
2. **空头偏置**: 训练数据含多轮熊市，模型倾向于低估上涨幅度

方向准确率约 50-55%，**模型的正确用法是价格区间估计而非方向博弈**。

## 依赖版本锁定

| 包      | 版本     | 说明                       |
| ------- | -------- | -------------------------- |
| taos-ws-py | latest | TDengine WebSocket 连接器     |

数据导入已切换至 TDengine，不再依赖 mootdx/opentdx/tdxdata。
