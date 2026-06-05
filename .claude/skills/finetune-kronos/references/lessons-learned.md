# 实践经验（2026-06 两阶段微调总结）

以下经验来自完整的 TDX 数据两阶段微调实战，总耗时约 2.3h。

## 两阶段训练策略

### IIB-only 直接训练会过拟合

**问题**: 冻结 Kronos 主体（103M 参数），仅训练旧版 IIB（560K 参数，0.55%），Val Loss 从 Epoch 1 开始上升。

**根因**: IIB 参数太少，学习率过高（1e-3），模型容量不足以学会有意义的协变量注入。

**修复**: 两阶段策略：
1. **Phase 1** 全参数微调（10 epoch，lr=4e-5），让模型先适配 A 股数据分布
2. **Phase 2** IIB + CZSC 渐进式解冻（30 epoch），在 Phase 1 基础上注入协变量

**效果**: Val Loss 从 IIB-only 的 3.73 → Phase 1 的 3.03 → Phase 2 的 2.78（总改善 25.5%）。

### CZSC D5 背驰极端异常值

**问题**: D5 背驰信号文档标注范围 ±0.5，实际数据中出现 ±49.79 的极端值，0.82% 的值超过 |10|。

**根因**: `get_beichi()` 中 `diff = (prev.power - current_bi.power) / (prev.power + 1e-8)`，当 `prev.power` 接近零时分母极小，比值爆炸。

**修复**: tanh 软裁剪：

```python
raw_diff = prev.power - current_bi.power
diff = np.tanh(raw_diff / (abs(prev.power) + abs(current_bi.power) + 1e-8))
```

修复后 D5 范围 [-0.76, 0.75]，std=0.308，在合理范围内。

**关键**: CZSC 缓存重建必须用**单进程**（`-n 1`）。多进程的 worker 进程可能加载旧模块代码，导致 D5 修复不生效。

### bf16 不需要 GradScaler

**问题**: Predictor 训练中使用 `GradScaler(enabled=use_amp)`，但 bf16 的动态范围与 fp32 相同，scaler 是无效操作。

**修复**: bf16 AMP 仅需 `torch.amp.autocast(dtype=torch.bfloat16)`，不需要 `GradScaler`。直接 `loss.backward()` + `optimizer.step()`。

### IIB 架构需要足够深度

**问题**: 旧版单层 FFN（concat→ReLU→Linear）仅 560K 参数，对 7 维 CZSC 协变量的表达能力不足。

**修复**: 升级为 2 层残差 MLP + LayerNorm（956K 参数）：

```
emb_proj(832→256) + cov_proj(7→256) → 相加
  → LayerNorm → Linear(256→512) → GELU → Linear(512→256) → Dropout(0.3) → 残差
  → LayerNorm → Linear(256→512) → GELU → Linear(512→256) → Dropout(0.3) → 残差
  → out_proj(256→832) → Dropout(0.3)
```

**关键**: 新旧架构通过 `n_layers` 参数兼容。`from_pretrained` 使用旧架构加载，Phase 2 手动替换 IIB 为升级版。

### 渐进式解冻策略有效

三阶段策略每个阶段都带来 Val Loss 改善：

| 阶段 | 可训练参数 | Val Loss 范围 | 改善 |
|------|-----------|---------------|------|
| A (iib_only) | 0.93% | 2.96 → 2.92 | IIB 学会了 CZSC 注入 |
| B (iib+top4) | 33% | 2.90 → 2.87 | 后 4 层适配协变量信号 |
| C (全参数) | 100% | 2.85 → 2.78 | 全模型微调（极低 LR） |

Stage C 的关键：使用极低学习率（base=5e-6），保护 Phase 1 已学到的 A 股分布表示。

## 复权因子缓存漂移（高危）

**问题**: 预测价格与实际市场价偏离 40%+，但模型本身没有问题。

**根因**: `compute_factor_from_xdxr` 依赖 kline 中的 `pre_close` 计算每步除权因子。不同时间获取的 kline 数据可能包含行情修正，导致同一股票的累积因子不同。

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

设置 `early_stop_patience=5`，Phase 1 全参数微调通常在 10 epoch 内收敛。

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

| 包      | 版本     | 关键修复                       |
| ------- | -------- | ------------------------------ |
| mootdx  | >=2.0.3  | `_clean_code` 统一处理市场前缀 |
| opentdx | >=0.5.10 | mootdx 适配层依赖              |
| tdxdata | >=0.8.4  | errors 模块导出补全            |
| czsc    | latest   | CZSC 缠论分析 + native 模块    |

升级顺序: opentdx → mootdx → tdxdata。
