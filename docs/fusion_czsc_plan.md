# Kronos + CZSC + ChronosX 融合方案

> 版本：v2.0 | 日期：2026-06-04 | 状态：待实施

## 一、方案概述

### 1.1 目标

为 Kronos 预测模型注入 CZSC（缠中说禅）结构化特征，通过 ChronosX 的 IIB（输入注入块）残差注入机制，提升**方向准确率**和**转折点识别**能力。

### 1.2 核心思路

```
原始 K 线数据 (OHLCVA)
        │
        ├──→ Kronos Tokenizer → Token Embedding [B, T, 832]
        │                                    │
        └──→ CZSC 特征提取 → 7 维协变量 [B, T, 7]
                                        │
                                   ┌────┴────┐
                                   │   IIB    │  ← 残差注入（仅训练此模块）
                                   │  FFN块   │
                                   └────┬────┘
                                        │
                              Embedding + IIB输出
                                        │
                              Kronos Transformer（冻结）
                                        │
                                    预测输出
```

**关键决策**：仅实现 IIB，暂不实现 OIB。原因见[第 3.4 节](#34-oib暂不实现)。

### 1.3 与原方案的区别

| 项目 | 原始文档方案（v1.0） | 本方案（v2.0） |
|------|---------------------|---------------|
| 协变量维度 | 5 维 | **7 维**（新增强分型、背驰、拟合度） |
| 注入方式 | IIB + OIB | **仅 IIB** |
| 未来协变量 | 三种策略 | **不使用** |
| ChronosX 代码 | 直接使用 | **不使用**，仅借鉴设计思想 |
| CZSC 买卖点 | 未包含 | **包含**（通过背驰信号 D5 编码） |

---

## 二、CZSC 7 维特征定义

### 2.1 特征总览

| 维度 | 名称 | 值域 | 类型 | CZSC API | 信息含义 |
|------|------|------|------|---------|---------|
| D1 | 强分型 | [-2, +2] | 离散 | `fx.mark` + `fx.power_str` | 转折点 + 强度 |
| D2 | 笔方向 | {-1, 0, +1} | 离散 | `bi.direction` | 当前趋势方向 |
| D3 | 中枢相对位置 | [-2, 3] | 连续 | 从 `bi_list` 计算 | 震荡区间归属 |
| D4 | 笔力度 | 连续 | 连续 | `bi.power` | 趋势动量强度 |
| D5 | 背驰信号 | [-0.5, +0.5] | 连续 | 相邻同向笔 `power` 比较 | 买卖点信号 |
| D6 | 笔拟合度 | [0, 1] | 连续 | `bi.rsq` | 趋势线性度 |
| D7 | 嵌套笔数 | 连续 | 连续 | `len(bi.fake_bis)` | 笔内结构复杂度 |

### 2.2 D1：强分型

**定义**：将分型（Fractal）按强度（power_str）分为三级，顶底方向分别编码。

**缠论含义**：三根相邻 K 线中，中间那根形成局部极值。强度由中间 K 线与两侧 K 线的高低点差距决定——差距越大，分型越强，转折信号越可靠。

| 编码值 | 含义 | 出现频率 | 直觉 |
|--------|------|---------|------|
| +2 | 强顶分型（power_str="强"，Mark.G） | ~6.4% | 高可信做空信号 |
| +1 | 中顶分型（power_str="中"，Mark.G） | ~12.0% | 普通做空信号 |
| 0 | 弱分型或无分型 | ~63.6% | 无明确信号 |
| -1 | 中底分型（power_str="中"，Mark.D） | ~12.2% | 普通做多信号 |
| -2 | 强底分型（power_str="强"，Mark.D） | ~5.8% | 高可信做多信号 |

**提取代码**：
```python
def get_strong_fractal(fx):
    """D1: 强分型编码"""
    if fx is None:
        return 0
    strength = {'弱': -1, '中': 0, '强': 1}.get(fx.power_str, 0)
    if fx.mark == Mark.G:  # 顶分型
        return 1 + strength  # 弱=0, 中=1, 强=2
    elif fx.mark == Mark.D:  # 底分型
        return -(1 + strength)  # 弱=0, 中=-1, 强=-2
    return 0
```

### 2.3 D2：笔方向

**定义**：当前 K 线所属笔（Bi）的方向。

**缠论含义**：从顶分型到底分型构成"向下笔"，从底分型到顶分型构成"向上笔"。笔代表了一段完整的单向价格运动。

| 编码值 | 含义 | 出现频率 |
|--------|------|---------|
| +1 | 向上笔（Direction.Up） | ~49% |
| -1 | 向下笔（Direction.Down） | ~48% |
| 0 | 未归属笔 | ~3% |

**提取代码**：
```python
def get_bi_direction(bi):
    """D2: 笔方向"""
    if bi is None:
        return 0
    if bi.direction == Direction.Up:
        return 1
    elif bi.direction == Direction.Down:
        return -1
    return 0
```

### 2.4 D3：中枢相对位置

**定义**：当前收盘价相对于最近中枢（Zhongshu）的归一化位置。

**缠论含义**：连续 3 笔的价格重叠区间构成中枢。中枢上沿 ZG = max(3 笔各自最低点)，中枢下沿 ZD = min(3 笔各自最高点)。价格在中枢内表示震荡，突破中枢表示趋势启动。

| 编码值 | 含义 | 直觉 |
|--------|------|------|
| < 0 | 在中枢下方 | 超卖或向下突破 |
| 0 | 在中枢下沿 | 强支撑位 |
| 0.5 | 在中枢正中间 | 典型震荡 |
| 1 | 在中枢上沿 | 强阻力位 |
| > 1 | 在中枢上方 | 超买或向上突破 |

**提取代码**：
```python
def compute_zhongshu(bi_list):
    """从笔列表计算中枢：连续3笔重叠区间"""
    zs_list = []
    for i in range(2, len(bi_list)):
        b1, b2, b3 = bi_list[i-2], bi_list[i-1], bi_list[i]
        zg = max(b1.low, b2.low, b3.low)  # 中枢上沿
        zd = min(b1.high, b2.high, b3.high)  # 中枢下沿
        if zg > zd:
            zs_list.append({'zg': zg, 'zd': zd,
                            'start': b1.sdt, 'end': b3.edt})
    return zs_list

def get_zhongshu_position(bar, zs_list):
    """D3: 中枢相对位置"""
    for zs in reversed(zs_list):
        if zs['start'] <= bar.dt <= zs['end']:
            span = zs['zg'] - zs['zd']
            if span > 0:
                return max(-2.0, min(3.0,
                    (bar.close - zs['zd']) / span))
    return 0.5  # 不在中枢内，返回中性值
```

### 2.5 D4：笔力度

**定义**：当前笔的价格变动幅度（power），经 z-score 归一化。

**缠论含义**：笔力度 = `(笔最高价 - 笔最低价) / 笔最低价`。力度越大，趋势越强。

| 统计量 | 值 |
|--------|-----|
| 均值 | ~1.05 |
| 范围 | [0.33, 2.36] |
| 归一化后 | z-score，均值=0 |

**提取代码**：
```python
# 预计算统计量
all_powers = [b.power for b in c.bi_list]
mean_power = np.mean(all_powers)
std_power = np.std(all_powers) + 1e-8

def get_bi_power(bi, mean_power, std_power):
    """D4: 笔力度归一化"""
    if bi is None:
        return 0
    return (bi.power - mean_power) / std_power
```

### 2.6 D5：背驰信号（买卖点）

**定义**：当前笔力度与同向前一笔力度的差值。正值表示底背驰（买方信号），负值表示顶背驰（卖方信号）。

**缠论含义**：这是缠论买卖点的核心机制。当价格创新低（新底分型）但下跌力度减弱（背驰），说明空方衰竭，是买入信号。反之亦然。

**回测结论**：D1（强分型）+ D5（背驰）的组合在历史回测中具有**较高的方向准确性和收益率**。

| 编码值 | 含义 | 出现频率 |
|--------|------|---------|
| > 0 | 底背驰（买方信号）| ~71% 的K线 |
| < 0 | 顶背驰（卖方信号）| ~23% 的K线 |
| = 0 | 无背驰 | ~6% |

**组合信号统计**：

| D1 + D5 组合 | 含义 | 出现频率 |
|-------------|------|---------|
| 强底(-2) + 底背驰(>0) | **强买信号** | ~3.6% |
| 强顶(+2) + 顶背驰(<0) | **强卖信号** | ~0.6% |
| 中底(-1) + 底背驰(>0) | 普通买信号 | ~10% |
| 中顶(+1) + 顶背驰(<0) | 普通卖信号 | ~3% |

> **注意**：D1 和 D5 作为独立维度输入 IIB，**不做预组合**。让模型通过 IIB 的 FFN 非线性变换自行学习维度间的组合规则。

**提取代码**：
```python
def get_beichi(bi_list, current_bi):
    """D5: 背驰信号

    返回值:
        正值 → 底背驰（向下笔力度减弱）→ 买方信号
        负值 → 顶背驰（向上笔力度减弱）→ 卖方信号
    """
    if current_bi is None:
        return 0.0
    # 找同方向的前一笔
    prev = None
    for b in reversed(bi_list):
        if b.direction == current_bi.direction and b is not current_bi:
            prev = b
            break
    if prev is None:
        return 0.0
    # 力度差
    diff = (prev.power - current_bi.power) / (prev.power + 1e-8)
    # 向下笔力度减弱 = 底背驰 = 正值（买）
    # 向上笔力度减弱 = 顶背驰 = 负值（卖）
    if current_bi.direction == Direction.Down:
        return diff
    else:
        return -diff
```

### 2.7 D6：笔拟合度

**定义**：笔内 K 线收盘价对线性趋势的 R² 拟合度。

**含义**：R² 接近 1 表示价格沿直线运动（强趋势），接近 0 表示价格随机波动（弱趋势或震荡）。

| 编码值 | 含义 |
|--------|------|
| 接近 1 | 强趋势，价格线性运动 |
| 接近 0 | 弱趋势或震荡 |
| 中间值 | 中等趋势强度 |

**提取代码**：
```python
def get_bi_rsq(bi):
    """D6: 笔拟合度"""
    if bi is None:
        return 0
    return bi.rsq  # 原值，范围 [0, 1]
```

### 2.8 D7：嵌套笔数

**定义**：笔内包含的子笔（fake_bis）数量，经 z-score 归一化。

**含义**：嵌套笔越多，说明笔内结构越复杂，可能是趋势中继或反转蓄力阶段。

**提取代码**：
```python
# 预计算统计量
all_fakes = [len(b.fake_bis) for b in c.bi_list]
mean_fk = np.mean(all_fakes)
std_fk = np.std(all_fakes) + 1e-8

def get_fake_bi_count(bi, mean_fk, std_fk):
    """D7: 嵌套笔数归一化"""
    if bi is None:
        return 0
    return (len(bi.fake_bis) - mean_fk) / std_fk
```

---

## 三、IIB 注入设计

### 3.1 注入位置

在 `model/kronos.py` 的 `Kronos.forward()` 方法中，`HierarchicalEmbedding` 之后、`TemporalEmbedding` 之前插入：

```
原始流程（kronos.py:254-258）：
  x = self.embedding([s1_ids, s2_ids])       # L254: [B, T, 832]
  if stamp is not None:                        # L255
      x = x + self.time_emb(stamp)            # L257
  x = self.token_drop(x)                      # L258

修改后：
  x = self.embedding([s1_ids, s2_ids])       # L254: [B, T, 832]
  x = x + self.iib(x, past_covariates)        # ← 新增 IIB 注入
  if stamp is not None:
      x = x + self.time_emb(stamp)
  x = self.token_drop(x)
```

### 3.2 IIB 模块结构

借鉴 ChronosX 的 `InputInjectionBlock`（`chronos-forecasting/src/chronosx/injection_blocks/input_injection_block.py`），适配 Kronos 的维度参数：

```
输入:
  token_embeddings: [B, T, 832]   ← Kronos d_model
  covariates:       [B, T, 7]     ← CZSC 7 维特征

内部:
  emb_in:  Linear(832 → 256)      ← 投影嵌入到隐藏维度
  cov_in:  Linear(7   → 256)      ← 投影协变量到隐藏维度
  concat:  [256 + 256] = 512
  ffn:     Linear(512 → 256) → ReLU → Linear(256 → 832)

输出:
  token_embeddings + ffn(ReLU(concat(emb_in(tokens), cov_in(covs))))
  即：原始嵌入 + 残差修正
```

### 3.3 参数量

| 层 | 计算 | 参数量 |
|----|------|--------|
| `emb_in` | 832 × 256 + 256 | 213,248 |
| `cov_in` | 7 × 256 + 256 | 2,048 |
| `FFN 层1` | 512 × 256 + 256 | 131,328 |
| `FFN 层2` | 256 × 832 + 832 | 213,664 |
| **IIB 总计** | — | **~560K** |

占 Kronos 总参数量（102.3M）的 **0.55%**。

### 3.4 OIB 暂不实现

OIB（输出注入块）需要未来协变量（预测时段的 CZSC 结构），但未来缠论结构本质上是未知的。文档提出的三种策略（延续假设/概率加权/Kronos 辅助）都存在理论缺陷：

- **延续假设**：趋势反转时完全失效
- **概率加权**：需大量历史统计，且 A 股结构性变化频繁
- **Kronos 辅助**：循环依赖（用预测结果作为协变量输入预测）

如果 IIB 实验证明 CZSC 特征有增益，可考虑在后续阶段引入 OIB。

---

## 四、实现计划

### 4.1 需修改的文件

| 文件 | 改动内容 | 改动量 |
|------|---------|--------|
| `model/kronos.py:254` | `forward()` 中插入 IIB 调用（1 行） | 微量 |
| `model/kronos.py:278-308` | `decode_s1()` 增加 `past_covariates` 参数 | ~5 行 |
| `model/kronos.py:310-328` | `decode_s2()` 增加 `past_covariates` 参数 | ~5 行 |
| `model/kronos.py:389-469` | `auto_regressive_inference()` 传递协变量 | ~15 行 |
| `finetune/dataset.py` | `__getitem__` 增加 CZSC 特征计算 | ~30 行 |
| `finetune/train_predictor_tdx.py` | 训练循环传入协变量 | ~10 行 |
| `scripts/predict_stocks.py` | 预测时传入协变量 | ~10 行 |
| `scripts/predict.py` | 预测时传入协变量 | ~10 行 |

### 4.2 需新增的文件

| 文件 | 内容 |
|------|------|
| `model/covariate.py` | IIB 模块 + CZSC 7 维特征提取器 + 中枢计算 + 背驰计算 |
| `scripts/build_czsc_cache.py` | 预计算 CZSC 特征并缓存（避免每次推理重复计算） |

### 4.3 实现阶段

#### 阶段 1：IIB 模块 + CZSC 特征提取（1 周）

```
目标：完成 IIB 模块和 CZSC 特征提取代码
验证：单元测试通过
  ├─ model/covariate.py: IIB 模块（借鉴 ChronosX InputInjectionBlock）
  ├─ model/covariate.py: CZSC 特征提取器（7 维）
  ├─ model/kronos.py: forward() / decode_s1() / decode_s2() 接入 IIB
  └─ 测试：输入 dummy 数据，确认输出形状正确
```

#### 阶段 2：训练数据构建（3 天）

```
目标：为训练集的每根 K 线计算 CZSC 7 维特征
验证：特征分布符合预期（无 NaN/Inf）
  ├─ scripts/build_czsc_cache.py: 批量预计算 CZSC 特征
  ├─ finetune/dataset.py: QlibDataset 加载 CZSC 特征
  └─ 验证：特征值范围和分布统计
```

#### 阶段 3：IIB 训练 + 对比实验（1 周）

```
目标：训练 IIB，对比基线
验证：方向准确率、MAPE 对比

实验设计：
  ├─ 基线 A：Kronos（冻结，无协变量）
  ├─ 实验 B：Kronos + IIB（4 维：D1-D4）
  ├─ 实验 C：Kronos + IIB（7 维：D1-D7）
  └─ 对比指标：
      - 方向准确率（D1-D5 分别统计）
      - MAPE（D1-D5 分别统计）
      - 极端预测数量（|涨跌| > 10%）
      - 强分型 + 背驰时的方向准确率

训练配置：
  - Kronos 主体冻结，仅训练 IIB
  - 学习率: 1e-3（适配器训练，高于全参数微调）
  - 优化器: AdamW
  - 训练步数: 2000-5000
  - 早停 patience: 5
```

#### 阶段 4：集成到预测流程（3 天）

```
目标：将训练好的 IIB 集成到 predict_stocks.py 和 predict.py
验证：端到端预测流程正常
  ├─ scripts/predict_stocks.py: 预测时计算并传入 CZSC 特征
  ├─ scripts/predict.py: 同上
  └─ 验证：70 只自选股批量预测，输出格式不变
```

---

## 五、CZSC 环境说明

### 5.1 依赖

```bash
# CZSC 库（Rust + Python 混合架构）
cd ~/peiking88/czsc
pip install -e .

# 缺失依赖
pip install wbt
```

### 5.2 关键 API

```python
from czsc._native import CZSC, RawBar, Freq, Mark, Direction

# 构造 CZSC 对象
c = CZSC(bars)  # bars: List[RawBar]

# 分型（Fractal）
c.fx_list       # List[FX] — 所有分型
fx.mark         # Mark.G=顶分型, Mark.D=底分型
fx.power_str    # "强"/"中"/"弱" — 分型强度
fx.has_zs       # bool — 分型是否在中枢内

# 笔（Bi）
c.bi_list       # List[BI] — 所有完成的笔
bi.direction    # Direction.Up / Direction.Down
bi.power        # float — 笔力度
bi.length       # int — 笔内K线数
bi.slope        # float — 笔斜率
bi.rsq          # float — 笔拟合度 R²
bi.SNR          # float — 笔信噪比
bi.fake_bis     # List[FakeBI] — 笔内嵌套的子笔

# 未完成笔（当前正在形成的笔）
c.ubi           # 未完成笔信息
```

### 5.3 注意事项

- CZSC 基础类**无直接中枢属性**，中枢需从 `bi_list` 自行计算（见 D3 代码）
- CZSC 基础类**无直接买卖点属性**，买卖点通过背驰计算间接获取（见 D5 代码）
- CZSC v1.0 核心算法已迁移至 Rust（PyO3），特征提取性能不是瓶颈
- 特征提取时**只使用已完成笔**（`c.bi_list`），不使用未完成笔（`c.ubi`）

---

## 六、预期收益与风险

### 6.1 预期收益

| 指标 | 当前基线 | 预期改善 | 改善幅度 |
|------|---------|---------|---------|
| 方向准确率 (D1) | 54.0% | 55-57% | +1~3 pp |
| 方向准确率 (D5) | 51.4% | 52-54% | +1~3 pp |
| MAPE (D1) | 2.8% | 2.6-2.8% | -0.0~-0.2 pp |
| MAPE (D5) | 6.4% | 6.0-6.4% | -0.0~-0.4 pp |
| 极端预测数 | 4 只 | 2-3 只 | -25~50% |
| 强分型+背驰时方向准确率 | 未知 | 预计 58-65% | 关键提升点 |

### 6.2 风险

| 风险 | 等级 | 应对措施 |
|------|------|---------|
| CZSC 特征与 OHLCV 量化后隐含信息冗余 | 🟡 中 | 消融实验验证每个维度的边际贡献 |
| D5 背驰信号偏度大（71% 买 vs 23% 卖） | 🟡 中 | clip ±0.5 并标准化 |
| 强买/强卖信号极稀疏（3.6% / 0.6%） | 🟡 中 | D1 和 D5 作为独立维度输入，不做预组合 |
| 训练数据中 CZSC 特征计算不一致 | 🔴 高 | 预计算并缓存，确保训练/推理使用同一份特征 |
| IIB 注入后反而降低性能 | 🟡 中 | 保留基线模型，实验不通过则回退 |

### 6.3 回退方案

如果 IIB 实验后方向准确率无显著提升（< 1pp），则：
1. 放弃协变量注入方案
2. 保留 CZSC 特征提取代码作为独立分析工具
3. 转向其他提升方向（更长上下文、概率预测输出、更好的采样策略）

---

## 七、参考来源

- [ChronosX 源码（chronosx 分支）](https://github.com/amazon-science/chronos-forecasting/tree/chronosx)
- [ChronosX 论文（AISTATS 2025）](https://arxiv.org/abs/2403.13978)
- [CZSC 库（缠中说禅技术分析工具）](https://github.com/waditu/czsc)
- [Kronos 模型（本项目）](https://huggingface.co/NeoQuasar/Kronos-base)
- [Chronos-2 股票预测系统（AK60000）](https://github.com/AK60000/chronos-2-finace)
