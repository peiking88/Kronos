# 两阶段微调方案：全参数微调 → IIB+CZSC 训练（v2 评审修订）

## 背景

当前 Predictor 微调存在严重过拟合问题（IIB-only 模式下 Val Loss 不降反升），且训练效率低下：
- Tokenizer 训练无 AMP（纯 fp32），仅用 5GB/16GB 显存
- Predictor 每 batch 重复调用 `tokenizer.encode()`（纯浪费）
- IIB 学习率过高（1e-3）、正则不足（dropout=0.1）
- CZSC D5 背驰特征有极端异常值（max=49.79，文档标注 ±0.5）
- IIB 架构过浅（单层 FFN，56 万参数）
- 无渐进式解冻策略

**目标**：分两阶段训练——先全参数微调打好基础，再训练 IIB+CZSC 协变量注入。

### 两阶段定位

- **Phase 1**（全参数微调）：让预训练 Kronos-base 适配 A 股数据分布，是「热身」阶段，epoch 少（~10 epoch），目标是 Val Loss 收敛
- **Phase 2**（IIB+CZSC 注入）：在 Phase 1 基础上，**仅训练 IIB + 解冻部分层**，不改变 Phase 1 已学到的表示。Stage C 解冻全参数时使用极低学习率（5e-6），仅做微调

---

## Phase 1：全参数快速微调（无 IIB/CZSC）

### 步骤 1.1：Tokenizer 训练添加 bf16 AMP

**文件**：`finetune/train_tokenizer_tdx.py`

- 添加 `--no-amp` CLI 参数
- 训练循环添加 `torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16)`
- **不使用 GradScaler**（bf16 动态范围足够，GradScaler 对 bf16 是无效操作）
- 用 `loss.backward()` 直接反向传播（同原始逻辑），仅 outer autocast 即可
- 验证循环同样包裹 autocast

**文件**：`finetune/config_tdx.py`

- `batch_size` 50 → 128（bf16 显存减半，可大幅增加）

### 步骤 1.2：运行 Tokenizer 微调

```bash
python finetune/train_tokenizer_tdx.py --data-dir data/tdx_import/1d
```

### 步骤 1.3：Predictor 全参数微调模式

**文件**：`finetune/train_predictor_tdx.py`

核心改动——添加 `--phase full|iib` CLI 参数：

**Phase `full` 模式（新增）**：
- 使用原始 `QlibDataset`（不做 token 预缓存——见下方说明）
- 加载 tokenizer（frozen）+ predictor（全参数可训练）
- **不传入协变量**（`past_covariates=None`，IIB 自动返回零，对模型无影响）
- 学习率 `predictor_learning_rate=4e-5`
- Epochs 降为 10（Phase 1 仅是热身，不宜过拟合）
- 训练循环与现有逻辑一致，仅跳过协变量

**不引入 Token 磁盘缓存的理由**：
- `tokenizer.encode()` 在 RTX 5080 上每个 batch 仅 ~2ms，不是瓶颈
- 真正的浪费是跨 epoch 重复编码同一窗口——但这个可通过 epoch 级内存缓存解决（未来优化）
- 3.4GB 磁盘缓存 + 新 Dataset 类 + index_map 一致性维护 = 过度工程化
- 当前优先解决「训练不收敛」而非「训练不够快」

**文件**：`finetune/config_tdx.py`

- 新增 `phase = 'full'`
- 新增 `phase1_epochs = 10`（Phase 1 专用 epoch 数）
- Phase 1: `freeze_predictor=False`, `use_iib=False`

### 步骤 1.4：运行 Predictor 全参数微调

```bash
python finetune/train_predictor_tdx.py --phase full --data-dir data/tdx_import/1d
```

---

## Phase 2：IIB + CZSC 训练

### 步骤 2.1：修复 D5 背驰公式

**文件**：`model/covariate.py` `get_beichi()` 函数

```python
# 原始（会爆炸）：
diff = (prev.power - current_bi.power) / (prev.power + 1e-8)

# 修复（tanh 软裁剪到 [-1, +1]）：
raw_diff = prev.power - current_bi.power
diff = np.tanh(raw_diff / (abs(prev.power) + abs(current_bi.power) + 1e-8))
```

修复后重建 CZSC 缓存：
```bash
python scripts/build_czsc_cache.py --data-dir data/tdx_import/1d -n 4
```

修复后验证 D5 分布：若其他维度（D1~D7）在 tanh 修复后均无极端异常值，
则**不引入额外的 per-dim 标准化**，避免对已归一化的维度做双重标准化。

### 步骤 2.2：升级 IIB 架构

**文件**：`model/covariate.py` `InputInjectionBlock` 类

从单层 FFN 升级为 2 层残差 MLP + LayerNorm：

```
emb_proj(832→256) + cov_proj(7→256) → 相加（替代 concat）
  → LayerNorm → GELU(Linear(256→512)) → Dropout(0.3) → Linear(512→256) → 残差
  → LayerNorm → GELU(Linear(256→512)) → Dropout(0.3) → Linear(512→256) → 残差
  → out_proj(256→832) → Dropout(0.3)
```

参数量：~956K（从 560K 增至 0.93%），表达力大幅提升。

构造函数新增 `n_layers` 参数，默认=1（向后兼容旧架构）。

### 步骤 2.3：Kronos.__init__ 接收 IIB 配置参数 + Phase 2 加载逻辑

**文件**：`model/kronos.py`

`Kronos.__init__` 新增可选参数（带向后兼容默认值）：
```python
def __init__(self, ..., iib_cov_dim=7, iib_hidden_dim=256, iib_dropout=None, iib_n_layers=1):
```

- `iib_dropout=None` 时退化为 `self.ffn_dropout_p`（保持旧行为）
- `iib_n_layers=1` 保持旧架构（向后兼容 `from_pretrained`）

**Phase 2 加载流程**（在 `train_predictor_tdx.py` 中）：

```python
# 1. from_pretrained 用旧 config.json 构建模型（n_layers=1 兼容旧权重）
model = Kronos.from_pretrained(config['finetuned_predictor_path'])
# 2. 丢弃旧 IIB，用升级版替换（随机初始化）
model.iib = InputInjectionBlock(
    d_model=832, cov_dim=7, hidden_dim=config['iib_hidden_dim'],
    dropout=config['iib_dropout'], n_layers=config['iib_n_layers'],
).to(device)
# 3. 其余权重正常加载（from_pretrained 已完成），IIB 从零训练
```

Phase 1 训练时 IIB 返回零（past_covariates=None），其权重学到了输出近零残差。
Phase 2 直接丢弃这些权重、随机初始化新 IIB，干净的起点。

### 步骤 2.4：渐进式解冻训练

**文件**：`finetune/train_predictor_tdx.py`

**设计原则**：Phase 2 的目标是注入 CZSC 协变量，不应大幅改变 Phase 1 已学到的表示。
因此 Stage C（全参数）使用极低学习率，仅做微调。

三个阶段：

| 阶段 | Epoch | 可训练参数 | 学习率 |
|------|-------|-----------|--------|
| A | 1-5 | 仅 IIB | 3e-4 |
| B | 6-10 | IIB + 后 4 层 transformer + head | IIB:3e-4, Transformer:1e-5 |
| C | 11-20 | 全参数 | IIB:3e-4, Top:1e-5, Base:5e-6 |

- `apply_freeze_stage(model, stage)` 管理参数冻结
- 阶段切换时重建 optimizer 和 scheduler
- 差异化学习率通过多参数组实现

**文件**：`finetune/config_tdx.py`

新增：
```python
# Phase 2: IIB 训练
self.phase = 'iib'
self.use_iib = True

# IIB 架构（升级）
self.iib_hidden_dim = 256
self.iib_dropout = 0.3          # 从 0.1 提高
self.iib_n_layers = 2           # 从 1 升级
self.iib_learning_rate = 3e-4   # 从 1e-3 降低
self.iib_weight_decay = 0.2     # 强正则化

# 渐进式解冻
self.iib_only_epochs = 5        # Stage A
self.iib_plus_top_epochs = 5    # Stage B
self.transformer_top_lr = 1e-5  # Stage B/C 顶层学习率
self.transformer_base_lr = 5e-6 # Stage C 全参数学习率（极低，保护 Phase 1）
```

### 步骤 2.5：运行 Phase 2 训练

```bash
python finetune/train_predictor_tdx.py --phase iib --data-dir data/tdx_import/1d
```

---

## 评审修订记录（v2）

基于 staff review 的 5 个 P1 问题修正：

| # | 原始方案 | 评审意见 | 修正 |
|---|----------|----------|------|
| 1 | Token 磁盘预缓存 + CachedQlibDataset | 过度工程化，3.4GB 磁盘数据、新脚本、新 Dataset 类 | **删除**。tokenizer.encode() 仅 ~2ms/batch，不是瓶颈。优先解决收敛性 |
| 2 | bf16 + GradScaler | GradScaler 对 bf16 无效（bf16 动态范围足够） | **修正**：bf16 autocast + 无 GradScaler，直接 backward() |
| 3 | IIB 架构升级 + from_pretrained | state_dict 键不匹配导致加载失败 | **修正**：from_pretrained 用旧 config（n_layers=1），加载后手动替换 IIB 为新版（n_layers=2），其余权重正常保留 |
| 4 | CZSC per-dim 标准化 + czsc_stats.pkl | D5 tanh 修复后大部分维度已归一化，额外标准化可能双重归一化 | **删除** stats 管线。D5 修复后验证分布，仅对仍有异常的维度做 clip |
| 5 | Phase 1: 30 epoch 全参数 → Phase 2 再全参数解冻 | Phase 2 Stage C 会覆盖 Phase 1 学到的表示 | **修正**：Phase 1 降至 10 epoch（热身）；Phase 2 Stage C 用极低 LR（5e-6）保护 Phase 1 |

---

## 文件修改清单

| 文件 | 操作 | Phase |
|------|------|-------|
| `finetune/config_tdx.py` | 修改：新增 phase/IIB/解冻参数 | 1+2 |
| `finetune/train_tokenizer_tdx.py` | 修改：添加 bf16 autocast（无 GradScaler） | 1 |
| `finetune/train_predictor_tdx.py` | 修改：phase 控制、IIB 升级加载、渐进解冻 | 1+2 |
| `model/covariate.py` | 修改：D5 tanh 修复、IIB 残差 MLP 升级 | 2 |
| `model/kronos.py` | 修改：__init__ 新增 IIB 配置参数（向后兼容） | 2 |
| `scripts/build_czsc_cache.py` | 仅重建缓存（D5 修复后） | 2 |

**已删除的文件**（评审后简化）：
- ~~`scripts/precompute_token_cache.py`~~（不需要磁盘缓存）
- ~~`finetune/cached_dataset.py`~~（使用原始 QlibDataset）
- ~~`finetune/dataset.py` CZSC stats 归一化~~（D5 修复足够，不做额外标准化）

## 验证方案

### Phase 1 验证
1. Tokenizer 微调：检查 Val MSE Loss 是否收敛
2. Predictor 全参数微调：检查 Val CE Loss 趋势（应稳定下降，对比 IIB-only 的过拟合）
3. 训练时间：bf16 AMP + bs=128 应比之前更快

### Phase 2 验证
1. D5 修复后检查 CZSC 缓存统计：D5 应在 [-1, +1] 范围内
2. IIB Stage A：Val Loss 应下降（学习率 3e-4，dropout 0.3，不过拟合）
3. Stage B：解冻后 4 层应进一步降低 Val Loss
4. Stage C：全参数微调用极低 LR，Val Loss 不应反弹
5. 最终对比：Phase 2 模型 vs Phase 1 模型的 Val Loss 差异
