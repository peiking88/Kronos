# 工作摘要

**时间:** 2026-05-18 19:55

## 变更概要

修复预测管线中后复权因子不一致导致预测价格严重偏离的问题。

### 问题
- `scripts/predict.py` 推理时输入不复权原始价格，但模型用后复权训练，输入空间不匹配
- `examples/prediction_cn_markets_day.py` 使用 akshare 不复权数据，同样不匹配
- `scripts/predict_stocks.py` 的 `load_factor()` 从 factor cache 读取因子，但 cache 可能被重新计算（kline 数据变化导致），与训练数据 pkl 中的隐含因子不一致（sh600353: cache=18.60 vs 隐含=10.71，预测偏低 42%）
- 偏差校正值在 hfq 空间计算但以实际价格单位显示，数值无意义

### 修复
1. **predict.py**: `import_from_tdx()` 应用后复权调整，预测结果除以 factor 转回实际价
2. **prediction_cn_markets_day.py**: 改用 `adjust="hfq"` 获取后复权数据，推导因子并转换
3. **predict_stocks.py**: `load_factor()` → `derive_factor(code, df_hfq)`，从数据本身推导因子（hfq/raw 比值），保证与 pkl 数据一致
4. 三个脚本的偏差校正值均除以 factor 转换到实际价格空间
5. 更新 finetune-kronos 技能：新增"复权因子缓存漂移"知识点和排查方法

### 未修改
- `tdxdata/` / `mootdx` 依赖模块：`compute_factor_from_xdxr` 逻辑正确，问题在于不同时间的 kline 数据差异
- 根本修复需用正确 factor 重新导入数据 + 重新微调（当前无 GPU，预估需 3.5-5 天 CPU-only）

## 最近提交
```
825c50e feat: 纠偏逻辑集成到全部预测入口，合并 predict_sse.py，新增共享校准模块
3c2d9f5 feat: 预测脚本自动检查数据新鲜度，过期则从TDX导入最新行情
7fd826e feat: 新增回测校准功能，预测自动修正系统性偏差
584c13a feat: 新增一键预测脚本、README用法说明、数据修复重训
f8fc069 chore: 排除数据文件和模型文件，更新 .gitignore
```
