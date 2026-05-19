# 工作摘要

**时间:** 2026-05-19 22:50
**版本:** 1.5.0

## 变更概要

### 新增功能

- `scripts/predict_stocks.py`：支持读取 TDX 自选股（zxg.blk）批量预测，无参数时自动读取
- 通过 mootdx 批量获取股票名称，替代硬编码映射
- 预测结果按 指数→看涨→看平→看跌 分类排序输出
- 10日涨跌幅分类（看涨>3%、看跌<-3%、看平）
- `pyproject.toml`：项目打包配置

### 缺陷修复

- **数据新鲜度检测**：`MAX_STALE_DAYS` 从 5 降为 0，确保每日收盘后导入最新数据
- **指数数据刷新**：指数代码不再被 `ensure_fresh_data` 跳过，用 `dividend_type="none"` 导入（实际点位）
- **指数复权因子**：`derive_factor` 对指数直接返回 1.0，避免 `Reader.daily` 读到错误标的
- **factor_ok 误判**：指数 factor=1.0 不再报"复权因子获取失败"错误
- **DataFrame 真值歧义**：`fresh_cache.get(code) or get_data(code)` 改为显式 `in` 检查

### 文档更新

- `docs/prediction_data_files.md`：表格格式修正
- `model/__init__.py`：新增 `__version__` 版本号
