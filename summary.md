# 工作摘要

**时间:** 2026-05-18 15:30

## 本次变更

### 纠偏逻辑集成到所有预测入口
- 新建 `scripts/calibrate.py` 共享纠偏模块（从 predict.py 抽取，改为接受 predictor 对象）
- `scripts/predict.py` — 模型加载优化为一次，使用共享校准模块
- `scripts/predict_stocks.py` — hfq 空间校准后换算实际市价，报告中显示偏差信息
- `webui/app.py` — /api/predict 端点数据充足时自动校准
- `examples/prediction_cn_markets_day.py` — 预测→涨跌停→校准→重新涨跌停

### 合并 predict_sse.py → predict_stocks.py
- 删除 `scripts/predict_sse.py`（129 行）
- `scripts/predict_stocks.py` 新增 `--format console` 参数（控制台表格输出）
- 报告文件路径规范为 `output/kronos_{symbols}.md`

### 文件变更
- 新增: `scripts/calibrate.py`
- 修改: `scripts/predict.py`, `scripts/predict_stocks.py`, `webui/app.py`, `examples/prediction_cn_markets_day.py`
- 删除: `scripts/predict_sse.py`
- 更新: `README.md`, `.claude/skills/finetune-kronos/SKILL.md`

## 最近提交
```
3c2d9f5 feat: 预测脚本自动检查数据新鲜度，过期则从TDX导入最新行情
7fd826e feat: 新增回测校准功能，预测自动修正系统性偏差
584c13a feat: 新增一键预测脚本、README用法说明、数据修复重训
f8fc069 chore: 排除数据文件和模型文件，更新 .gitignore
a46b647 feat: 依赖升级、后复权修复与TDX数据微调完成
```
