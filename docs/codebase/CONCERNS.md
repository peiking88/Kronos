# Codebase Concerns

## Core Sections (Required)

### 1) Top Risks (Prioritized)

| Severity | Concern | Evidence | Impact | Suggested action |
|----------|---------|----------|--------|------------------|
| high | No automated test coverage measurement | `tests/` contains only 2 test functions | Regressions undetected in CI | Install pytest-cov, enforce >80% threshold |
| high | Model revision pinning blocks model updates | `tests/test_kronos_regression.py:30-31` — hardcoded git hashes for model and tokenizer | Cannot upgrade model without breaking tests | Move to config or use `latest` tag |
| medium | No input validation for price data integrity | KronosPredictor only checks NaN/columns, not price logic (high<low, positive prices, chronological order) | Garbage-in-garbage-out predictions | Add sanity checks: open/high/low/close ordering, positive values, monotonic timestamps |
| medium | Hardcoded HuggingFace model IDs and paths | `examples/`, `webui/`, `finetune/` all hardcode `"NeoQuasar/Kronos-base"` etc. | Refactoring model names touches many files | Centralize model registry |
| medium | No structured logging | All logging uses `print()` with emoji | No log aggregation, filtering, or alerting | Use Python `logging` module |
| low | Comet ML API key placeholder in config | `finetune/config.py:79` — `"YOUR_COMET_API_KEY"` | Accidental commit of real key | Move to env var immediately |

### 2) Technical Debt

| Debt item | Why it exists | Where | Risk if ignored | Suggested fix |
|-----------|---------------|-------|-----------------|---------------|
| `sys.path.append("../")` in examples | Library not installed as package | All example files | Import fragility, IDE confusion | Create `setup.py`/`pyproject.toml`, install in editable mode |
| pandas 2.2.2 pin upgraded to 3.0.2 | Python 3.14 incompatibility with old pandas | `requirements.txt:8` | API breakage from pandas 3.x changes | [ASK USER] verify pandas 3.0.2 API compatibility, add regression tests |
| Duplicated data fetching logic | Each example script implements its own data loading | `examples/get_akshare_date_*`, `examples/prediction_cn_markets_day.py`, `examples/prediction_akshare_2024-2025.py` | Inconsistent behavior, maintenance burden | Extract to shared `data/` module |
| Large file committed (not in git) | `finetune_csv/data/HK_ali_09988_kline_5min_all.csv` is 5.7MB | `finetune_csv/data/` | Bloats repo; may contain proprietary data | Verify git tracking, add to .gitignore if appropriate |
| AI-generated comments in finetune/ | README warns comments may be inaccurate | `finetune/` directory per `README.md:308` | Developer confusion from wrong comments | Review and correct or remove |

### 3) Security Concerns

| Risk | OWASP category (if applicable) | Evidence | Current mitigation | Gap |
|------|--------------------------------|----------|--------------------|-----|
| Unvalidated file paths in WebUI | A01:2021 — Broken Access Control | `webui/app.py:60-76` reads files from `../data/` by name | Path constructed relative to script | No path traversal protection |
| HTTP used for EastMoney API | A02:2021 — Cryptographic Failures | `examples/get_akshare_date_2024-2025_x.py:46` — `http://push2his.eastmoney.com` | None | Use HTTPS endpoint if available |
| Unsafe model deserialization | A08:2021 — Software and Data Integrity Failures | `torch.load()` in HuggingFace `from_pretrained()` | `safetensors` format used for weights | HuggingFace Hub trust model — no signature verification |
| No authentication on WebUI | A01:2021 — Broken Access Control | `webui/app.py:24` — Flask app with CORS enabled, no auth | None | [ASK USER] is WebUI intended for local-only use? |

### 4) Performance and Scaling Concerns

| Concern | Evidence | Current symptom | Scaling risk | Suggested improvement |
|---------|----------|-----------------|-------------|-----------------------|
| Autoregressive inference is O(pred_len) serial | `model/kronos.py:420-469` — sequential for-loop over pred_len | Slow for long predictions (>120 steps) | Cannot scale to long-horizon forecasting | Investigate speculative decoding or non-autoregressive variants |
| Z-score normalization per-sample on CPU | `model/kronos.py:544-547` — numpy mean/std per call | Overhead for high-frequency batch prediction | Bottleneck for real-time streaming | Pre-compute normalization statistics, use GPU |
| Ring buffer on GPU for each sample | `model/kronos.py:408-454` | Memory allocation per inference call | GPU memory fragmentation | Pre-allocate buffer pool |
| No batching across different-length series | `predict_batch` requires identical seq_len + pred_len | Cannot mix different timeframes in one batch | Under-utilized GPU for heterogeneous queries | Padding + attention masking for variable lengths |

### 5) Fragile/High-Churn Areas

| Area | Why fragile | Churn signal | Safe change strategy |
|------|-------------|-------------|----------------------|
| `examples/` | Hardcoded absolute paths (Windows `D:\lianghuajiaoyi\...`), environment-specific settings | Most files in repo | Run examples as integration tests, parameterize paths |
| `finetune/config.py` | Centralized config with many interdependent derived paths | 4 TODO items | Split into base config + env-specific overrides |
| `model/module.py` (571 lines) | Multiple responsibilities: quantizer, attention, embeddings, temporal encoding | Largest Python file | Extract BSQuantizer, attention, embeddings into submodules |

### 6) `[ASK USER]` Questions

1. [ASK USER] 复权方式：当前多个数据源使用不同的复权策略 — 东方财富API使用前复权(fqt=1)，akshare使用不复权(adjust="")，Qlib数据取决于本地数据准备。是否需要统一所有数据源为同一种复权方式？如果需要，是前复权、后复权还是不复权？
2. [ASK USER] 数据周期：预训练模型覆盖哪些具体周期（5min、日线、周线、月线）？文档仅提及"45+ global exchanges"，但没有明确列出支持的时间频率。
3. [ASK USER] 市场覆盖范围：README 声称"45 global exchanges"，但示例代码仅覆盖 A股（沪深）和港股（阿里）。是否需要记录完整的交易所列表和对应的数据格式？
4. [ASK USER] WebUI 安全：WebUI 是否仅用于本地开发？如果是，当前的 CORS+无认证设计可接受；如果不是，需要添加认证和路径遍历保护。
5. [ASK USER] pandas 3.0.2 兼容性：requirements.txt 中 pandas 版本从 2.2.2 升级到 3.0.2（因 Python 3.14 不兼容）。是否有 pandas 2.x 特有的 API 依赖需要检查？

### 7) Evidence

- `.codebase-scan.txt` — 4 TODOs in production code, 0 lint configs, 0 CI/CD
- `model/kronos.py:519-559` — predict() validation logic
- `tests/test_kronos_regression.py:30-31` — pinned model revisions
- `examples/prediction_cn_markets_day.py:57` — akshare adjust="" (不复权)
- `examples/get_akshare_date_2024-2025_x.py:53` — EastMoney fqt=1 (前复权)
- `finetune/config.py:79` — Comet ML API key placeholder
- `webui/app.py:24-25` — Flask CORS, no auth
