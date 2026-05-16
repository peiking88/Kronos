# 工作摘要

**时间:** 2026-05-05 09:35:17

## 变更概要
```
 .gitignore                                   |    3 +
 docs/codebase/ARCHITECTURE.md                |   79 ++
 docs/codebase/CONCERNS.md                    |   68 ++
 docs/codebase/CONVENTIONS.md                 |   50 ++
 docs/codebase/INTEGRATIONS.md                |   48 ++
 docs/codebase/STACK.md                       |   74 ++
 docs/codebase/STRUCTURE.md                   |   55 ++
 docs/codebase/TESTING.md                     |   60 ++
 docs/codebase/WORKFLOWS.md                   |  214 ++++++
 requirements.txt                             |    2 +-
 summary.md                                   |  183 +----
 tdxdata/.gitignore                           |   14 +
 tdxdata/.trae/rules/project_rules.md         |   59 ++
 tdxdata/README.md                            |  367 ++++++++++
 tdxdata/docs/PRD.md                          |  209 ++++++
 tdxdata/docs/api_reference.md                | 1007 ++++++++++++++++++++++++++
 tdxdata/docs/dataspec.txt                    |  264 +++++++
 tdxdata/docs/tdxdata.txt                     |  245 +++++++
 tdxdata/plans/tdx-data-reader.md             |  415 +++++++++++
 tdxdata/pyproject.toml                       |   31 +
 tdxdata/tdxdata/__init__.py                  |    4 +
 tdxdata/tdxdata/api.py                       |  206 ++++++
 tdxdata/tdxdata/core/__init__.py             |   11 +
 tdxdata/tdxdata/core/connection.py           |   58 ++
 tdxdata/tdxdata/core/data_manager.py         |   79 ++
 tdxdata/tdxdata/core/plugin_manager.py       |   92 +++
 tdxdata/tdxdata/core/registry.py             |   70 ++
 tdxdata/tdxdata/errors/__init__.py           |   19 +
 tdxdata/tdxdata/errors/circuit_breaker.py    |   82 +++
 tdxdata/tdxdata/errors/exceptions.py         |   26 +
 tdxdata/tdxdata/errors/resource.py           |   20 +
 tdxdata/tdxdata/errors/retry.py              |   48 ++
 tdxdata/tdxdata/logging/__init__.py          |    3 +
 tdxdata/tdxdata/logging/logger.py            |   16 +
 tdxdata/tdxdata/qlib/__init__.py             |   19 +
 tdxdata/tdxdata/qlib/converter.py            |  104 +++
 tdxdata/tdxdata/qlib/qlib_bin.py             |  117 +++
 tdxdata/tdxdata/sources/__init__.py          |   19 +
 tdxdata/tdxdata/sources/adjust.py            |   80 ++
 tdxdata/tdxdata/sources/base.py              |   21 +
 tdxdata/tdxdata/sources/daily_basic.py       |   36 +
 tdxdata/tdxdata/sources/f10.py               |   60 ++
 tdxdata/tdxdata/sources/financial.py         |   40 +
 tdxdata/tdxdata/sources/history_kline.py     |  108 +++
 tdxdata/tdxdata/sources/hybrid_kline.py      |  253 +++++++
 tdxdata/tdxdata/sources/local_kline.py       |  123 ++++
 tdxdata/tdxdata/sources/realtime_snapshot.py |   87 +++
 tdxdata/tdxdata/sources/tick.py              |   51 ++
 tdxdata/tdxdata/storage/__init__.py          |   13 +
 tdxdata/tdxdata/storage/base.py              |   17 +
 tdxdata/tdxdata/storage/csv.py               |   44 ++
 tdxdata/tdxdata/storage/dataframe.py         |   13 +
 tdxdata/tdxdata/storage/parquet.py           |   44 ++
 tdxdata/tdxdata/storage/qlib.py              |   75 ++
 tdxdata/tdxdata/storage/sqlite.py            |   48 ++
 tdxdata/tdxdata/sync/__init__.py             |    5 +
 tdxdata/tdxdata/sync/gap_detector.py         |   25 +
 tdxdata/tdxdata/sync/manager.py              |   29 +
 tdxdata/tdxdata/sync/state.py                |   60 ++
 tdxdata/tests/__init__.py                    |    0
 tdxdata/tests/conftest.py                    |   41 ++
 tdxdata/tests/test_circuit_breaker.py        |   67 ++
 tdxdata/tests/test_connection.py             |   92 +++
 tdxdata/tests/test_history_kline.py          |  151 ++++
 tdxdata/tests/test_hybrid_kline.py           |  221 ++++++
 tdxdata/tests/test_integration.py            |  219 ++++++
 tdxdata/tests/test_live.py                   |  653 +++++++++++++++++
 tdxdata/tests/test_live_local.py             |  175 +++++
 tdxdata/tests/test_local_kline.py            |  115 +++
 tdxdata/tests/test_qlib.py                   |  244 +++++++
 tdxdata/tests/test_registry.py               |   76 ++
 tdxdata/tests/test_retry.py                  |   51 ++
 tdxdata/tests/test_sources.py                |  216 ++++++
 tdxdata/tests/test_storage.py                |  110 +++
 tdxdata/tests/test_sync.py                   |  105 +++
 75 files changed, 8031 insertions(+), 177 deletions(-)
```

## 最近提交
```
2c6d22b Add TDX local data fine-tuning pipeline for Kronos
67b630e Merge pull request #243 from ElhamDevelopmentStudio/fix/batch-dimension-training
```
