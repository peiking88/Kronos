# Design Document: TDX Data Reader - 通达信行情数据读取接口

> Source PRD: [docs/PRD.md](../docs/PRD.md) | GitHub Issue: [peiking88/tdxdata#1](https://github.com/peiking88/tdxdata/issues/1)
>
> Review baseline: [docs/dataspec.txt](../docs/dataspec.txt) — 数据格式参考规范

## Design Review Log

| Date | Reviewer | Decision |
|------|----------|----------|
| 2026-04-05 | Design Review vs dataspec.txt | Column naming keeps existing design (`stock_code`, `date`); missing fields from dataspec added where applicable; K-line keeps OHLCV core only (no derived fields); fundamental data separated as independent static data module; file layout stays flat structure |

## Architectural Decisions

Durable decisions that apply across all phases:

### Data Format Standards

- **Stock code format**: `CODE.MARKET` (e.g., `600519.SH`, `000001.SZ`)
- **Stock code column name**: `stock_code` (NOT `symbol` from dataspec — keeps consistency with existing design)
- **Date column name**: `date` (NOT `trade_date` from dataspec — keeps consistency)
- **Column naming**: English lowercase snake_case (open, high, low, close, volume, amount, etc.)
- **Date/time type**: `datetime64`
- **Numeric types**: `float64` / `int64`, missing values as `NaN`
- **Default dividend adjustment**: Front-adjusted (`front`)

### API Design

Two levels of API:

- **Unified entry**: `TdxData.fetch(source, ...)` — generic plugin-driven interface
- **Convenience methods**:
  - `fetch_history()` — fetch K-line data (daily + 1-min from tqcenter; 5/10/15/30/60-min aggregated from 1-min; weekly/monthly aggregated from daily). Format unified with daily K-line schema.
  - `fetch_realtime()` — fetch single-stock realtime snapshot including depth quotes (bid/ask 5-level data, full market depth)
  - `fetch_f10()` — fetch structured F10 data
  - `fetch_tick()` — fetch tick-by-tick trade data
  - `fetch_basic()` — fetch static fundamental data (total_share, float_share, pe, pb, total_mv, circ_mv, etc.), stored independently, assembled on-demand during analysis

> **NOTE**: The exact tqcenter API surface (which methods are available, their parameters and return formats) will be confirmed after downloading the plugin and its documentation. The interfaces below are designed based on the reference material in `docs/tdxdata.txt` and may need adjustment.

### K-line Aggregation Rules

| Period | Source | Method |
|-------|--------|--------|
| 1d (daily) | tqcenter direct | `tq.get_market_data(period='1d')` |
| 1m (1-minute) | tqcenter direct | `tq.get_market_data(period='1m')` |
| 5m | 1m data | Resample via `df.resample('5min').agg(...)` |
| 10m | 1m data | Resample via `df.resample('10min').agg(...)` |
| 15m | 1m data | Resample via `df.resample('15min').agg(...)` |
| 30m | 1m data | Resample via `df.resample('30min').agg(...)` |
| 60m | 1m data | Resample via `df.resample('60min').agg(...)` |
| 1w (weekly) | daily data | Resample via `df.resample('W').agg(...)` |
| 1mon (monthly) | daily data | Resample via `df.resample('ME').agg(...)` |

All K-line periods produce the **same DataFrame schema** as daily K-line (see below).

### Aggregation OHLCV Rules

```python
agg_rules = {
    'open': 'first',
    'high': 'max',
    'low': 'min',
    'close': 'last',
    'volume': 'sum',
    'amount': 'sum'
}
```

### K-line DataFrame Schema (All Periods)

Core OHLCV fields only. No derived fields (pre_close, change, pct_chg, etc.) — these can be calculated by the user from the core data.

| Column | Type | Description |
|--------|------|-------------|
| stock_code | str | Stock code (e.g., 600519.SH) |
| date | datetime64 | Trading date (for intraday: datetime with time component) |
| open | float64 | Opening price |
| high | float64 | Highest price |
| low | float64 | Lowest price |
| close | float64 | Closing price |
| volume | float64 | Trading volume (shares) |
| amount | float64 | Trading amount (yuan) |

### Realtime Snapshot DataFrame Schema

Full depth quote data including 5-level bid/ask. Column naming follows existing design conventions.

| Column | Type | Description |
|--------|------|-------------|
| stock_code | str | Stock code |
| name | str | Stock name |
| datetime | datetime64 | Snapshot time |
| open | float64 | Today's open |
| high | float64 | Today's high |
| low | float64 | Today's low |
| close | float64 | Current price (last trade price) |
| last_close | float64 | Yesterday's close |
| volume | float64 | Total volume (shares) |
| amount | float64 | Total amount (yuan) |
| bid_price1 ~ bid_price5 | float64 | Bid prices (5 levels) |
| bid_volume1 ~ bid_volume5 | float64 | Bid volumes (5 levels, shares) |
| ask_price1 ~ ask_price5 | float64 | Ask prices (5 levels) |
| ask_volume1 ~ ask_volume5 | float64 | Ask volumes (5 levels, shares) |
| turnover_rate | float64 | Turnover rate (%) |

### Fundamental Data Schema (Static, Independent Storage)

Fundamental / daily basic data is stored separately from realtime snapshots. Fetched via `fetch_basic()` and stored in its own table/file. Users assemble this data with market data on-demand during analysis.

| Column | Type | Description |
|--------|------|-------------|
| stock_code | str | Stock code |
| date | datetime64 | Trading date |
| close | float64 | Closing price |
| turnover_rate | float64 | Turnover rate (%) |
| volume_ratio | float64 | Volume ratio |
| pe | float64 | P/E ratio |
| pb | float64 | P/B ratio |
| ps_ttm | float64 | P/S ratio (TTM) |
| dv_ratio | float64 | Dividend yield (%) |
| total_share | float64 | Total shares (10k shares) |
| float_share | float64 | Float shares (10k shares) |
| total_mv | float64 | Total market value (10k yuan) |
| circ_mv | float64 | Circulating market value (10k yuan) |

### Tick DataFrame Schema

| Column | Type | Description |
|--------|------|-------------|
| stock_code | str | Stock code |
| datetime | datetime64 | Tick time |
| price | float64 | Trade price |
| volume | float64 | Trade volume (shares) |
| amount | float64 | Trade amount (yuan) |
| buy_sell_flag | str | Buy/sell direction (B/S/-) |

### Storage Layout

Flat structure, one file per stock per data type:

- **Sync state**: `~/.tdxdata/sync_state.json`
- **SQLite schema**: One table per stock code per data type (e.g., `kline_1d_600519_SH`, `snapshot_600519_SH`, `basic_600519_SH`)
- **CSV layout**: `{output_path}/{data_type}/{stock_code}.csv`
- **Parquet layout**: `{output_path}/{data_type}/{stock_code}.parquet`

### Plugin Registration

- Data sources register via `@register_source("name")` decorator
- Storage providers register via `@register_storage("name")` decorator
- Registry maintains `dict[str, type]` mappings in a central `PluginRegistry`

### tqcenter API Surface (TBD)

The following tqcenter methods are referenced in the design. Actual availability will be confirmed after plugin download:

| Method | Purpose | Status |
|--------|---------|--------|
| `tq.initialize(__file__)` | Initialize connection | Referenced in docs/tdxdata.txt |
| `tq.close()` | Close connection | Referenced in docs/tdxdata.txt |
| `tq.refresh_cache()` | Refresh market data cache | Referenced in docs/tdxdata.txt |
| `tq.get_market_data(...)` | Fetch historical K-line | Referenced in docs/tdxdata.txt |
| `tq.get_market_snapshot(...)` | Fetch realtime snapshot | Referenced in docs/tdxdata.txt |
| F10 / Financial API | Fetch F10 and financial data | **To be confirmed** |
| Tick data API | Fetch tick-by-tick data | **To be confirmed** |
| Basic fundamental API | Fetch daily basic indicators | **To be confirmed** |

### Package Structure

```
tdxdata/
├── __init__.py
├── api.py                    # TdxData main class (public API)
├── core/
│   ├── __init__.py
│   ├── connection.py         # TdxConnection
│   ├── plugin_manager.py     # PluginManager
│   ├── data_manager.py       # DataManager
│   └── registry.py           # PluginRegistry (decorator-based)
├── sources/
│   ├── __init__.py
│   ├── base.py               # DataSourceBase (ABC)
│   ├── history_kline.py      # HistoryKlineSource
│   ├── realtime_snapshot.py  # RealtimeSnapshotSource
│   ├── tick.py               # TickDataSource
│   ├── financial.py          # FinancialDataSource
│   ├── f10.py                # F10DataSource
│   └── daily_basic.py        # DailyBasicSource (static fundamental data)
├── storage/
│   ├── __init__.py
│   ├── base.py               # StorageBase (ABC)
│   ├── dataframe.py          # DataFrameStorage
│   ├── csv.py                # CSVStorage
│   ├── sqlite.py             # SQLiteStorage
│   └── parquet.py            # ParquetStorage
├── sync/
│   ├── __init__.py
│   ├── manager.py            # SyncManager
│   ├── state.py              # SyncState
│   └── gap_detector.py       # GapDetector
├── errors/
│   ├── __init__.py
│   ├── exceptions.py         # Custom exception hierarchy
│   ├── retry.py              # RetryPolicy
│   ├── circuit_breaker.py    # CircuitBreaker
│   └── resource.py           # ResourceManager (context manager)
└── logging/
    ├── __init__.py
    └── logger.py             # Structured logger
tests/
├── __init__.py
├── conftest.py               # Shared fixtures (mock tqcenter)
├── test_connection.py
├── test_plugin_manager.py
├── test_registry.py
├── test_history_kline.py
├── test_realtime_snapshot.py
├── test_tick.py
├── test_financial.py
├── test_f10.py
├── test_daily_basic.py
├── test_storage_csv.py
├── test_storage_sqlite.py
├── test_storage_parquet.py
├── test_sync_manager.py
├── test_sync_state.py
├── test_gap_detector.py
├── test_retry.py
├── test_circuit_breaker.py
└── test_integration.py
```

---

## Phase 1: Project Skeleton + Plugin Discovery

**User stories**: #26, #27, #28, #36

### What to build

Set up the Python package structure, plugin registry mechanism, and PluginManager that auto-discovers the local TDX installation and tqcenter plugin. The PluginManager searches common Windows installation paths, locates `PYPlugins/user`, and configures `sys.path` accordingly. If tqcenter is not found locally, it downloads from the official source. This phase also downloads and catalogs the tqcenter plugin documentation to confirm the actual API surface (which methods are available for F10, tick, financial, and basic data).

### Acceptance criteria

- [ ] `tdxdata` package is installable via `pip install -e .` with correct dependencies (pandas, numpy, pyarrow, etc.)
- [ ] `PluginRegistry` supports `@register_source()` and `@register_storage()` decorators
- [ ] `PluginManager` discovers TDX installation path on Windows (searches common paths: `D:\new_tdx`, `C:\new_tdx`, etc.)
- [ ] `PluginManager` configures `sys.path` to include `PYPlugins/user`
- [ ] `PluginManager` downloads tqcenter plugin when not found locally
- [ ] tqcenter plugin documentation is cataloged — all available API methods identified and documented in the codebase
- [ ] Unit tests for registry and plugin discovery with mocked file system
- [ ] `pyproject.toml` with project metadata and dependencies declared

---

## Phase 2: Core Connection Layer + DataFrame Output

**User stories**: #1, #4, #5, #17, #21, #29, #33, #34

### What to build

Implement the TdxConnection lifecycle manager (initialize/close), ResourceManager context manager (guarantees `tq.close()` on exception), basic error handling framework (custom exception hierarchy, RetryPolicy with exponential backoff), and DataFrameStorage as the default output. The main `TdxData` class provides the `fetch()` unified entry point and context manager protocol (`with TdxData() as tdx`). This phase creates a complete vertical slice: connect → fetch raw data → convert to DataFrame → return. Uses a simple HistoryKlineSource stub to validate the end-to-end path with daily K-line data.

### Acceptance criteria

- [ ] `TdxConnection` wraps `tq.initialize()` and `tq.close()` with proper lifecycle management
- [ ] `ResourceManager` context manager guarantees `tq.close()` is called even on exceptions
- [ ] Custom exception hierarchy: `TdxDataError` → `ConnectionError`, `DataFetchError`, `StorageError`, `PluginNotFoundError`
- [ ] `RetryPolicy` with configurable max_retries, base_delay, and exponential backoff
- [ ] `DataFrameStorage` returns pandas DataFrame with standardized column names (`stock_code`, `date`, OHLCV)
- [ ] `TdxData` class supports `with TdxData() as tdx:` context manager
- [ ] `TdxData.fetch(source="history_kline", stock_list=["600519.SH"], period="1d")` returns a DataFrame with daily K-line data (front-adjusted by default)
- [ ] Raw tqcenter data (dict-of-Series) correctly converted to standardized DataFrame with `stock_code` and `date` columns
- [ ] Unit tests for connection lifecycle, resource cleanup, retry policy, and DataFrame output with mocked tqcenter
- [ ] Structured logging records connection events, fetch operations, and errors

---

## Phase 3: History K-line Data Source

**User stories**: #1, #2, #3, #4, #5, #6

### What to build

Implement the full HistoryKlineSource with all K-line periods. The source directly fetches daily and 1-minute data from tqcenter, then aggregates into 5/10/15/30/60-minute periods from 1-minute data, and weekly/monthly periods from daily data using the standard OHLCV aggregation rules. Supports front-adjust (default), back-adjust, and no-adjust dividend types. Handles batch stock fetching. The `fetch_history()` convenience method provides a streamlined API. All periods produce the same unified K-line DataFrame schema.

### Acceptance criteria

- [ ] `fetch_history(stock_list, start_date, end_date, period="1d", dividend_type="front")` works for daily K-line
- [ ] 1-minute K-line fetched directly from tqcenter via `tq.get_market_data(period='1m')`
- [ ] 5/10/15/30/60-minute K-lines generated by resampling 1-minute data with correct OHLCV aggregation
- [ ] Weekly K-line generated by resampling daily data (`W` frequency)
- [ ] Monthly K-line generated by resampling daily data (`ME` frequency)
- [ ] All periods produce the same DataFrame schema (stock_code, date, open, high, low, close, volume, amount)
- [ ] Front-adjust (`front`) is the default; `back` and `none` also supported
- [ ] Batch fetching for multiple stocks (e.g., `["600519.SH", "000001.SZ"]`)
- [ ] Raw tqcenter data (dict-of-Series) correctly converted to standardized DataFrame
- [ ] Empty data and missing fields handled gracefully with NaN
- [ ] Unit tests for all period types with mocked tqcenter data
- [ ] Unit tests for aggregation correctness (verify OHLCV calculation)

---

## Phase 4: Multi-format Storage

**User stories**: #18, #19, #20, #21

### What to build

Implement CSV, SQLite, and Parquet storage plugins. Each plugin implements the `StorageBase` interface (`save()` / `load()`). The unified `output` parameter in `fetch()` routes data to the appropriate storage plugin. Flat file layout: one file per stock per data type. SQLite stores data in tables named `{data_type}_{stock_code}` (dots replaced with underscores). All storage plugins preserve the standardized DataFrame schema and data types.

### Acceptance criteria

- [ ] `CSVStorage.save()` writes DataFrame to CSV with utf-8-sig encoding, flat layout: `{output_path}/{data_type}/{stock_code}.csv`
- [ ] `CSVStorage.load()` reads CSV back into DataFrame with correct types (datetime64, float64)
- [ ] `SQLiteStorage.save()` creates/updates SQLite database with one table per stock per data type
- [ ] `SQLiteStorage.load()` queries SQLite and returns DataFrame
- [ ] `ParquetStorage.save()` writes DataFrame to Parquet with snappy compression
- [ ] `ParquetStorage.load()` reads Parquet back into DataFrame
- [ ] `TdxData.fetch(output="csv"|"sqlite"|"parquet", output_path="...")` routes to correct storage plugin
- [ ] `TdxData.fetch(output="dataframe")` remains the default, returning in-memory DataFrame
- [ ] File/directory creation is automatic (no manual directory setup required)
- [ ] Unit tests for each storage plugin with round-trip verification (save → load → compare)

---

## Phase 5: Incremental Update Mechanism

**User stories**: #22, #23, #24, #25, #32

### What to build

Implement SyncManager, SyncState, and GapDetector. SyncState persists to `~/.tdxdata/sync_state.json`, recording the last sync timestamp per stock per data type. When `incremental=True` is passed to `fetch()`, SyncManager reads the last sync time, calculates the new time range, and only fetches new data. GapDetector scans existing stored data for date gaps and generates fetch requests to fill them. Supports forced full refresh via `incremental=False`. For batch operations with large stock lists, supports resume from interruption point.

### Acceptance criteria

- [ ] `SyncState` persists to `~/.tdxdata/sync_state.json` with structure: `{stock_code: {data_type: {last_sync, date_range}}}`
- [ ] `SyncManager` reads last sync time and calculates incremental fetch range: `last_sync + 1` to `now`
- [ ] `fetch(incremental=True)` only fetches new data since last sync
- [ ] `fetch(incremental=False)` forces full re-fetch
- [ ] `GapDetector` identifies missing trading days in stored data
- [ ] Gap filling generates targeted fetch requests for missing periods only
- [ ] Incremental data is correctly appended/merged with existing data (no duplicates)
- [ ] Resume support: tracks progress in batch operations, can restart from interruption point
- [ ] Unit tests for sync state persistence, incremental range calculation, gap detection, and merge logic

---

## Phase 6: Realtime Market Data Source

**User stories**: #7, #8, #9, #10

### What to build

Implement RealtimeSnapshotSource and TickDataSource. RealtimeSnapshotSource fetches single-stock and batch realtime snapshots including full depth quote data (5-level bid/ask prices and volumes, turnover rate, etc.) via `tq.get_market_snapshot()`. The `fetch_realtime()` convenience method provides a streamlined API. TickDataSource fetches tick-by-tick trade records. All realtime data follows the unified snapshot/tick DataFrame schemas. Fundamental data (pe, pb, market value, shares) is NOT included in snapshots — it is available separately via `fetch_basic()`.

### Acceptance criteria

- [ ] `fetch_realtime(stock_code="600519.SH")` returns single-stock snapshot as DataFrame with depth quote fields
- [ ] `fetch_realtime(stock_list=["600519.SH", "000001.SZ"])` returns batch snapshots
- [ ] Snapshot DataFrame includes: stock_code, name, datetime, OHLCV, last_close, bid/ask 5-level prices and volumes, turnover_rate
- [ ] Snapshot DataFrame does NOT include fundamental data (pe, pb, total_share, etc.) — these are in `fetch_basic()`
- [ ] `fetch_tick(stock_code, date)` returns tick-by-tick trade records for specified date
- [ ] Tick DataFrame includes: stock_code, datetime, price, volume, amount, buy_sell_flag
- [ ] Invalid stock codes return clear error messages (not silent failures)
- [ ] Realtime data compatible with all storage plugins (CSV, SQLite, Parquet, DataFrame)
- [ ] Unit tests with mocked `tq.get_market_snapshot()` and tick data responses

---

## Phase 7: Financial, F10 & Fundamental Data Source

**User stories**: #11, #12, #13, #14, #15, #16

### What to build

Implement FinancialDataSource for the three major financial statements (balance sheet, income statement, cash flow statement), F10DataSource for comprehensive structured F10 data (company profile, shareholder info, dividend history, and all other F10 sections), and DailyBasicSource for static fundamental indicators (pe, pb, total_share, float_share, total_mv, circ_mv, etc.). The `fetch_basic()` convenience method retrieves fundamental data that is stored independently and assembled with market data on-demand during analysis. All data source implementations depend on the actual tqcenter API surface confirmed in Phase 1.

### Acceptance criteria

- [ ] `fetch_financial(stock_code, statement="balance_sheet"|"income"|"cashflow", report_date)` returns financial statement as DataFrame
- [ ] Financial statement DataFrame has standardized columns (report_date, item_name, value, etc.)
- [ ] `fetch_f10(stock_code, sections=["all"])` returns dict of DataFrames for all F10 sections
- [ ] `fetch_f10(stock_code, sections=["shareholder", "finance", "dividend", "profile"])` returns only requested sections
- [ ] F10 sections: company_profile, top10_shareholders, float_shareholders, shareholder_changes, balance_sheet, income_statement, cashflow_statement, dividend_history, and all other available sections
- [ ] Each F10 section's DataFrame has appropriate standardized columns for that section
- [ ] `fetch_basic(stock_code, date)` returns daily fundamental indicators as DataFrame (pe, pb, total_share, float_share, total_mv, circ_mv, turnover_rate, volume_ratio, etc.)
- [ ] Fundamental data is stored independently from realtime snapshots and K-line data
- [ ] Handles version differences in F10 section structure gracefully (missing sections reported, not crashed)
- [ ] Actual API methods used depend on tqcenter plugin documentation confirmed in Phase 1
- [ ] Unit tests with mocked tqcenter F10/financial/basic data responses

---

## Phase 8: Testing Hardening + Documentation

**User stories**: #30, #31, #34, #35, #36

### What to build

Complete the test suite with comprehensive coverage across all modules. Create integration tests that exercise the full pipeline (connect → fetch → store) with mocked tqcenter. Add edge case tests (empty data, missing fields, network timeouts, concurrent access). Generate API documentation. Add usage examples for common workflows.

### Acceptance criteria

- [ ] All DataSource plugins tested with: normal data, empty data, missing fields, malformed data
- [ ] All Storage plugins tested with: round-trip verification, concurrent access, large datasets
- [ ] SyncManager tested with: state persistence, gap detection, resume after interruption
- [ ] Error handling tested with: retry exhaustion, circuit breaker activation, resource cleanup on crash
- [ ] Integration test: end-to-end flow from connection to data output with all storage formats
- [ ] Test coverage ≥ 90% for core modules
- [ ] All tests pass without requiring a running TDX instance (fully mocked)
- [ ] API documentation covers all public methods with parameter descriptions and return types
- [ ] Usage examples in README: daily K-line fetch, realtime snapshot, F10 data, incremental update, multi-format output
- [ ] CircuitBreaker unit tests verify state transitions (closed → open → half-open → closed)
- [ ] RetryPolicy unit tests verify exponential backoff timing
