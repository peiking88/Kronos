# tdxdata

A Python library for reading TDX (通达信) market data, built on top of [mootdx](https://github.com/mootdx/mootdx).

## Features

- **History K-line** — daily, weekly, monthly, and intraday bars (1m/5m/15m/30m/1h)
- **Realtime Snapshot** — level-2 bid/ask quotes with 5-level depth
- **Tick Data** — historical and intraday tick-by-tick transactions
- **Financial Data** — company financial statements
- **F10 Data** — company fundamentals by section
- **Daily Basic** — ex-rights/ex-dividend (除权除息) information
- **Local File Reader** — read K-line data directly from local TDX binary files
- **Multiple Storage Backends** — DataFrame (in-memory), CSV, SQLite, Parquet
- **Plugin Architecture** — extensible source and storage registry
- **Sync & Gap Detection** — incremental sync with gap detection

## Installation

```bash
pip install -e .
```

### Dependencies

- Python >= 3.10
- pandas >= 2.0
- numpy >= 1.24
- pyarrow >= 14.0
- mootdx >= 0.11

## Quick Start

```python
from tdxdata import TdxData

# Connect using mootdx (auto-discovers best server)
with TdxData() as api:
    # Fetch daily K-line
    df = api.fetch_history(
        stock_list=["600519", "000001"],
        start_date="2024-01-01",
        end_date="2024-06-30",
        period="1d",
    )
    print(df.head())

    # Fetch realtime snapshot
    quotes = api.fetch_realtime(stock_list=["600519", "000001"])
    print(quotes)

    # Fetch tick data
    ticks = api.fetch_tick(stock_code="600519", date="2024-01-02")
    print(ticks.head())

    # Fetch financial data
    financial = api.fetch_financial(stock_code="600519")
    print(financial)

    # Fetch F10 data (returns dict of DataFrames)
    f10 = api.fetch_f10(stock_code="600519", sections=["公司概况", "最新提示"])
    print(f10.keys())

    # Fetch ex-rights/ex-dividend info
    basic = api.fetch_basic(stock_code="600519")
    print(basic)
```

### Custom Server

```python
# Specify a custom TDX server
api = TdxData(server=("110.41.147.114", 7709), timeout=30)
api.connect()
# ... use api ...
api.close()
```

## Local File Reader

Read K-line data directly from local TDX binary files (no network required):

```python
from tdxdata import TdxData

api = TdxData()
api.connect()

# Read daily K-line from local files
# Default TDX directory: ~/.local/share/tdxcfv/drive_c/tc/
df = api.fetch_local(stock_code="600519", period="1d")
print(df.head())

# Read 1-minute bars
df_1m = api.fetch_local(stock_code="600519", period="1m")

# Read 5-minute bars
df_5m = api.fetch_local(stock_code="600519", period="5m")

# Read multiple stocks
df_multi = api.fetch_local(stock_list=["600519", "000001"], period="1d")

# Specify custom TDX directory
df = api.fetch_local(stock_code="600519", period="1d", tdxdir="/path/to/tdx")

# Save to storage
df = api.fetch_local(
    stock_code="600519",
    period="1d",
    output="parquet",
    output_path="./data",
)
```

### Supported Periods for Local Reader

| Period | Description | File Location |
|---|---|---|
| `"1d"` | Daily bars | `vipdoc/{sh,sz}/lday/*.day` |
| `"1m"` | 1-minute bars | `vipdoc/{sh,sz}/minline/*.lc1` |
| `"5m"` | 5-minute bars | `vipdoc/{sh,sz}/fzline/*.lc5` |

### Default TDX Directory

Default: `~/.local/share/tdxcfv/drive_c/tc/`

Override via `tdxdir` parameter or set the `TDXDIR` environment variable.

## Local Data vs Server Data Comparison

### Key Differences

| Feature | Local Data (`fetch_local`) | Server Data (`fetch_history`) |
|---|---|---|
| **Data Source** | Local TDX binary files | TDX servers (network) |
| **Network Required** | ❌ No | ✅ Yes |
| **Speed** | ⚡ Very fast (local I/O) | 🐢 Slower (network latency) |
| **Data Freshness** | Depends on local download | Real-time / latest |
| **Supported Periods** | 1d, 1m, 5m only | 1d, 1m, 5m, 15m, 30m, 1h, 1w, 1mon |
| **Date Range** | All available local data | Customizable (`start_date`, `end_date`) |
| **Dividend Adjustment** | ❌ Not supported | ✅ Front/back/none adjustment |
| **Error Handling** | Skip missing files | Network retry logic |

### When to Use Each

| Use Case | Recommended | Reason |
|---|---|---|
| **Historical Backtesting** | `fetch_local` | Fast, complete data, no network |
| **Real-time Monitoring** | `fetch_history` | Latest data |
| **Offline Environment** | `fetch_local` | No network required |
| **Batch Analysis** | `fetch_local` | Higher efficiency |
| **Latest Data** | `fetch_history` | Real-time updates |
| **Multiple Timeframes** | `fetch_history` | More period options |

### Performance Comparison

```python
import time
from tdxdata import TdxData

api = TdxData()
api.connect()

# Local data (fast)
start = time.time()
df_local = api.fetch_local(stock_code="600519", period="1d")
local_time = time.time() - start
print(f"Local fetch: {local_time:.3f}s, {len(df_local)} rows")

# Server data (slower)
start = time.time()
df_server = api.fetch_history(
    stock_list=["600519"],
    start_date="2024-01-01",
    end_date="2024-12-31",
    period="1d"
)
server_time = time.time() - start
print(f"Server fetch: {server_time:.3f}s, {len(df_server)} rows")
```

## Storage Backends

Save data directly to different formats:

```python
# Save to CSV
api.fetch_history(
    stock_list=["600519"],
    start_date="2024-01-01",
    end_date="2024-06-30",
    output="csv",
    output_path="./data",
)

# Save to Parquet
api.fetch_history(
    stock_list=["600519"],
    start_date="2024-01-01",
    end_date="2024-06-30",
    output="parquet",
    output_path="./data",
)

# Save to SQLite
api.fetch_history(
    stock_list=["600519"],
    start_date="2024-01-01",
    end_date="2024-06-30",
    output="sqlite",
    output_path="./data",
)
```

## Project Structure

```
tdxdata/
├── api.py                          # Main entry point (TdxData class)
├── core/
│   ├── connection.py               # TdxConnection wrapping mootdx StdQuotes
│   ├── data_manager.py             # DataManager orchestrating source + storage
│   └── registry.py                 # Plugin registry (register_source / register_storage)
├── sources/
│   ├── base.py                     # DataSourceBase ABC with _normalize_columns()
│   ├── history_kline.py            # History K-line source
│   ├── realtime_snapshot.py        # Realtime snapshot source
│   ├── tick.py                     # Tick data source
│   ├── financial.py                # Financial data source
│   ├── f10.py                      # F10 company info source
│   ├── daily_basic.py              # Ex-rights/ex-dividend source
│   └── local_kline.py              # Local file reader source
├── storage/
│   ├── base.py                     # StorageBase ABC
│   ├── dataframe.py                # In-memory DataFrame storage
│   ├── csv.py                      # CSV file storage
│   ├── sqlite.py                   # SQLite database storage
│   └── parquet.py                  # Parquet file storage
├── sync/
│   ├── state.py                    # Sync state persistence
│   ├── gap_detector.py             # Data gap detection
│   └── manager.py                  # Sync manager
├── errors/
│   ├── exceptions.py               # Exception hierarchy
│   ├── retry.py                    # Retry policy with exponential backoff
│   ├── circuit_breaker.py          # Circuit breaker pattern
│   └── resource.py                 # Resource manager (context manager)
└── logging/
    └── logger.py                   # Logging utilities
```

## Plugin System

### Register a Custom Data Source

```python
from tdxdata.core.registry import register_source
from tdxdata.sources.base import DataSourceBase

@register_source("my_source")
class MySource(DataSourceBase):
    def fetch(self, **kwargs):
        client = self._connection.client
        # Use mootdx client to fetch data
        return client.some_method(...)
```

### Register a Custom Storage Backend

```python
from tdxdata.core.registry import register_storage
from tdxdata.storage.base import StorageBase

@register_storage("my_storage")
class MyStorage(StorageBase):
    def save(self, df, **kwargs):
        # Custom save logic
        pass

    def load(self, **kwargs):
        # Custom load logic
        pass
```

## API Reference

### `TdxData(server=None, timeout=15)`

Main entry point for the library.

| Method | Description |
|---|---|
| `fetch_history(stock_list, start_date, end_date, period, dividend_type, output, output_path)` | Fetch historical K-line data |
| `fetch_realtime(stock_code, stock_list, output, output_path)` | Fetch realtime market snapshot |
| `fetch_tick(stock_code, date, output, output_path)` | Fetch tick-by-tick transaction data |
| `fetch_f10(stock_code, sections)` | Fetch F10 company information |
| `fetch_basic(stock_code, date, output, output_path)` | Fetch ex-rights/ex-dividend data |
| `fetch_financial(stock_code, output, output_path)` | Fetch financial statements |
| `fetch_local(stock_code, stock_list, period, tdxdir, output, output_path)` | Read K-line from local TDX files |

### Supported K-line Periods

| Period | Description |
|---|---|
| `"1m"` | 1-minute bars |
| `"5m"` | 5-minute bars |
| `"15m"` | 15-minute bars |
| `"30m"` | 30-minute bars |
| `"1h"` | 1-hour bars |
| `"1d"` | Daily bars |
| `"1w"` | Weekly bars |
| `"1mon"` | Monthly bars |

### Dividend Types

| Type | Description |
|---|---|
| `"front"` | Forward-adjusted (前复权) |
| `"back"` | Backward-adjusted (后复权) |
| `"none"` | No adjustment |

## Testing

```bash
pip install -e ".[dev]"

# Run unit tests (mock-based, fast)
pytest tests/ -v

# Run unit tests only (exclude live/local)
pytest tests/ -m "not live and not local" -v

# Run network integration tests
pytest tests/test_live.py -v -s

# Run local file tests
pytest tests/test_live_local.py -v -s
```

### Test Suite Overview (148 tests)

| Category | Count | Description |
|---|---|---|
| **Unit Tests** | 98 | Mock-based, fast, stable |
| **Live Tests** | 38 | Real network to TDX servers |
| **Local Tests** | 12 | Real local binary file reading |

### Test Files

| File | Tests | Description |
|---|---|---|
| `test_connection.py` | 10 | TdxConnection lifecycle |
| `test_registry.py` | 9 | Plugin registry |
| `test_history_kline.py` | 8 | History K-line source |
| `test_sources.py` | 17 | All data sources (realtime, tick, financial, F10, daily_basic) |
| `test_local_kline.py` | 11 | Local file reader source |
| `test_storage.py` | 9 | CSV, SQLite, Parquet storage |
| `test_sync.py` | 11 | Sync state, gap detection, sync manager |
| `test_integration.py` | 8 | End-to-end integration tests |
| `test_retry.py` | 6 | Retry policy |
| `test_circuit_breaker.py` | 9 | Circuit breaker |
| `test_live.py` | 38 | Real network integration tests |
| `test_live_local.py` | 12 | Local file integration tests |

## License

MIT
