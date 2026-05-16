# tdxdata API Reference

> Python library for reading TDX (通达信) market data, designed for quantitative trading developers.

## Table of Contents

- [Getting Started](#getting-started)
- [Connection Management](#connection-management)
- [History K-line](#fetch_history)
- [Realtime Snapshot](#fetch_realtime)
- [Tick Data](#fetch_tick)
- [Financial Data](#fetch_financial)
- [Ex-rights/Ex-dividend](#fetch_basic)
- [F10 Company Info](#fetch_f10)
- [Local File Reader](#fetch_local)
- [Hybrid Data Source](#fetch_hybrid)
- [Storage Backends](#storage-backends)
- [Plugin System](#plugin-system)
- [Error Handling](#error-handling)

---

## Getting Started

### Installation

```bash
pip install -e .
```

### Dependencies

- Python >= 3.10
- pandas >= 2.0
- numpy >= 1.24
- pyarrow >= 14.0
- mootdx >= 0.11

### Quick Example

```python
from tdxdata import TdxData

api = TdxData()
api.connect()

# Fetch daily K-line for Kweichow Moutai (贵州茅台)
df = api.fetch_history(
    stock_list=["600519"],
    start_date="2024-01-01",
    end_date="2024-12-31",
    period="1d",
)
print(df.head())

api.close()
```

### Context Manager

```python
with TdxData() as api:
    df = api.fetch_realtime(stock_list=["600519", "000001"])
    print(df)
```

---

## Connection Management

### `TdxData(server=None, timeout=15)`

Create a TdxData instance.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `server` | `tuple[str, int]` | `None` | Custom TDX server address, e.g. `("110.41.147.114", 7709)`. Auto-discovers best server when `None`. |
| `timeout` | `int` | `15` | Connection timeout in seconds |

### `connect()`

Establish connection to TDX server. Must be called before any data fetch.

```python
api = TdxData()
api.connect()
```

### `close()`

Close the connection and release resources.

```python
api.close()
```

### Custom Server

```python
api = TdxData(server=("110.41.147.114", 7709), timeout=30)
api.connect()
```

---

## `fetch_history()`

Fetch historical K-line data from remote TDX server. Supports daily, weekly, monthly, and intraday bars.

### Signature

```python
def fetch_history(
    self,
    stock_list: list[str],
    start_date: str,
    end_date: str,
    period: str = "1d",
    dividend_type: str = "front",
    output: str = "dataframe",
    output_path: Optional[str] = None,
) -> pd.DataFrame
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `stock_list` | `list[str]` | required | List of stock codes, e.g. `["600519", "000001"]` |
| `start_date` | `str` | required | Start date in `YYYY-MM-DD` format |
| `end_date` | `str` | required | End date in `YYYY-MM-DD` format |
| `period` | `str` | `"1d"` | K-line period (see table below) |
| `dividend_type` | `str` | `"front"` | Dividend adjustment type |
| `output` | `str` | `"dataframe"` | Storage backend: `"dataframe"`, `"csv"`, `"parquet"`, `"sqlite"` |
| `output_path` | `str` | `None` | Output directory path (required when `output` is not `"dataframe"`) |

### Supported Periods

| Period | Description | Data Source |
|---|---|---|
| `"1m"` | 1-minute bars | `client.bars(frequency=8)` |
| `"5m"` | 5-minute bars | `client.bars(frequency=0)` |
| `"15m"` | 15-minute bars | `client.bars(frequency=1)` |
| `"30m"` | 30-minute bars | `client.bars(frequency=2)` |
| `"1h"` | 1-hour bars | `client.bars(frequency=3)` |
| `"1d"` | Daily bars | `client.get_k_data()` |
| `"1w"` | Weekly bars | `client.get_k_data()` |
| `"1mon"` | Monthly bars | `client.get_k_data()` |

### Dividend Adjustment Types

| Type | Description | Implementation |
|---|---|---|
| `"front"` | Forward-adjusted (前复权) | Fetches qfq factor from Sina, applies via `merge_asof(direction="backward")` |
| `"back"` | Backward-adjusted (后复权) | Fetches hfq factor from Sina, applies via `merge_asof(direction="forward")` |
| `"none"` | No adjustment | Returns raw data from TDX server |

> **Note:** Dividend adjustment is only applied to daily/weekly/monthly bars (`1d`, `1w`, `1mon`). Intraday bars (`1m`, `5m`, etc.) are not adjusted. The adjustment factor is fetched from Sina Finance API; if the factor fetch fails, raw data is returned with a warning log.

### Return Value

`pd.DataFrame` with columns:

| Column | Type | Description |
|---|---|---|
| `stock_code` | `str` | Stock code, e.g. `"600519"` |
| `date` | `datetime64` | Date/time of the bar |
| `open` | `float64` | Opening price |
| `high` | `float64` | Highest price |
| `low` | `float64` | Lowest price |
| `close` | `float64` | Closing price |
| `volume` | `float64` | Trading volume (手) |
| `amount` | `float64` | Trading amount (元) |

### Examples

**Daily K-line for multiple stocks:**

```python
df = api.fetch_history(
    stock_list=["600519", "000001", "600036"],
    start_date="2024-01-01",
    end_date="2024-06-30",
    period="1d",
)
```

**5-minute bars with forward adjustment:**

```python
df = api.fetch_history(
    stock_list=["600519"],
    start_date="2024-12-01",
    end_date="2024-12-31",
    period="5m",
    dividend_type="front",
)
```

**Weekly K-line and save to Parquet:**

```python
df = api.fetch_history(
    stock_list=["600519"],
    start_date="2024-01-01",
    end_date="2024-12-31",
    period="1w",
    output="parquet",
    output_path="./data",
)
```

### Quantitative Use Cases

**Calculate moving average:**

```python
df = api.fetch_history(
    stock_list=["600519"],
    start_date="2024-01-01",
    end_date="2024-12-31",
    period="1d",
)
df["ma5"] = df["close"].rolling(5).mean()
df["ma20"] = df["close"].rolling(20).mean()
```

**Multi-stock return comparison:**

```python
df = api.fetch_history(
    stock_list=["600519", "000001", "600036"],
    start_date="2024-01-01",
    end_date="2024-12-31",
    period="1d",
)
for code in df["stock_code"].unique():
    sub = df[df["stock_code"] == code].sort_values("date")
    ret = (sub["close"].iloc[-1] / sub["close"].iloc[0] - 1) * 100
    print(f"{code}: {ret:.2f}%")
```

---

## `fetch_realtime()`

Fetch realtime level-2 market snapshot with 5-level bid/ask depth.

### Signature

```python
def fetch_realtime(
    self,
    stock_code: Optional[str] = None,
    stock_list: Optional[list[str]] = None,
    output: str = "dataframe",
    output_path: Optional[str] = None,
) -> pd.DataFrame
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `stock_code` | `str` | `None` | Single stock code. Use either `stock_code` or `stock_list`. |
| `stock_list` | `list[str]` | `None` | List of stock codes. Use either `stock_code` or `stock_list`. |
| `output` | `str` | `"dataframe"` | Storage backend |
| `output_path` | `str` | `None` | Output directory path |

### Return Value

`pd.DataFrame` with columns:

| Column | Type | Description |
|---|---|---|
| `stock_code` | `str` | Stock code |
| `name` | `str` | Stock name |
| `datetime` | `datetime` | Snapshot timestamp |
| `open` | `float64` | Opening price of the day |
| `high` | `float64` | Highest price of the day |
| `low` | `float64` | Lowest price of the day |
| `close` | `float64` | Latest price |
| `last_close` | `float64` | Previous close |
| `volume` | `float64` | Trading volume |
| `amount` | `float64` | Trading amount |
| `bid_price1` ~ `bid_price5` | `float64` | Bid prices (level 1-5) |
| `bid_volume1` ~ `bid_volume5` | `float64` | Bid volumes (level 1-5) |
| `ask_price1` ~ `ask_price5` | `float64` | Ask prices (level 1-5) |
| `ask_volume1` ~ `ask_volume5` | `float64` | Ask volumes (level 1-5) |
| `turnover_rate` | `float64` | Turnover rate |

### Examples

**Single stock realtime:**

```python
df = api.fetch_realtime(stock_code="600519")
print(f"贵州茅台 最新价: {df['close'].iloc[0]}")
print(f"买一: {df['bid_price1'].iloc[0]} × {df['bid_volume1'].iloc[0]}")
print(f"卖一: {df['ask_price1'].iloc[0]} × {df['ask_volume1'].iloc[0]}")
```

**Multiple stocks:**

```python
df = api.fetch_realtime(stock_list=["600519", "000001", "600036"])
for _, row in df.iterrows():
    print(f"{row['name']}({row['stock_code']}): {row['close']}")
```

### Quantitative Use Cases

**Order book imbalance:**

```python
df = api.fetch_realtime(stock_list=["600519"])
total_bid = sum(df[f"bid_volume{i}"].iloc[0] for i in range(1, 6))
total_ask = sum(df[f"ask_volume{i}"].iloc[0] for i in range(1, 6))
imbalance = (total_bid - total_ask) / (total_bid + total_ask)
print(f"Order book imbalance: {imbalance:.4f}")
```

**Spread calculation:**

```python
spread = df["ask_price1"].iloc[0] - df["bid_price1"].iloc[0]
mid_price = (df["ask_price1"].iloc[0] + df["bid_price1"].iloc[0]) / 2
relative_spread = spread / mid_price * 10000  # in basis points
```

---

## `fetch_tick()`

Fetch tick-by-tick transaction data for a specific stock and date.

### Signature

```python
def fetch_tick(
    self,
    stock_code: str,
    date: Optional[str] = None,
    output: str = "dataframe",
    output_path: Optional[str] = None,
) -> pd.DataFrame
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `stock_code` | `str` | required | Stock code, e.g. `"600519"` |
| `date` | `str` | `None` | Date in `YYYY-MM-DD` format. Defaults to today when `None`. |
| `output` | `str` | `"dataframe"` | Storage backend |
| `output_path` | `str` | `None` | Output directory path |

### Return Value

`pd.DataFrame` with columns:

| Column | Type | Description |
|---|---|---|
| `stock_code` | `str` | Stock code |
| `datetime` | `datetime64` | Transaction timestamp |
| `price` | `float64` | Transaction price |
| `volume` | `float64` | Transaction volume |
| `buy_sell_flag` | `int` | Buy/sell indicator: `0` = buy, `1` = sell |
| `amount` | `float64` | Transaction amount (if available) |

### Examples

**Today's tick data:**

```python
df = api.fetch_tick(stock_code="600519")
print(f"Total ticks: {len(df)}")
```

**Specific date:**

```python
df = api.fetch_tick(stock_code="600519", date="2024-12-20")
```

### Quantitative Use Cases

**Volume-Weighted Average Price (VWAP):**

```python
df = api.fetch_tick(stock_code="600519")
vwap = (df["price"] * df["volume"]).sum() / df["volume"].sum()
print(f"VWAP: {vwap:.2f}")
```

**Buy/sell pressure analysis:**

```python
buy_vol = df[df["buy_sell_flag"] == 0]["volume"].sum()
sell_vol = df[df["buy_sell_flag"] == 1]["volume"].sum()
pressure = buy_vol / (buy_vol + sell_vol)
print(f"Buy pressure ratio: {pressure:.2%}")
```

---

## `fetch_financial()`

Fetch company financial statement data.

### Signature

```python
def fetch_financial(
    self,
    stock_code: str,
    output: str = "dataframe",
    output_path: Optional[str] = None,
) -> pd.DataFrame
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `stock_code` | `str` | required | Stock code |
| `output` | `str` | `"dataframe"` | Storage backend |
| `output_path` | `str` | `None` | Output directory path |

### Return Value

`pd.DataFrame` — columns depend on mootdx financial data output. Always includes:

| Column | Type | Description |
|---|---|---|
| `stock_code` | `str` | Stock code |
| `report_date` | `datetime64` | Financial report date (if available) |

### Example

```python
df = api.fetch_financial(stock_code="600519")
print(f"Financial reports: {len(df)}")
print(df.columns.tolist())
```

---

## `fetch_basic()`

Fetch ex-rights/ex-dividend (除权除息) information, including bonus issues, rights issues, and dividend distributions.

### Signature

```python
def fetch_basic(
    self,
    stock_code: str,
    date: Optional[str] = None,
    output: str = "dataframe",
    output_path: Optional[str] = None,
) -> pd.DataFrame
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `stock_code` | `str` | required | Stock code |
| `date` | `str` | `None` | Specific date filter |
| `output` | `str` | `"dataframe"` | Storage backend |
| `output_path` | `str` | `None` | Output directory path |

### Return Value

`pd.DataFrame` — columns depend on mootdx xdxr output. Always includes:

| Column | Type | Description |
|---|---|---|
| `stock_code` | `str` | Stock code |
| `date` | `datetime64` | Ex-rights date |

### Example

```python
df = api.fetch_basic(stock_code="600519")
print(f"Ex-rights records: {len(df)}")
print(df.tail(5))
```

### Quantitative Use Cases

**Adjust historical prices:**

```python
kline = api.fetch_history(
    stock_list=["600519"],
    start_date="2020-01-01",
    end_date="2024-12-31",
    period="1d",
    dividend_type="front",
)
```

---

## `fetch_f10()`

Fetch F10 company fundamental information by section.

### Signature

```python
def fetch_f10(
    self,
    stock_code: str,
    sections: Optional[list[str]] = None,
) -> dict[str, pd.DataFrame]
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `stock_code` | `str` | required | Stock code |
| `sections` | `list[str]` | `None` | List of F10 section names. Defaults to all sections. |

### Default Sections

| Section | Description |
|---|---|
| `"最新提示"` | Latest tips/alerts |
| `"公司概况"` | Company overview |
| `"财务分析"` | Financial analysis |
| `"股东研究"` | Shareholder research |
| `"股本结构"` | Share capital structure |
| `"分红扩股"` | Dividend and stock split |
| `"行业题材"` | Industry and sector |

### Return Value

`dict[str, pd.DataFrame]` — key is section name, value is the corresponding DataFrame.

### Examples

**Fetch specific sections:**

```python
result = api.fetch_f10(stock_code="600519", sections=["公司概况", "财务分析"])
for section, df in result.items():
    print(f"\n=== {section} ===")
    print(df)
```

**Fetch all sections:**

```python
result = api.fetch_f10(stock_code="600519")
print(f"Available sections: {list(result.keys())}")
```

---

## `fetch_local()`

Read K-line data directly from local TDX binary files. Supports dividend adjustment for daily bars (requires network to fetch factor from Sina Finance).

### Signature

```python
def fetch_local(
    self,
    stock_list: Optional[list[str]] = None,
    stock_code: Optional[str] = None,
    period: str = "1d",
    tdxdir: Optional[str] = None,
    dividend_type: str = "none",
    output: str = "dataframe",
    output_path: Optional[str] = None,
) -> pd.DataFrame
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `stock_list` | `list[str]` | `None` | List of stock codes |
| `stock_code` | `str` | `None` | Single stock code |
| `period` | `str` | `"1d"` | K-line period (see table below) |
| `tdxdir` | `str` | `None` | TDX data directory path. Default: `~/.local/share/tdxcfv/drive_c/tc/` |
| `dividend_type` | `str` | `"none"` | Dividend adjustment type (only for `"1d"` period) |
| `output` | `str` | `"dataframe"` | Storage backend |
| `output_path` | `str` | `None` | Output directory path |

### Supported Periods

| Period | Description | File Location |
|---|---|---|
| `"1d"` | Daily bars | `vipdoc/{sh,sz}/lday/*.day` |
| `"1m"` | 1-minute bars | `vipdoc/{sh,sz}/minline/*.lc1` |
| `"5m"` | 5-minute bars | `vipdoc/{sh,sz}/fzline/*.lc5` |

### Dividend Adjustment

| Type | Description |
|---|---|
| `"none"` | No adjustment (default, local binary data as-is) |
| `"front"` | Forward-adjusted (前复权) — latest price unchanged, historical prices adjusted |
| `"back"` | Backward-adjusted (后复权) — all prices adjusted to reflect total returns |

> **Note:** Dividend adjustment is only applied to daily bars (`period="1d"`). Minute bars are not adjusted. The adjustment factor is fetched from Sina Finance API (requires network). If the factor fetch fails, raw data is returned with a warning.

### Return Value

`pd.DataFrame` with columns:

| Column | Type | Description |
|---|---|---|
| `stock_code` | `str` | Stock code |
| `date` | `datetime64` | Date/time of the bar |
| `open` | `float64` | Opening price |
| `high` | `float64` | Highest price |
| `low` | `float64` | Lowest price |
| `close` | `float64` | Closing price |
| `volume` | `float64` | Trading volume |
| `amount` | `float64` | Trading amount |

### Examples

**Read daily K-line from local files:**

```python
df = api.fetch_local(stock_code="600519", period="1d")
print(f"Local daily bars: {len(df)}")
```

**Read daily K-line with forward adjustment:**

```python
df = api.fetch_local(stock_code="600519", period="1d", dividend_type="front")
```

**Read 1-minute bars:**

```python
df = api.fetch_local(stock_code="600519", period="1m")
```

**Multiple stocks:**

```python
df = api.fetch_local(stock_list=["600519", "000001"], period="1d")
```

**Custom TDX directory:**

```python
df = api.fetch_local(
    stock_code="600519",
    period="1d",
    tdxdir="/path/to/tdx",
)
```

**Save to Parquet:**

```python
df = api.fetch_local(
    stock_code="600519",
    period="1d",
    output="parquet",
    output_path="./data",
)
```

### Quantitative Use Cases

**Backtesting with local data (no network):**

```python
df = api.fetch_local(stock_code="600519", period="1d")
df["returns"] = df["close"].pct_change()
df["ma20"] = df["close"].rolling(20).mean()
df["signal"] = (df["close"] > df["ma20"]).astype(int)
```

**High-frequency analysis with minute data:**

```python
df_1m = api.fetch_local(stock_code="600519", period="1m")
df_1m["vwap"] = (df_1m["close"] * df_1m["volume"]).cumsum() / df_1m["volume"].cumsum()
```

---

## `fetch_hybrid()`

Read K-line data from local TDX binary files first, then supplement missing data from the network. Combines the speed of local reads with the completeness of remote data.

### How It Works

1. **Read local data** — Load all available K-line data from local TDX binary files
2. **Detect gaps** — Compare local data date range with requested `start_date`/`end_date`
3. **Fetch missing data** — If local data is incomplete, fetch only the missing portion from the network
4. **Merge & deduplicate** — Combine local and remote data, removing duplicates
5. **Apply adjustment** — If `dividend_type` is specified, apply adjustment to the merged dataset

### Signature

```python
def fetch_hybrid(
    self,
    stock_list: Optional[list[str]] = None,
    stock_code: Optional[str] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    period: str = "1d",
    tdxdir: Optional[str] = None,
    dividend_type: str = "none",
    output: str = "dataframe",
    output_path: Optional[str] = None,
) -> pd.DataFrame
```

### Parameters

| Parameter | Type | Default | Description |
|---|---|---|---|
| `stock_list` | `list[str]` | `None` | List of stock codes |
| `stock_code` | `str` | `None` | Single stock code |
| `start_date` | `str` | `None` | Start date (`YYYY-MM-DD`). Defaults to local data start. |
| `end_date` | `str` | `None` | End date (`YYYY-MM-DD`). Defaults to today. |
| `period` | `str` | `"1d"` | K-line period (see table below) |
| `tdxdir` | `str` | `None` | TDX data directory path. Default: `~/.local/share/tdxcfv/drive_c/tc/` |
| `dividend_type` | `str` | `"none"` | Dividend adjustment type (only for `"1d"` period) |
| `output` | `str` | `"dataframe"` | Storage backend |
| `output_path` | `str` | `None` | Output directory path |

### Supported Periods

| Period | Description |
|---|---|
| `"1d"` | Daily bars |
| `"1m"` | 1-minute bars |
| `"5m"` | 5-minute bars |

### Dividend Adjustment

| Type | Description |
|---|---|
| `"none"` | No adjustment (default) |
| `"front"` | Forward-adjusted (前复权) — latest price unchanged, historical prices adjusted |
| `"back"` | Backward-adjusted (后复权) — all prices adjusted to reflect total returns |

> **Note:** Dividend adjustment is only applied to daily bars (`period="1d"`). The adjustment factor is fetched from Sina Finance API (requires network). If the factor fetch fails, raw data is returned with a warning.

### Return Value

`pd.DataFrame` with columns:

| Column | Type | Description |
|---|---|---|
| `stock_code` | `str` | Stock code |
| `date` | `datetime64` | Date/time of the bar |
| `open` | `float64` | Opening price |
| `high` | `float64` | Highest price |
| `low` | `float64` | Lowest price |
| `close` | `float64` | Closing price |
| `volume` | `float64` | Trading volume |
| `amount` | `float64` | Trading amount |

### Examples

**Basic usage — local data + network supplement:**

```python
df = api.fetch_hybrid(stock_code="600519", period="1d")
print(f"Date range: {df['date'].min().date()} ~ {df['date'].max().date()}")
```

**With date range:**

```python
df = api.fetch_hybrid(
    stock_code="600519",
    start_date="2024-01-01",
    end_date="2024-12-31",
    period="1d",
)
```

**With forward adjustment:**

```python
df = api.fetch_hybrid(
    stock_code="600519",
    start_date="2024-01-01",
    end_date="2024-12-31",
    period="1d",
    dividend_type="front",
)
```

**Multiple stocks:**

```python
df = api.fetch_hybrid(
    stock_list=["600519", "000001"],
    start_date="2025-01-01",
    end_date="2025-03-01",
    period="1d",
)
```

### Fallback Behavior

- If the TDX directory is not found, falls back to pure remote fetch
- If the remote connection is unavailable, returns local data only
- If the remote fetch fails (e.g., requesting today's data before market close), returns local data with a warning

---

## Storage Backends

All `fetch_*` methods (except `fetch_f10`) support persisting data to different storage backends via the `output` parameter.

### Supported Backends

| Backend | `output` Value | Description |
|---|---|---|
| DataFrame | `"dataframe"` | In-memory, returned as pd.DataFrame (default) |
| CSV | `"csv"` | Save to CSV file with UTF-8-BOM encoding |
| Parquet | `"parquet"` | Save to Parquet file (Snappy compression) |
| SQLite | `"sqlite"` | Save to SQLite database |

### Usage

```python
# Save to CSV
api.fetch_history(
    stock_list=["600519"],
    start_date="2024-01-01",
    end_date="2024-06-30",
    output="csv",
    output_path="./data",
)
# Creates: ./data/history_kline/600519.csv

# Save to Parquet
api.fetch_history(
    stock_list=["600519"],
    start_date="2024-01-01",
    end_date="2024-06-30",
    output="parquet",
    output_path="./data",
)
# Creates: ./data/history_kline/600519.parquet

# Save to SQLite
api.fetch_history(
    stock_list=["600519"],
    start_date="2024-01-01",
    end_date="2024-06-30",
    output="sqlite",
    output_path="./data",
)
# Creates: ./data/tdxdata.db (table: history_kline_600519)
```

### File Naming Convention

| Backend | Path Pattern |
|---|---|
| CSV | `{output_path}/{source}/{stock_code}.csv` |
| Parquet | `{output_path}/{source}/{stock_code}.parquet` |
| SQLite | `{output_path}/tdxdata.db` → table `{source}_{stock_code}` |

---

## Plugin System

tdxdata uses a plugin registry pattern for extensible data sources and storage backends.

### Register a Custom Data Source

```python
from tdxdata.core.registry import register_source
from tdxdata.sources.base import DataSourceBase

@register_source("my_source")
class MySource(DataSourceBase):
    def fetch(self, **kwargs):
        client = self._connection.client
        return client.some_method(...)
```

### Register a Custom Storage Backend

```python
from tdxdata.core.registry import register_storage
from tdxdata.storage.base import StorageBase

@register_storage("my_storage")
class MyStorage(StorageBase):
    def save(self, df, **kwargs):
        pass

    def load(self, **kwargs):
        pass
```

### Use Custom Plugins

```python
api.fetch(source="my_source", param1="value1")
```

---

## Error Handling

### Exception Hierarchy

```
TdxDataError (base)
├── ConnectionError      — Connection failures
├── SourceError           — Data source errors
└── StorageError          — Storage backend errors
```

### Retry Policy

Built-in retry with exponential backoff:

```python
from tdxdata.errors.retry import RetryPolicy

policy = RetryPolicy(max_retries=3, base_delay=1.0, max_delay=30.0)
```

### Circuit Breaker

Prevents cascading failures:

```python
from tdxdata.errors.circuit_breaker import CircuitBreaker

cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
```

### Resource Manager

Ensures connection cleanup:

```python
from tdxdata.errors.resource import ResourceManager

rm = ResourceManager(connection)
with rm as conn:
    # Use connection
    pass
# Connection is automatically closed on error
```

---

## Stock Code Convention

tdxdata uses plain stock codes without exchange prefix/suffix:

| Market | Code Pattern | Example |
|---|---|---|
| Shanghai Main Board | `6xxxxx` | `600519` (贵州茅台) |
| Shenzhen Main Board | `000xxx` / `001xxx` | `000001` (平安银行) |
| ChiNext (创业板) | `300xxx` | `300750` (宁德时代) |
| STAR Market (科创板) | `688xxx` | `688981` (中芯国际) |
| Shanghai Index | `000001` | 上证指数 |
| Shenzhen Index | `399001` | 深证成指 |

> **Note:** For index queries, use `client.index()` directly via `TdxConnection.client`.

---

## Testing

```bash
# Run unit tests (mock-based, fast)
pytest tests/ -m "not live and not local" -v

# Run network integration tests
pytest tests/test_live.py -v -s

# Run local file tests
pytest tests/test_live_local.py -v -s

# Run all tests
pytest tests/ -v
```

### Test Coverage

| Category | Count | Description |
|---|---|---|
| Unit Tests | 98 | Mock-based, fast, stable |
| Live Tests | 38 | Real network to TDX servers |
| Local Tests | 12 | Real local binary file reading |
| **Total** | **148** | |
