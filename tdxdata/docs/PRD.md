# PRD: TDX Data Reader - 通达信行情数据读取接口

## Problem Statement

作为个人投资者，需要一个统一、便捷的工具来获取通达信的全量行情数据（历史K线、实时快照、分笔Tick、财务数据、F10信息），用于投资分析和决策。当前面临以下问题：

- 通达信官方 tqcenter 插件的使用方式较为原始，需要手动配置路径、手动处理数据格式转换
- 不同类型的数据（K线、快照、Tick、财务）需要不同的接口调用方式，缺乏统一封装
- 数据输出格式单一，无法灵活选择存储方式
- 缺乏增量更新机制，每次需要全量拉取，效率低下
- 错误处理不完善，网络异常或数据缺失时容易中断

## Solution

构建一个基于插件式架构的 Python 数据读取库 `tdxdata`，封装通达信 tqcenter 官方插件，提供统一的 API 接口获取全量行情数据。支持多格式输出（DataFrame / CSV / SQLite / Parquet），支持增量更新，具备完善的错误处理和重试机制，并自动搜索下载官方插件。

## User Stories

### 核心数据获取

1. As a 个人投资者, I want to 通过一行代码获取指定股票的历史K线数据, so that 我可以快速进行技术分析
2. As a 个人投资者, I want to 获取日线/周线/月线等多种周期的历史K线, so that 我可以从不同时间维度分析走势
3. As a 个人投资者, I want to 获取1分钟/5分钟/15分钟/30分钟/60分钟等分钟级别K线, so that 我可以进行精细化的短线分析
4. As a 个人投资者, I want to 历史数据默认采用前复权方式, so that 我可以直接进行连续的价格分析而不受除权影响
5. As a 个人投资者, I want to 也能选择后复权或不复权, so that 我可以根据不同分析需求灵活选择
6. As a 个人投资者, I want to 获取多只股票的批量历史数据, so that 我可以高效地进行板块或组合分析

### 实时行情

7. As a 个人投资者, I want to 获取单只股票的实时行情快照, so that 我可以了解当前盘面情况
8. As a 个人投资者, I want to 获取多只股票的批量实时快照, so that 我可以同时监控多只股票
9. As a 个人投资者, I want to 实时快照包含五档买卖盘口数据, so that 我可以分析盘口深度和委托分布
10. As a 个人投资者, I want to 获取实时成交明细（分笔Tick数据）, so that 我可以进行逐笔分析

### 财务与F10数据

11. As a 个人投资者, I want to 获取结构化的F10公司概况数据, so that 我可以了解公司的基本信息
12. As a 个人投资者, I want to 获取结构化的F10股东信息（十大股东、流通股东、股东变化）, so that 我可以分析股权结构和主力动向
13. As a 个人投资者, I want to 获取结构化的财务报表（资产负债表、利润表、现金流量表）, so that 我可以进行基本面分析
14. As a 个人投资者, I want to 获取分红配股历史记录, so that 我可以了解公司的分红政策和历史
15. As a 个人投资者, I want to 获取F10所有栏目的结构化数据, so that 我可以获得全面的公司信息
16. As a 个人投资者, I want to F10数据以 pandas DataFrame 格式返回, so that 我可以方便地进行数据分析和处理

### 数据输出与存储

17. As a 个人投资者, I want to 数据默认以 pandas DataFrame 格式返回, so that 我可以直接用 Python 进行分析
18. As a 个人投资者, I want to 将数据导出为 CSV 文件, so that 我可以用 Excel 等工具查看
19. As a 个人投资者, I want to 将数据存储到 SQLite 数据库, so that 我可以进行高效的 SQL 查询
20. As a 个人投资者, I want to 将数据存储为 Parquet 格式, so that 我可以获得更好的压缩比和查询性能
21. As a 个人投资者, I want to 通过统一的接口选择输出格式, so that 我不需要学习不同的导出方法

### 增量更新

22. As a 个人投资者, I want to 支持增量更新历史K线数据, so that 我不需要每次都全量下载
23. As a 个人投资者, I want to 系统自动记录已拉取数据的时间点, so that 增量更新时只需获取新增部分
24. As a 个人投资者, I want to 也能选择全量重新拉取, so that 我可以在数据异常时进行修复
25. As a 个人投资者, I want to 增量更新时自动处理缺失数据的填充, so that 数据连续完整

### 插件管理

26. As a 个人投资者, I want to 系统自动搜索本地已安装的通达信及 tqcenter 插件, so that 我不需要手动配置路径
27. As a 个人投资者, I want to 当本地未找到插件时自动从官方下载, so that 我可以快速开始使用
28. As a 个人投资者, I want to 系统自动配置 Python 路径, so that 我可以在任何 IDE 中使用

### 错误处理与可靠性

29. As a 个人投资者, I want to 网络异常时自动重试, so that 临时网络波动不会导致数据采集中断
30. As a 个人投资者, I want to 获取到详细的错误日志, so that 我可以定位和排查问题
31. As a 个人投资者, I want to 数据缺失时得到明确的提示, so that 我不会在不知情的情况下使用不完整数据
32. As a 个人投资者, I want to 大批量数据拉取支持断点续传, so that 中断后不需要从头开始
33. As a 个人投资者, I want to 与通达信的连接异常时自动清理资源, so that 通达信进程不会卡死

### 代码质量

34. As a 开发者, I want to 所有核心模块都有单元测试, so that 我可以放心地修改和扩展功能
35. As a 开发者, I want to 项目有清晰的 API 文档, so that 我可以快速上手使用
36. As a 开发者, I want to 插件式架构便于扩展新的数据源或存储格式, so that 未来可以轻松添加功能

## Implementation Decisions

### Architecture: Plugin-based Architecture

采用插件式架构，分为以下核心模块：

#### 1. Core Module (`tdxdata/core/`)
- **TdxConnection**: 管理与通达信终端的连接生命周期（initialize / close），确保资源正确释放
- **PluginManager**: 自动搜索本地通达信安装路径，发现 tqcenter 插件，自动配置 Python 路径；若未找到则从官方下载
- **DataManager**: 数据拉取的统一入口，协调 DataSource 和 StorageProvider

#### 2. Data Source Plugins (`tdxdata/sources/`)
- **HistoryKlineSource**: 获取历史K线数据（日线/周线/月线/分钟线），支持前复权/后复权/不复权
- **RealtimeSnapshotSource**: 获取实时行情快照，包含五档盘口
- **TickDataSource**: 获取分笔成交明细数据
- **FinancialDataSource**: 获取财务报表数据（资产负债表/利润表/现金流量表）
- **F10DataSource**: 获取结构化F10全量数据（公司概况/股东信息/分红配送/所有栏目）
- **DataSourceBase**: 抽象基类，定义统一的 `fetch()` 接口，所有数据源插件继承此基类

#### 3. Storage Plugins (`tdxdata/storage/`)
- **DataFrameStorage**: 默认存储，直接返回 pandas DataFrame
- **CSVStorage**: 将数据保存为 CSV 文件
- **SQLiteStorage**: 将数据保存到 SQLite 数据库，按股票代码分表
- **ParquetStorage**: 将数据保存为 Parquet 列式文件
- **StorageBase**: 抽象基类，定义统一的 `save()` / `load()` 接口

#### 4. Sync Module (`tdxdata/sync/`)
- **SyncManager**: 管理增量更新逻辑，记录每只股票每种数据类型的最后同步时间
- **SyncState**: 持久化存储同步状态（JSON 文件），记录已拉取数据的时间范围
- **GapDetector**: 检测数据缺口，支持补全缺失时间段的数据

#### 5. Error Handling (`tdxdata/errors/`)
- **RetryPolicy**: 可配置的重试策略（重试次数、间隔、退避策略）
- **ErrorHandler**: 统一的异常处理，将 tqcenter 原始错误转换为有意义的业务异常
- **CircuitBreaker**: 连续失败时熔断，避免对通达信造成过大压力
- **ResourceManager**: 上下文管理器，确保 tq.close() 在异常时也能被调用

#### 6. Logging (`tdxdata/logging/`)
- **Logger**: 基于 Python logging 模块的结构化日志，记录数据拉取的详细信息

### Key Interfaces

```python
# Unified fetch interface
result = tdx.fetch(
    source="history_kline",        # Data source plugin name
    stock_list=["600519.SH"],      # Stock codes
    start_date="20250101",         # Start date
    end_date="20251231",           # End date
    period="1d",                   # K-line period
    dividend_type="front",         # Dividend adjustment: front/back/none
    output="dataframe",            # Output format: dataframe/csv/sqlite/parquet
    output_path="output/",         # Output path (for file-based formats)
    incremental=True               # Enable incremental update
)

# Realtime snapshot
snapshot = tdx.fetch(
    source="realtime_snapshot",
    stock_list=["600519.SH", "000001.SZ"],
    output="dataframe"
)

# F10 structured data
f10_data = tdx.fetch(
    source="f10",
    stock_code="600519.SH",
    sections=["all"],              # Or specific: ["shareholder", "finance", "dividend", "profile"]
    output="dataframe"
)

# Context manager for connection
with TdxData() as tdx:
    df = tdx.fetch_history(stock_list=["600519.SH"], period="1d")
```

### Data Format Standards

- 所有数据统一返回 pandas DataFrame
- 列名采用英文标准化命名（open, high, low, close, volume, amount 等）
- 日期时间统一为 datetime64 类型
- 股票代码格式统一为 `代码.市场`（如 600519.SH, 000001.SZ）
- 数值类型统一为 float64/int64，缺失值为 NaN

### Incremental Update Design

- 同步状态存储在 `~/.tdxdata/sync_state.json`
- 每只股票每种数据类型独立记录最后同步时间
- 增量拉取时自动计算时间范围：`last_sync_time + 1` 到 `now`
- 提供强制全量刷新选项

## Testing Decisions

### Testing Philosophy
- 只测试外部行为，不测试实现细节
- 每个 DataSource 插件独立测试
- 每个 Storage 插件独立测试
- 使用 mock 隔离 tqcenter 依赖，确保测试不依赖通达信运行

### Test Coverage

1. **TdxConnection 测试**: 测试连接初始化、关闭、资源清理
2. **PluginManager 测试**: 测试插件搜索、路径配置、自动下载逻辑
3. **DataSource 测试**: 对每个数据源插件，测试正常数据获取、空数据、异常数据
4. **Storage 测试**: 对每个存储插件，测试数据写入、读取、格式正确性
5. **SyncManager 测试**: 测试增量更新逻辑、状态持久化、缺口检测
6. **ErrorHandler 测试**: 测试重试策略、熔断机制、异常转换
7. **集成测试**: 使用 mock 的 tqcenter 端到端测试完整数据流

### Test Data Strategy
- 准备 fixture 数据模拟 tqcenter 返回的原始格式
- 覆盖正常数据、空数据、缺失字段、异常值等边界情况

## Out of Scope

- 实时行情订阅推送（WebSocket 等）
- 自动定时任务调度
- Web UI 或图形界面
- 多数据源对比（如迅投 MiniQMT、Pytdx 等）
- 交易接口（下单、撤单等）
- 策略回测引擎
- 跨平台支持（当前仅支持 Windows + 通达信环境）
- Docker 容器化部署

## Further Notes

- 通达信金融终端（64位量化版）必须在 Windows 上运行，因此本工具主要在 Windows 环境下使用
- tqcenter 插件路径通常在通达信安装目录的 `PYPlugins/user` 下
- 分笔 Tick 数据量较大，需注意存储空间
- F10 数据的栏目结构可能因通达信版本不同而有差异，需做好兼容处理
- 前复权数据会随除权除息事件变化，历史前复权数据需要定期刷新
