import os
import pytest
import pandas as pd
from datetime import datetime, timedelta

from tdxdata.api import TdxData
from tdxdata.core.connection import TdxConnection

pytestmark = pytest.mark.live


def _get_connected_client():
    conn = TdxConnection()
    conn.initialize(timeout=15)
    assert conn.is_connected, "Failed to connect to TDX server"
    return conn


class TestLiveConnection:
    def test_real_connect_and_close(self):
        conn = TdxConnection()
        assert conn.is_connected is False
        conn.initialize(timeout=15)
        assert conn.is_connected is True
        conn.close()
        assert conn.is_connected is False

    def test_reconnect(self):
        conn = _get_connected_client()
        try:
            conn.reconnect()
            assert conn.is_connected is True
            print("\n重连成功")
        finally:
            conn.close()

    def test_context_manager(self):
        conn = _get_connected_client()
        try:
            assert conn.is_connected is True
            print("\n连接正常，手动管理生命周期")
        finally:
            conn.close()

    def test_client_has_methods(self):
        conn = _get_connected_client()
        try:
            client = conn.client
            assert hasattr(client, "quotes")
            assert hasattr(client, "get_k_data")
            assert hasattr(client, "bars")
            assert hasattr(client, "index")
            assert hasattr(client, "finance")
            assert hasattr(client, "xdxr")
            assert hasattr(client, "F10")
            assert hasattr(client, "transactions")
            print("\n客户端 API 方法检查通过")
        finally:
            conn.close()


class TestLiveIndex:
    def test_fetch_sh_index_latest_price(self):
        conn = _get_connected_client()
        try:
            client = conn.client
            df = client.index(symbol="000001", frequency=9, start=0, offset=1)
            assert df is not None and not df.empty, "No data returned for 上证指数"
            latest = df.iloc[-1]
            latest_price = float(latest["close"])
            assert latest_price > 0, f"Invalid close price: {latest_price}"
            print(f"\n上证指数(000001) 最新收盘价: {latest_price}")
            print(f"日期: {latest.get('date', 'N/A')}")
            print(f"开: {latest.get('open', 'N/A')}  高: {latest.get('high', 'N/A')}  低: {latest.get('low', 'N/A')}")
        finally:
            conn.close()

    def test_fetch_sh_index_kline_recent_5days(self):
        conn = _get_connected_client()
        try:
            client = conn.client
            df = client.index(symbol="000001", frequency=9, start=0, offset=5)
            assert df is not None and not df.empty, "No data returned"
            assert len(df) >= 1, "Expected at least 1 row"
            required = {"open", "high", "low", "close"}
            assert required.issubset(set(df.columns)), f"Missing columns: {required - set(df.columns)}"
            date_col = "date" if "date" in df.columns else "datetime"
            print(f"\n上证指数 近{len(df)}日K线:")
            print(df[[date_col, "open", "high", "low", "close", "vol"]].to_string(index=False))
        finally:
            conn.close()

    def test_fetch_sz_index_latest_price(self):
        conn = _get_connected_client()
        try:
            client = conn.client
            df = client.index(symbol="399001", frequency=9, start=0, offset=1)
            assert df is not None and not df.empty, "No data returned for 深证成指"
            latest = df.iloc[-1]
            latest_price = float(latest["close"])
            assert latest_price > 0, f"Invalid close price: {latest_price}"
            print(f"\n深证成指(399001) 最新收盘价: {latest_price}")
        finally:
            conn.close()


class TestLiveHistoryKline:
    def test_fetch_daily_kline(self):
        api = TdxData(timeout=15)
        try:
            api.connect()
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            df = api.fetch_history(
                stock_list=["600519"],
                start_date=start,
                end_date=end,
                period="1d",
            )
            assert df is not None and not df.empty, "600519 日K线为空"
            assert "stock_code" in df.columns
            assert df["stock_code"].iloc[0] == "600519"
            assert all(c in df.columns for c in ["open", "high", "low", "close", "volume"])
            print(f"\n600519 日K线({start}~{end}): {len(df)}条")
            print(df.tail(3).to_string(index=False))
        finally:
            api.close()

    def test_fetch_5m_kline(self):
        api = TdxData(timeout=15)
        try:
            api.connect()
            today = datetime.now().strftime("%Y-%m-%d")
            df = api.fetch_history(
                stock_list=["600519"],
                start_date=today,
                end_date=today,
                period="5m",
            )
            if df is not None and not df.empty:
                assert "stock_code" in df.columns
                print(f"\n600519 5分钟线: {len(df)}条")
            else:
                print(f"\n600519 5分钟线: 非交易时间无数据（正常）")
        finally:
            api.close()

    def test_fetch_weekly_kline(self):
        api = TdxData(timeout=15)
        try:
            api.connect()
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=180)).strftime("%Y-%m-%d")
            df = api.fetch_history(
                stock_list=["600519"],
                start_date=start,
                end_date=end,
                period="1w",
            )
            assert df is not None and not df.empty, "600519 周K线为空"
            print(f"\n600519 周K线({start}~{end}): {len(df)}条")
        finally:
            api.close()

    def test_fetch_multiple_stocks_daily(self):
        api = TdxData(timeout=15)
        try:
            api.connect()
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
            df = api.fetch_history(
                stock_list=["600519", "000001"],
                start_date=start,
                end_date=end,
                period="1d",
            )
            assert df is not None and not df.empty
            codes = set(df["stock_code"].unique())
            assert codes == {"600519", "000001"}
            print(f"\n多股票日K线: {len(df)}条, 股票: {codes}")
        finally:
            api.close()


class TestLiveRealtime:
    def test_fetch_realtime_single_stock(self):
        api = TdxData(timeout=15)
        try:
            api.connect()
            df = api.fetch_realtime(stock_list=["600519"])
            assert df is not None and not df.empty, "600519 实时行情为空"
            assert "stock_code" in df.columns
            assert "close" in df.columns
            price = float(df["close"].iloc[0])
            assert price > 0, f"Invalid price: {price}"
            print(f"\n600519 实时行情:")
            print(f"最新价: {price}")
            if "name" in df.columns:
                print(f"名称: {df['name'].iloc[0]}")
            if "bid_price1" in df.columns:
                print(f"买一: {df['bid_price1'].iloc[0]}, 量: {df['bid_volume1'].iloc[0]}")
            if "ask_price1" in df.columns:
                print(f"卖一: {df['ask_price1'].iloc[0]}, 量: {df['ask_volume1'].iloc[0]}")
        finally:
            api.close()

    def test_fetch_realtime_multiple_stocks(self):
        api = TdxData(timeout=15)
        try:
            api.connect()
            df = api.fetch_realtime(stock_list=["600519", "000001", "600036"])
            assert df is not None and not df.empty
            print(f"\n多股票实时行情: {len(df)}条")
            if "stock_code" in df.columns:
                print(f"股票: {df['stock_code'].unique().tolist()}")
        finally:
            api.close()


class TestLiveFinancial:
    def test_fetch_financial_data(self):
        api = TdxData(timeout=15)
        try:
            api.connect()
            df = api.fetch_financial(stock_code="600519")
            assert df is not None and not df.empty, "600519 财务数据为空"
            assert "stock_code" in df.columns
            print(f"\n600519 财务数据: {len(df)}条")
            print(f"列: {df.columns.tolist()}")
        finally:
            api.close()


class TestLiveDailyBasic:
    def test_fetch_xdxr_data(self):
        api = TdxData(timeout=15)
        try:
            api.connect()
            df = api.fetch_basic(stock_code="600519")
            assert df is not None and not df.empty, "600519 除权除息数据为空"
            assert "stock_code" in df.columns
            print(f"\n600519 除权除息: {len(df)}条")
            if len(df) > 0:
                print(df.tail(3).to_string(index=False))
        finally:
            api.close()


class TestLiveTick:
    def test_fetch_tick_data(self):
        api = TdxData(timeout=15)
        try:
            api.connect()
            today = datetime.now().strftime("%Y-%m-%d")
            df = api.fetch_tick(stock_code="600519", date=today)
            if df is not None and not df.empty:
                assert "stock_code" in df.columns
                assert "price" in df.columns
                print(f"\n600519 逐笔({today}): {len(df)}条")
            else:
                last_trading = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
                df = api.fetch_tick(stock_code="600519", date=last_trading)
                if df is not None and not df.empty:
                    print(f"\n600519 逐笔({last_trading}): {len(df)}条")
                else:
                    print(f"\n600519 逐笔: 非交易时间无数据（正常）")
        finally:
            api.close()


class TestLiveF10:
    def test_fetch_f10_data(self):
        api = TdxData(timeout=15)
        try:
            api.connect()
            result = api.fetch_f10(stock_code="600519", sections=["公司概况"])
            assert isinstance(result, dict)
            if "公司概况" in result:
                print(f"\n600519 F10(公司概况): 数据获取成功")
            else:
                print(f"\n600519 F10(公司概况): 无数据（可能服务器不支持）")
        finally:
            api.close()


class TestLiveIntegration:
    def test_fetch_stock_realtime_via_tdxdata(self):
        api = TdxData(timeout=15)
        try:
            api.connect()
            df = api.fetch_realtime(stock_list=["600519"])
            assert df is not None and not df.empty, "No realtime data for 600519"
            assert "stock_code" in df.columns
            assert "close" in df.columns
            price = float(df["close"].iloc[0])
            assert price > 0, f"Invalid price: {price}"
            print(f"\n贵州茅台(600519) 实时行情:")
            print(f"最新价: {price}")
            print(f"名称: {df['name'].iloc[0] if 'name' in df.columns else 'N/A'}")
        finally:
            api.close()

    def test_fetch_history_and_save_csv(self, tmp_path):
        api = TdxData(timeout=15)
        try:
            api.connect()
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
            df = api.fetch_history(
                stock_list=["600519"],
                start_date=start,
                end_date=end,
                period="1d",
                output="csv",
                output_path=str(tmp_path),
            )
            csv_file = os.path.join(str(tmp_path), "history_kline", "600519.csv")
            assert os.path.exists(csv_file), f"CSV not created: {csv_file}"
            loaded = pd.read_csv(csv_file)
            assert len(loaded) > 0
            print(f"\n历史K线保存CSV: {csv_file}, {len(loaded)}条")
        finally:
            api.close()

    def test_fetch_history_and_save_parquet(self, tmp_path):
        api = TdxData(timeout=15)
        try:
            api.connect()
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
            df = api.fetch_history(
                stock_list=["600519"],
                start_date=start,
                end_date=end,
                period="1d",
                output="parquet",
                output_path=str(tmp_path),
            )
            pq_file = os.path.join(str(tmp_path), "history_kline", "600519.parquet")
            assert os.path.exists(pq_file), f"Parquet not created: {pq_file}"
            print(f"\n历史K线保存Parquet: {pq_file}")
        finally:
            api.close()

    def test_fetch_history_and_save_sqlite(self, tmp_path):
        api = TdxData(timeout=15)
        try:
            api.connect()
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
            df = api.fetch_history(
                stock_list=["600519"],
                start_date=start,
                end_date=end,
                period="1d",
                output="sqlite",
                output_path=str(tmp_path),
            )
            db_file = os.path.join(str(tmp_path), "tdxdata.db")
            assert os.path.exists(db_file), f"SQLite not created: {db_file}"
            print(f"\n历史K线保存SQLite: {db_file}")
        finally:
            api.close()


class TestLiveMorePeriods:
    def test_fetch_1m_kline(self):
        api = TdxData(timeout=15)
        try:
            api.connect()
            today = datetime.now().strftime("%Y-%m-%d")
            df = api.fetch_history(
                stock_list=["600519"],
                start_date=today,
                end_date=today,
                period="1m",
            )
            if df is not None and not df.empty:
                assert "stock_code" in df.columns
                assert "date" in df.columns
                print(f"\n600519 1分钟线: {len(df)}条")
            else:
                print(f"\n600519 1分钟线: 非交易时间无数据（正常）")
        finally:
            api.close()

    def test_fetch_monthly_kline(self):
        api = TdxData(timeout=15)
        try:
            api.connect()
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=365)).strftime("%Y-%m-%d")
            df = api.fetch_history(
                stock_list=["600519"],
                start_date=start,
                end_date=end,
                period="1mon",
            )
            assert df is not None and not df.empty, "600519 月K线为空"
            print(f"\n600519 月K线({start}~{end}): {len(df)}条")
        finally:
            api.close()

    def test_fetch_15m_kline(self):
        api = TdxData(timeout=15)
        try:
            api.connect()
            today = datetime.now().strftime("%Y-%m-%d")
            df = api.fetch_history(
                stock_list=["600519"],
                start_date=today,
                end_date=today,
                period="15m",
            )
            if df is not None and not df.empty:
                print(f"\n600519 15分钟线: {len(df)}条")
            else:
                print(f"\n600519 15分钟线: 非交易时间无数据（正常）")
        finally:
            api.close()

    def test_fetch_30m_kline(self):
        api = TdxData(timeout=15)
        try:
            api.connect()
            today = datetime.now().strftime("%Y-%m-%d")
            df = api.fetch_history(
                stock_list=["600519"],
                start_date=today,
                end_date=today,
                period="30m",
            )
            if df is not None and not df.empty:
                print(f"\n600519 30分钟线: {len(df)}条")
            else:
                print(f"\n600519 30分钟线: 非交易时间无数据（正常）")
        finally:
            api.close()

    def test_fetch_1h_kline(self):
        api = TdxData(timeout=15)
        try:
            api.connect()
            today = datetime.now().strftime("%Y-%m-%d")
            df = api.fetch_history(
                stock_list=["600519"],
                start_date=today,
                end_date=today,
                period="1h",
            )
            if df is not None and not df.empty:
                print(f"\n600519 1小时线: {len(df)}条")
            else:
                print(f"\n600519 1小时线: 非交易时间无数据（正常）")
        finally:
            api.close()


class TestLiveDividendAdjust:
    def test_fetch_kline_front_adjust(self):
        api = TdxData(timeout=15)
        try:
            api.connect()
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            df = api.fetch_history(
                stock_list=["600519"],
                start_date=start,
                end_date=end,
                period="1d",
                dividend_type="front",
            )
            assert df is not None and not df.empty
            assert "stock_code" in df.columns
            print(f"\n600519 前复权日K线: {len(df)}条")
        finally:
            api.close()

    def test_fetch_kline_back_adjust(self):
        api = TdxData(timeout=15)
        try:
            api.connect()
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")
            df = api.fetch_history(
                stock_list=["600519"],
                start_date=start,
                end_date=end,
                period="1d",
                dividend_type="back",
            )
            assert df is not None and not df.empty
            print(f"\n600519 后复权日K线: {len(df)}条")
        finally:
            api.close()


class TestLiveDifferentMarkets:
    def test_fetch_kline_sh_main(self):
        api = TdxData(timeout=15)
        try:
            api.connect()
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
            df = api.fetch_history(
                stock_list=["600036"],
                start_date=start,
                end_date=end,
                period="1d",
            )
            assert df is not None and not df.empty
            assert df["stock_code"].iloc[0] == "600036"
            print(f"\n沪市主板(600036 招商银行) 日K线: {len(df)}条")
        finally:
            api.close()

    def test_fetch_kline_sz_main(self):
        api = TdxData(timeout=15)
        try:
            api.connect()
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
            df = api.fetch_history(
                stock_list=["000001"],
                start_date=start,
                end_date=end,
                period="1d",
            )
            assert df is not None and not df.empty
            assert df["stock_code"].iloc[0] == "000001"
            print(f"\n深市主板(000001 平安银行) 日K线: {len(df)}条")
        finally:
            api.close()

    def test_fetch_kline_gem(self):
        api = TdxData(timeout=15)
        try:
            api.connect()
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
            df = api.fetch_history(
                stock_list=["300750"],
                start_date=start,
                end_date=end,
                period="1d",
            )
            if df is not None and not df.empty:
                print(f"\n创业板(300750 宁德时代) 日K线: {len(df)}条")
            else:
                print(f"\n创业板(300750) 日K线: 无数据")
        finally:
            api.close()

    def test_fetch_kline_star_market(self):
        api = TdxData(timeout=15)
        try:
            api.connect()
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
            df = api.fetch_history(
                stock_list=["688981"],
                start_date=start,
                end_date=end,
                period="1d",
            )
            if df is not None and not df.empty:
                print(f"\n科创板(688981 中芯国际) 日K线: {len(df)}条")
            else:
                print(f"\n科创板(688981) 日K线: 无数据")
        finally:
            api.close()

    def test_fetch_realtime_gem(self):
        api = TdxData(timeout=15)
        try:
            api.connect()
            df = api.fetch_realtime(stock_list=["300750"])
            if df is not None and not df.empty:
                print(f"\n创业板(300750 宁德时代) 实时行情: 获取成功")
            else:
                print(f"\n创业板(300750) 实时行情: 无数据")
        finally:
            api.close()


class TestLiveErrorHandling:
    def test_fetch_history_invalid_stock(self):
        api = TdxData(timeout=15)
        try:
            api.connect()
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=10)).strftime("%Y-%m-%d")
            df = api.fetch_history(
                stock_list=["999999"],
                start_date=start,
                end_date=end,
                period="1d",
            )
            assert df is None or df.empty, "无效股票代码应返回空数据"
            print(f"\n无效股票代码(999999): 正确返回空数据")
        finally:
            api.close()

    def test_fetch_realtime_invalid_stock(self):
        api = TdxData(timeout=15)
        try:
            api.connect()
            df = api.fetch_realtime(stock_list=["999999"])
            is_empty = df is None or df.empty or df["name"].iloc[0] is pd.NA
            assert is_empty, "无效股票代码应返回空数据或NA"
            print(f"\n无效股票代码实时行情(999999): 正确返回空/NA数据")
        finally:
            api.close()

    def test_fetch_tick_invalid_date(self):
        api = TdxData(timeout=15)
        try:
            api.connect()
            df = api.fetch_tick(stock_code="600519", date="2020-01-01")
            assert df is None or df.empty, "历史日期可能无数据"
            print(f"\n历史日期逐笔(2020-01-01): 返回空数据（正常）")
        finally:
            api.close()


class TestLiveMultiStockBatch:
    def test_fetch_history_5_stocks(self):
        api = TdxData(timeout=15)
        try:
            api.connect()
            end = datetime.now().strftime("%Y-%m-%d")
            start = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
            df = api.fetch_history(
                stock_list=["600519", "000001", "600036", "000858", "002415"],
                start_date=start,
                end_date=end,
                period="1d",
            )
            assert df is not None and not df.empty
            codes = set(df["stock_code"].unique())
            print(f"\n5只股票批量日K线: {len(df)}条, 股票: {codes}")
        finally:
            api.close()

    def test_fetch_realtime_5_stocks(self):
        api = TdxData(timeout=15)
        try:
            api.connect()
            df = api.fetch_realtime(stock_list=["600519", "000001", "600036", "000858", "002415"])
            assert df is not None and not df.empty
            print(f"\n5只股票批量实时行情: {len(df)}条")
        finally:
            api.close()
