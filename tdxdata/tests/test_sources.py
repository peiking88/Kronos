import pytest
import pandas as pd
from unittest.mock import MagicMock

from tdxdata.core.registry import PluginRegistry
from tdxdata.sources.realtime_snapshot import RealtimeSnapshotSource
from tdxdata.sources.tick import TickDataSource
from tdxdata.sources.financial import FinancialDataSource
from tdxdata.sources.f10 import F10DataSource
from tdxdata.sources.daily_basic import DailyBasicSource


class TestRealtimeSnapshotSource:
    def setup_method(self):
        PluginRegistry.clear()

    def test_fetch_single_stock(self, mock_connection, mock_client):
        mock_client.quotes.return_value = pd.DataFrame([{
            "code": "600519",
            "name": "贵州茅台",
            "open": 100.0,
            "high": 105.0,
            "low": 99.0,
            "price": 104.0,
            "last_close": 100.0,
            "vol": 100000,
            "amount": 10000000,
            "bid1": 103.5, "bid_vol1": 500,
            "ask1": 104.5, "ask_vol1": 400,
        }])

        source = RealtimeSnapshotSource(mock_connection)
        df = source.fetch(stock_list=["600519"])

        assert not df.empty
        assert df["stock_code"].iloc[0] == "600519"
        assert "bid_price1" in df.columns
        assert "ask_price1" in df.columns

    def test_fetch_multiple_stocks(self, mock_connection, mock_client):
        mock_client.quotes.return_value = pd.DataFrame([
            {"code": "600519", "name": "贵州茅台", "price": 104.0},
            {"code": "000001", "name": "平安银行", "price": 12.5},
        ])

        source = RealtimeSnapshotSource(mock_connection)
        df = source.fetch(stock_list=["600519", "000001"])

        assert len(df) == 2

    def test_fetch_empty_result(self, mock_connection, mock_client):
        mock_client.quotes.return_value = pd.DataFrame()

        source = RealtimeSnapshotSource(mock_connection)
        df = source.fetch(stock_list=["600519"])

        assert df.empty

    def test_no_stock_code_raises(self, mock_connection):
        source = RealtimeSnapshotSource(mock_connection)
        with pytest.raises(ValueError, match="stock_list or stock_code"):
            source.fetch()


class TestTickDataSource:
    def setup_method(self):
        PluginRegistry.clear()

    def test_fetch_tick_with_date(self, mock_connection, mock_client):
        mock_client.transactions.return_value = pd.DataFrame([
            {"time": "09:30:00", "price": 100.0, "vol": 100, "buyorsell": 0},
            {"time": "09:30:01", "price": 100.5, "vol": 200, "buyorsell": 1},
        ])

        source = TickDataSource(mock_connection)
        df = source.fetch(stock_code="600519", date="2024-01-02")

        assert not df.empty
        assert len(df) == 2
        assert "stock_code" in df.columns
        assert "buy_sell_flag" in df.columns
        mock_client.transactions.assert_called_once_with(
            symbol="600519", start=0, offset=2000, date="2024-01-02"
        )

    def test_fetch_tick_without_date(self, mock_connection, mock_client):
        mock_client.transaction.return_value = pd.DataFrame([
            {"time": "09:30:00", "price": 100.0, "vol": 100, "buyorsell": 0},
        ])

        source = TickDataSource(mock_connection)
        df = source.fetch(stock_code="600519")

        assert not df.empty
        mock_client.transaction.assert_called_once_with(
            symbol="600519", start=0, offset=2000
        )

    def test_fetch_empty_result(self, mock_connection, mock_client):
        mock_client.transactions.return_value = pd.DataFrame()

        source = TickDataSource(mock_connection)
        df = source.fetch(stock_code="600519", date="2024-01-02")

        assert df.empty

    def test_fetch_error_returns_empty(self, mock_connection, mock_client):
        mock_client.transactions.side_effect = Exception("Network error")

        source = TickDataSource(mock_connection)
        df = source.fetch(stock_code="600519", date="2024-01-02")

        assert df.empty


class TestFinancialDataSource:
    def setup_method(self):
        PluginRegistry.clear()

    def test_fetch_financial(self, mock_connection, mock_client):
        mock_client.finance.return_value = pd.DataFrame([{
            "code": "600519",
            "report_date": "2024-03-31",
            "eps": 10.5,
            "bps": 120.0,
        }])

        source = FinancialDataSource(mock_connection)
        df = source.fetch(stock_code="600519")

        assert not df.empty
        assert "stock_code" in df.columns
        mock_client.finance.assert_called_once_with(symbol="600519")

    def test_fetch_empty_result(self, mock_connection, mock_client):
        mock_client.finance.return_value = pd.DataFrame()

        source = FinancialDataSource(mock_connection)
        df = source.fetch(stock_code="600519")

        assert df.empty

    def test_fetch_error_returns_empty(self, mock_connection, mock_client):
        mock_client.finance.side_effect = Exception("API error")

        source = FinancialDataSource(mock_connection)
        df = source.fetch(stock_code="600519")

        assert df.empty


class TestF10DataSource:
    def setup_method(self):
        PluginRegistry.clear()

    def test_fetch_f10(self, mock_connection, mock_client):
        mock_client.F10.return_value = pd.DataFrame([
            {"title": "公司概况", "content": "测试内容"},
        ])

        source = F10DataSource(mock_connection)
        result = source.fetch(stock_code="600519", sections=["公司概况"])

        assert isinstance(result, dict)
        assert "公司概况" in result

    def test_fetch_f10_empty_result(self, mock_connection, mock_client):
        mock_client.F10.return_value = None

        source = F10DataSource(mock_connection)
        result = source.fetch(stock_code="600519", sections=["公司概况"])

        assert isinstance(result, dict)
        assert len(result) == 0

    def test_fetch_f10_default_sections(self, mock_connection, mock_client):
        mock_client.F10.return_value = pd.DataFrame([{"content": "test"}])

        source = F10DataSource(mock_connection)
        result = source.fetch(stock_code="600519")

        assert isinstance(result, dict)
        assert mock_client.F10.call_count > 1


class TestDailyBasicSource:
    def setup_method(self):
        PluginRegistry.clear()

    def test_fetch_daily_basic(self, mock_connection, mock_client):
        mock_client.xdxr.return_value = pd.DataFrame([
            {"code": "600519", "date": "2024-01-02", "category": 1, "fhsl": 10.0},
        ])

        source = DailyBasicSource(mock_connection)
        df = source.fetch(stock_code="600519")

        assert not df.empty
        assert "stock_code" in df.columns
        mock_client.xdxr.assert_called_once_with(symbol="600519")

    def test_fetch_empty_result(self, mock_connection, mock_client):
        mock_client.xdxr.return_value = pd.DataFrame()

        source = DailyBasicSource(mock_connection)
        df = source.fetch(stock_code="600519")

        assert df.empty

    def test_fetch_error_returns_empty(self, mock_connection, mock_client):
        mock_client.xdxr.side_effect = Exception("API error")

        source = DailyBasicSource(mock_connection)
        df = source.fetch(stock_code="600519")

        assert df.empty
