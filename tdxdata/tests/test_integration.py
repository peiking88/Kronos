import os
import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock, patch

from tdxdata.api import TdxData
from tdxdata.core.connection import TdxConnection
from tdxdata.core.registry import PluginRegistry


def _make_mock_connection():
    conn = TdxConnection()
    mock_client = MagicMock()
    mock_client.close = MagicMock()
    mock_client.reconnect = MagicMock()
    conn._client = mock_client
    conn._initialized = True
    return conn, mock_client


class TestIntegration:
    def test_context_manager_lifecycle(self):
        conn, mock_client = _make_mock_connection()
        from tdxdata.errors.resource import ResourceManager
        rm = ResourceManager(conn)
        with rm as c:
            assert c.is_connected
        mock_client.close.assert_called_once()

    def test_fetch_history_daily_end_to_end(self):
        conn, mock_client = _make_mock_connection()
        dates = pd.date_range("2024-01-01", periods=5, freq="B")
        kline_df = pd.DataFrame({
            "open": [100.0 + i for i in range(5)],
            "high": [105.0 + i for i in range(5)],
            "low": [99.0 + i for i in range(5)],
            "close": [104.0 + i for i in range(5)],
            "vol": [10000.0] * 5,
            "amount": [1000000.0] * 5,
            "date": dates,
        })
        mock_client.get_k_data.return_value = kline_df
        mock_client.bars.return_value = kline_df

        from tdxdata.core.data_manager import DataManager
        dm = DataManager(conn)
        df = dm.fetch(
            source="history_kline",
            stock_list=["600519"],
            start_date="2024-01-01",
            end_date="2024-01-07",
            period="1d",
        )

        assert not df.empty
        assert "stock_code" in df.columns
        assert len(df) == 5

    def test_fetch_realtime_end_to_end(self):
        conn, mock_client = _make_mock_connection()
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
            "bid2": 103.0, "bid_vol2": 600,
            "bid3": 102.5, "bid_vol3": 700,
            "bid4": 102.0, "bid_vol4": 800,
            "bid5": 101.5, "bid_vol5": 900,
            "ask1": 104.5, "ask_vol1": 400,
            "ask2": 105.0, "ask_vol2": 300,
            "ask3": 105.5, "ask_vol3": 200,
            "ask4": 106.0, "ask_vol4": 100,
            "ask5": 106.5, "ask_vol5": 50,
        }])

        from tdxdata.core.data_manager import DataManager
        dm = DataManager(conn)
        df = dm.fetch(
            source="realtime_snapshot",
            stock_code="600519",
        )

        assert not df.empty
        assert "stock_code" in df.columns
        assert "bid_price1" in df.columns
        assert "ask_price1" in df.columns

    def test_fetch_to_csv_storage(self, tmp_path):
        conn, mock_client = _make_mock_connection()
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        kline_df = pd.DataFrame({
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [99.0, 100.0, 101.0],
            "close": [104.0, 105.0, 106.0],
            "vol": [10000.0, 11000.0, 12000.0],
            "amount": [1e6, 1.1e6, 1.2e6],
            "date": dates,
        })
        mock_client.get_k_data.return_value = kline_df
        mock_client.bars.return_value = kline_df

        output_dir = str(tmp_path)
        from tdxdata.core.data_manager import DataManager
        dm = DataManager(conn)
        df = dm.fetch(
            source="history_kline",
            output="csv",
            output_path=output_dir,
            stock_list=["600519"],
            start_date="2024-01-01",
            end_date="2024-01-05",
            period="1d",
        )

        csv_file = os.path.join(output_dir, "history_kline", "600519.csv")
        assert os.path.exists(csv_file)

    def test_fetch_to_parquet_storage(self, tmp_path):
        conn, mock_client = _make_mock_connection()
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        kline_df = pd.DataFrame({
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [99.0, 100.0, 101.0],
            "close": [104.0, 105.0, 106.0],
            "vol": [10000.0, 11000.0, 12000.0],
            "amount": [1e6, 1.1e6, 1.2e6],
            "date": dates,
        })
        mock_client.get_k_data.return_value = kline_df
        mock_client.bars.return_value = kline_df

        output_dir = str(tmp_path)
        from tdxdata.core.data_manager import DataManager
        dm = DataManager(conn)
        df = dm.fetch(
            source="history_kline",
            output="parquet",
            output_path=output_dir,
            stock_list=["600519"],
            start_date="2024-01-01",
            end_date="2024-01-05",
            period="1d",
        )

        pq_file = os.path.join(output_dir, "history_kline", "600519.parquet")
        assert os.path.exists(pq_file)

    def test_fetch_to_sqlite_storage(self, tmp_path):
        conn, mock_client = _make_mock_connection()
        dates = pd.date_range("2024-01-01", periods=3, freq="B")
        kline_df = pd.DataFrame({
            "open": [100.0, 101.0, 102.0],
            "high": [105.0, 106.0, 107.0],
            "low": [99.0, 100.0, 101.0],
            "close": [104.0, 105.0, 106.0],
            "vol": [10000.0, 11000.0, 12000.0],
            "amount": [1e6, 1.1e6, 1.2e6],
            "date": dates,
        })
        mock_client.get_k_data.return_value = kline_df
        mock_client.bars.return_value = kline_df

        output_dir = str(tmp_path)
        from tdxdata.core.data_manager import DataManager
        dm = DataManager(conn)
        df = dm.fetch(
            source="history_kline",
            output="sqlite",
            output_path=output_dir,
            stock_list=["600519"],
            start_date="2024-01-01",
            end_date="2024-01-05",
            period="1d",
        )

        db_file = os.path.join(output_dir, "tdxdata.db")
        assert os.path.exists(db_file)

    def test_fetch_tick_end_to_end(self):
        conn, mock_client = _make_mock_connection()
        mock_client.transactions.return_value = pd.DataFrame([
            {"time": "09:30:00", "price": 100.0, "vol": 100, "amount": 10000, "buyorsell": 0},
            {"time": "09:30:01", "price": 100.5, "vol": 200, "amount": 20100, "buyorsell": 1},
        ])

        from tdxdata.core.data_manager import DataManager
        dm = DataManager(conn)
        df = dm.fetch(
            source="tick",
            stock_code="600519",
            date="2024-01-02",
        )

        assert not df.empty
        assert len(df) == 2
        assert "buy_sell_flag" in df.columns

    def test_resource_cleanup_on_error(self):
        conn, mock_client = _make_mock_connection()
        from tdxdata.errors.resource import ResourceManager
        rm = ResourceManager(conn)

        try:
            with rm:
                raise RuntimeError("Simulated error")
        except RuntimeError:
            pass

        mock_client.close.assert_called_once()
