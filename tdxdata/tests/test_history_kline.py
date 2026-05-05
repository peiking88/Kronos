import pytest
import pandas as pd
import numpy as np

from tdxdata.core.registry import PluginRegistry
from tdxdata.sources.history_kline import HistoryKlineSource


class TestHistoryKlineSource:
    def setup_method(self):
        PluginRegistry.clear()

    def _make_kline_df(self, n=5, code="600519"):
        dates = pd.date_range("2024-01-01", periods=n, freq="B")
        return pd.DataFrame({
            "open": [100.0 + i for i in range(n)],
            "high": [105.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [104.0 + i for i in range(n)],
            "vol": [10000.0 + i * 1000 for i in range(n)],
            "amount": [1000000.0 + i * 100000 for i in range(n)],
            "date": dates,
        })

    def test_fetch_daily(self, mock_connection, mock_client):
        kline_df = self._make_kline_df(5)
        mock_client.get_k_data.return_value = kline_df
        mock_client.bars.return_value = kline_df

        source = HistoryKlineSource(mock_connection)
        df = source.fetch(
            stock_list=["600519"],
            start_date="2024-01-01",
            end_date="2024-01-07",
            period="1d",
        )

        assert not df.empty
        assert "stock_code" in df.columns
        assert "date" in df.columns
        assert all(c in df.columns for c in ["open", "high", "low", "close", "volume", "amount"])
        assert df["stock_code"].iloc[0] == "600519"

    def test_fetch_empty_data(self, mock_connection, mock_client):
        mock_client.get_k_data.return_value = pd.DataFrame()

        source = HistoryKlineSource(mock_connection)
        df = source.fetch(
            stock_list=["600519"],
            start_date="2024-01-01",
            end_date="2024-01-07",
            period="1d",
        )
        assert df.empty

    def test_invalid_period_raises(self, mock_connection):
        source = HistoryKlineSource(mock_connection)
        with pytest.raises(ValueError, match="Unsupported period"):
            source.fetch(
                stock_list=["600519"],
                start_date="2024-01-01",
                end_date="2024-01-07",
                period="2d",
            )

    def test_fetch_5m_bars(self, mock_connection, mock_client):
        dates = pd.date_range("2024-01-01 09:30", periods=20, freq="5min")
        kline_df = pd.DataFrame({
            "open": [100.0 + i for i in range(20)],
            "high": [105.0 + i for i in range(20)],
            "low": [99.0 + i for i in range(20)],
            "close": [104.0 + i for i in range(20)],
            "vol": [1000.0] * 20,
            "amount": [10000.0] * 20,
            "datetime": dates,
        })
        mock_client.bars.return_value = kline_df

        source = HistoryKlineSource(mock_connection)
        df = source.fetch(
            stock_list=["600519"],
            start_date="2024-01-01",
            end_date="2024-01-01",
            period="5m",
        )

        assert not df.empty
        assert "stock_code" in df.columns
        mock_client.bars.assert_called_once_with(symbol="600519", frequency=0, offset=800)

    def test_fetch_weekly(self, mock_connection, mock_client):
        kline_df = self._make_kline_df(3)
        mock_client.get_k_data.return_value = kline_df
        mock_client.bars.return_value = kline_df

        source = HistoryKlineSource(mock_connection)
        df = source.fetch(
            stock_list=["600519"],
            start_date="2024-01-01",
            end_date="2024-01-31",
            period="1w",
        )

        assert not df.empty

    def test_fetch_multiple_stocks(self, mock_connection, mock_client):
        kline_df = self._make_kline_df(3)
        mock_client.get_k_data.return_value = kline_df
        mock_client.bars.return_value = kline_df

        source = HistoryKlineSource(mock_connection)
        df = source.fetch(
            stock_list=["600519", "000001"],
            start_date="2024-01-01",
            end_date="2024-01-07",
            period="1d",
        )

        assert not df.empty
        assert set(df["stock_code"].unique()) == {"600519", "000001"}

    def test_fetch_with_error_continues(self, mock_connection, mock_client):
        kline_df = self._make_kline_df(3)
        mock_client.get_k_data.side_effect = [Exception("Network error"), kline_df]
        mock_client.bars.return_value = kline_df

        source = HistoryKlineSource(mock_connection)
        df = source.fetch(
            stock_list=["600519", "000001"],
            start_date="2024-01-01",
            end_date="2024-01-07",
            period="1d",
        )

        assert not df.empty
        assert len(df["stock_code"].unique()) == 1

    def test_vol_renamed_to_volume(self, mock_connection, mock_client):
        kline_df = self._make_kline_df(3)
        mock_client.get_k_data.return_value = kline_df
        mock_client.bars.return_value = kline_df

        source = HistoryKlineSource(mock_connection)
        df = source.fetch(
            stock_list=["600519"],
            start_date="2024-01-01",
            end_date="2024-01-07",
            period="1d",
        )

        assert "volume" in df.columns
