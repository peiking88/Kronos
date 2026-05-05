import os
import pytest
import pandas as pd
from unittest.mock import MagicMock, patch

from tdxdata.core.registry import PluginRegistry
from tdxdata.sources.hybrid_kline import HybridKlineSource

TDXDIR = os.path.expanduser("~/.local/share/tdxcfv/drive_c/tc/")

SKIP_REASON = f"TDX directory not found: {TDXDIR}"


def _has_tdx_data():
    return os.path.isdir(TDXDIR)


def _make_source(mock_client=None):
    from tdxdata.core.connection import TdxConnection
    conn = TdxConnection()
    if mock_client:
        conn._client = mock_client
    conn._initialized = True
    return HybridKlineSource(conn, tdxdir=TDXDIR)


class TestHybridKlineSourceUnit:
    def setup_method(self):
        PluginRegistry.clear()

    def _make_local_df(self, n=10, start="2024-01-01"):
        dates = pd.date_range(start, periods=n, freq="B")
        return pd.DataFrame({
            "open": [100.0 + i for i in range(n)],
            "high": [105.0 + i for i in range(n)],
            "low": [99.0 + i for i in range(n)],
            "close": [104.0 + i for i in range(n)],
            "vol": [10000.0 + i * 1000 for i in range(n)],
            "amount": [1000000.0 + i * 100000 for i in range(n)],
            "date": dates,
        })

    def _make_remote_df(self, n=5, start="2024-01-15"):
        dates = pd.date_range(start, periods=n, freq="B")
        return pd.DataFrame({
            "open": [110.0 + i for i in range(n)],
            "high": [115.0 + i for i in range(n)],
            "low": [109.0 + i for i in range(n)],
            "close": [114.0 + i for i in range(n)],
            "vol": [20000.0 + i * 1000 for i in range(n)],
            "amount": [2000000.0 + i * 100000 for i in range(n)],
            "date": dates,
        })

    def test_no_stock_code_raises(self, mock_connection):
        source = HybridKlineSource(mock_connection, tdxdir="/tmp")
        with pytest.raises(ValueError, match="stock_list or stock_code"):
            source.fetch()

    def test_invalid_period_raises(self, mock_connection):
        source = HybridKlineSource(mock_connection, tdxdir="/tmp")
        with pytest.raises(ValueError, match="Unsupported period"):
            source.fetch(stock_code="600519", period="1h")

    def test_local_only_no_remote_needed(self, mock_connection, mock_client):
        local_df = self._make_local_df(10, "2024-01-01")
        mock_client.get_k_data.return_value = pd.DataFrame()

        source = HybridKlineSource(mock_connection, tdxdir="/tmp")
        with patch.object(source, "_read_local", return_value=local_df):
            df = source.fetch(
                stock_code="600519",
                start_date="2024-01-01",
                end_date="2024-01-14",
                period="1d",
            )

        assert not df.empty
        assert "stock_code" in df.columns
        assert len(df) == 10

    def test_remote_supplements_local(self, mock_connection, mock_client):
        local_df = self._make_local_df(10, "2024-01-01")
        remote_df = self._make_remote_df(5, "2024-01-15")
        mock_client.get_k_data.return_value = remote_df

        source = HybridKlineSource(mock_connection, tdxdir="/tmp")
        with patch.object(source, "_read_local", return_value=local_df):
            df = source.fetch(
                stock_code="600519",
                start_date="2024-01-01",
                end_date="2024-01-21",
                period="1d",
            )

        assert not df.empty
        assert df["date"].max() >= pd.Timestamp("2024-01-19")

    def test_no_local_data_fetches_remote(self, mock_connection, mock_client):
        remote_df = self._make_remote_df(5, "2024-01-01")
        mock_client.get_k_data.return_value = remote_df

        source = HybridKlineSource(mock_connection, tdxdir="/tmp")
        with patch.object(source, "_read_local", return_value=None):
            df = source.fetch(
                stock_code="600519",
                start_date="2024-01-01",
                end_date="2024-01-07",
                period="1d",
            )

        assert not df.empty
        mock_client.get_k_data.assert_called()

    def test_local_covers_full_range(self, mock_connection, mock_client):
        local_df = self._make_local_df(20, "2024-01-01")
        mock_client.get_k_data.return_value = pd.DataFrame()

        source = HybridKlineSource(mock_connection, tdxdir="/tmp")
        with patch.object(source, "_read_local", return_value=local_df):
            df = source.fetch(
                stock_code="600519",
                start_date="2024-01-01",
                end_date="2024-01-26",
                period="1d",
            )

        assert not df.empty
        mock_client.get_k_data.assert_not_called()

    def test_multiple_stocks(self, mock_connection, mock_client):
        local_df = self._make_local_df(10, "2024-01-01")
        mock_client.get_k_data.return_value = pd.DataFrame()

        source = HybridKlineSource(mock_connection, tdxdir="/tmp")
        with patch.object(source, "_read_local", return_value=local_df):
            df = source.fetch(
                stock_list=["600519", "000001"],
                start_date="2024-01-01",
                end_date="2024-01-14",
                period="1d",
            )

        assert not df.empty
        assert set(df["stock_code"].unique()) == {"600519", "000001"}

    def test_date_filtering(self, mock_connection, mock_client):
        local_df = self._make_local_df(20, "2024-01-01")
        mock_client.get_k_data.return_value = pd.DataFrame()

        source = HybridKlineSource(mock_connection, tdxdir="/tmp")
        with patch.object(source, "_read_local", return_value=local_df):
            df = source.fetch(
                stock_code="600519",
                start_date="2024-01-03",
                end_date="2024-01-10",
                period="1d",
            )

        assert not df.empty
        assert df["date"].min() >= pd.Timestamp("2024-01-03")
        assert df["date"].max() <= pd.Timestamp("2024-01-10")

    def test_dedup_on_merge(self, mock_connection, mock_client):
        local_df = self._make_local_df(10, "2024-01-01")
        overlap_df = self._make_remote_df(5, "2024-01-08")
        mock_client.get_k_data.return_value = overlap_df

        source = HybridKlineSource(mock_connection, tdxdir="/tmp")
        with patch.object(source, "_read_local", return_value=local_df):
            df = source.fetch(
                stock_code="600519",
                start_date="2024-01-01",
                end_date="2024-01-14",
                period="1d",
            )

        assert not df.empty
        date_counts = df["date"].value_counts()
        assert all(c == 1 for c in date_counts)

    def test_vol_renamed_to_volume(self, mock_connection, mock_client):
        local_df = self._make_local_df(5, "2024-01-01")
        mock_client.get_k_data.return_value = pd.DataFrame()

        source = HybridKlineSource(mock_connection, tdxdir="/tmp")
        with patch.object(source, "_read_local", return_value=local_df):
            df = source.fetch(
                stock_code="600519",
                start_date="2024-01-01",
                end_date="2024-01-07",
                period="1d",
            )

        assert "volume" in df.columns


@pytest.mark.skipif(not _has_tdx_data(), reason=SKIP_REASON)
class TestHybridKlineSourceRealData:
    def setup_method(self):
        PluginRegistry.clear()

    def test_fetch_daily_single_stock(self):
        source = _make_source()
        df = source.fetch(stock_code="600519", period="1d")
        assert not df.empty
        assert "stock_code" in df.columns
        assert "date" in df.columns
        assert all(c in df.columns for c in ["open", "high", "low", "close", "volume"])

    def test_fetch_with_date_range(self):
        source = _make_source()
        df = source.fetch(
            stock_code="600519",
            start_date="2024-01-01",
            end_date="2024-12-31",
            period="1d",
        )
        assert not df.empty
        assert df["date"].min() >= pd.Timestamp("2024-01-01")
        assert df["date"].max() <= pd.Timestamp("2024-12-31")
