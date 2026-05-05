import os
import pytest
import pandas as pd

from tdxdata.core.registry import PluginRegistry
from tdxdata.sources.local_kline import LocalKlineSource, DEFAULT_TDXDIR

TDXDIR = os.path.expanduser("~/.local/share/tdxcfv/drive_c/tc/")

SKIP_REASON = f"TDX directory not found: {TDXDIR}"


def _has_tdx_data():
    return os.path.isdir(TDXDIR)


def _make_source():
    from tdxdata.core.connection import TdxConnection
    conn = TdxConnection()
    conn._initialized = True
    return LocalKlineSource(conn, tdxdir=TDXDIR)


@pytest.mark.skipif(not _has_tdx_data(), reason=SKIP_REASON)
class TestLocalKlineSourceRealData:
    def setup_method(self):
        PluginRegistry.clear()

    def test_fetch_daily_single_stock(self):
        source = _make_source()
        df = source.fetch(stock_code="600519", period="1d")

        assert not df.empty
        assert "stock_code" in df.columns
        assert "date" in df.columns
        assert "open" in df.columns
        assert "high" in df.columns
        assert "low" in df.columns
        assert "close" in df.columns
        assert "volume" in df.columns
        assert df["stock_code"].iloc[0] == "600519"

    def test_fetch_daily_multiple_stocks(self):
        source = _make_source()
        df = source.fetch(stock_list=["600519", "000001"], period="1d")

        assert not df.empty
        assert set(df["stock_code"].unique()) == {"600519", "000001"}

    def test_fetch_1m_minute(self):
        source = _make_source()
        df = source.fetch(stock_code="600519", period="1m")

        assert not df.empty
        assert "stock_code" in df.columns
        assert "date" in df.columns
        assert "open" in df.columns

    def test_fetch_5m_minute(self):
        source = _make_source()
        df = source.fetch(stock_code="600519", period="5m")

        assert not df.empty
        assert "stock_code" in df.columns
        assert "date" in df.columns

    def test_fetch_sz_stock(self):
        source = _make_source()
        df = source.fetch(stock_code="000001", period="1d")

        assert not df.empty
        assert df["stock_code"].iloc[0] == "000001"

    def test_fetch_with_error_continues(self):
        source = _make_source()
        df = source.fetch(stock_list=["000000", "600519"], period="1d")

        assert not df.empty
        assert len(df["stock_code"].unique()) == 1
        assert df["stock_code"].iloc[0] == "600519"

    def test_invalid_period_raises(self):
        source = _make_source()
        with pytest.raises(ValueError, match="Unsupported period"):
            source.fetch(stock_code="600519", period="1h")

    def test_no_stock_code_raises(self):
        source = _make_source()
        with pytest.raises(ValueError, match="stock_list or stock_code"):
            source.fetch()

    def test_tdxdir_not_found_raises(self):
        from tdxdata.core.connection import TdxConnection
        conn = TdxConnection()
        conn._initialized = True
        source = LocalKlineSource(conn, tdxdir="/nonexistent/path")

        with pytest.raises(FileNotFoundError, match="TDX directory not found"):
            source.fetch(stock_code="600519", period="1d")

    def test_tdxdir_override_in_fetch(self):
        source = _make_source()
        df = source.fetch(stock_code="600519", period="1d", tdxdir=TDXDIR)

        assert not df.empty
        assert source._tdxdir == TDXDIR


class TestLocalKlineSourceUnit:
    def setup_method(self):
        PluginRegistry.clear()

    def test_default_tdxdir(self):
        assert DEFAULT_TDXDIR.endswith("/tc/")
        assert ".local/share/tdxcfv/drive_c/tc/" in DEFAULT_TDXDIR
