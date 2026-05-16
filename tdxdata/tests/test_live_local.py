import os
import pytest
import pandas as pd

from mootdx.reader import Reader

from tdxdata.api import TdxData
from tdxdata.sources.local_kline import DEFAULT_TDXDIR

pytestmark = pytest.mark.local

TDXDIR = os.path.expanduser("~/.local/share/tdxcfv/drive_c/tc/")


def _skip_if_no_data():
    if not os.path.isdir(TDXDIR):
        pytest.skip(f"TDX data directory not found: {TDXDIR}")


class TestLocalDailyKline:
    def test_read_sh600519_daily(self):
        _skip_if_no_data()
        reader = Reader.factory(market="std", tdxdir=TDXDIR)
        df = reader.daily(symbol="600519")
        assert df is not None and not df.empty, "600519 日K线数据为空"
        required = {"open", "high", "low", "close"}
        assert required.issubset(set(df.columns))
        assert all(df["close"] > 0)
        print(f"\n600519(贵州茅台) 日K线 共{len(df)}条")
        print(f"最新: {df.iloc[-1].to_dict()}")

    def test_read_sz000001_daily(self):
        _skip_if_no_data()
        reader = Reader.factory(market="std", tdxdir=TDXDIR)
        df = reader.daily(symbol="000001")
        assert df is not None and not df.empty, "000001 日K线数据为空"
        assert all(df["close"] > 0)
        print(f"\n000001(平安银行) 日K线 共{len(df)}条")

    def test_read_sh000001_index_daily(self):
        _skip_if_no_data()
        reader = Reader.factory(market="std", tdxdir=TDXDIR)
        df = reader.daily(symbol="000001")
        assert df is not None and not df.empty, "上证指数日K线为空"
        print(f"\n上证指数 日K线 共{len(df)}条, 最新收盘: {df.iloc[-1]['close']}")


class TestLocalMinuteKline:
    def test_read_sh600519_1min(self):
        _skip_if_no_data()
        reader = Reader.factory(market="std", tdxdir=TDXDIR)
        df = reader.minute(symbol="600519", suffix=1)
        assert df is not None and not df.empty, "600519 1分钟线数据为空"
        assert "close" in df.columns
        print(f"\n600519 1分钟线 共{len(df)}条")
        if len(df) > 0:
            print(f"最新: {df.iloc[-1].to_dict()}")

    def test_read_sh600519_5min(self):
        _skip_if_no_data()
        reader = Reader.factory(market="std", tdxdir=TDXDIR)
        df = reader.fzline(symbol="600519")
        assert df is not None and not df.empty, "600519 5分钟线数据为空"
        assert "close" in df.columns
        print(f"\n600519 5分钟线 共{len(df)}条")


class TestLocalKlineViaAPI:
    def test_fetch_local_daily_via_api(self):
        _skip_if_no_data()
        api = TdxData()
        try:
            api.connect()
            df = api.fetch_local(stock_code="600519", period="1d", tdxdir=TDXDIR)
            assert df is not None and not df.empty
            assert "stock_code" in df.columns
            assert df["stock_code"].iloc[0] == "600519"
            assert "volume" in df.columns
            assert "date" in df.columns
            print(f"\nAPI fetch_local 日K线: {len(df)}条")
            print(df.tail(3).to_string(index=False))
        finally:
            api.close()

    def test_fetch_local_1min_via_api(self):
        _skip_if_no_data()
        api = TdxData()
        try:
            api.connect()
            df = api.fetch_local(stock_code="600519", period="1m", tdxdir=TDXDIR)
            assert df is not None and not df.empty
            assert "stock_code" in df.columns
            print(f"\nAPI fetch_local 1分钟线: {len(df)}条")
        finally:
            api.close()

    def test_fetch_local_5min_via_api(self):
        _skip_if_no_data()
        api = TdxData()
        try:
            api.connect()
            df = api.fetch_local(stock_code="600519", period="5m", tdxdir=TDXDIR)
            assert df is not None and not df.empty
            assert "stock_code" in df.columns
            print(f"\nAPI fetch_local 5分钟线: {len(df)}条")
        finally:
            api.close()

    def test_fetch_local_multiple_stocks(self):
        _skip_if_no_data()
        api = TdxData()
        try:
            api.connect()
            df = api.fetch_local(
                stock_list=["600519", "000001"],
                period="1d",
                tdxdir=TDXDIR,
            )
            assert df is not None and not df.empty
            assert set(df["stock_code"].unique()) == {"600519", "000001"}
            print(f"\nAPI fetch_local 多股票: {len(df)}条, 股票: {df['stock_code'].unique().tolist()}")
        finally:
            api.close()

    def test_fetch_local_with_default_dir(self):
        if not os.path.isdir(DEFAULT_TDXDIR):
            pytest.skip(f"Default TDX dir not found: {DEFAULT_TDXDIR}")
        api = TdxData()
        try:
            api.connect()
            df = api.fetch_local(stock_code="600519", period="1d")
            assert df is not None and not df.empty
            print(f"\nAPI fetch_local 默认目录: {len(df)}条")
        finally:
            api.close()


class TestLocalKlineWithStorage:
    def test_fetch_local_save_to_csv(self, tmp_path):
        _skip_if_no_data()
        api = TdxData()
        try:
            api.connect()
            df = api.fetch_local(
                stock_code="600519",
                period="1d",
                tdxdir=TDXDIR,
                output="csv",
                output_path=str(tmp_path),
            )
            csv_file = os.path.join(str(tmp_path), "local_kline", "600519.csv")
            assert os.path.exists(csv_file), f"CSV not created: {csv_file}"
            loaded = pd.read_csv(csv_file)
            assert len(loaded) > 0
            print(f"\n本地K线保存CSV: {csv_file}, {len(loaded)}条")
        finally:
            api.close()

    def test_fetch_local_save_to_parquet(self, tmp_path):
        _skip_if_no_data()
        api = TdxData()
        try:
            api.connect()
            df = api.fetch_local(
                stock_code="600519",
                period="1d",
                tdxdir=TDXDIR,
                output="parquet",
                output_path=str(tmp_path),
            )
            pq_file = os.path.join(str(tmp_path), "local_kline", "600519.parquet")
            assert os.path.exists(pq_file), f"Parquet not created: {pq_file}"
            print(f"\n本地K线保存Parquet: {pq_file}")
        finally:
            api.close()
