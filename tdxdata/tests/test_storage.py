import os
import pytest
import numpy as np
import pandas as pd

from tdxdata.core.registry import PluginRegistry
from tdxdata.storage.csv import CSVStorage
from tdxdata.storage.sqlite import SQLiteStorage
from tdxdata.storage.parquet import ParquetStorage


class TestCSVStorage:
    def setup_method(self):
        PluginRegistry.clear()

    def test_save_and_load(self, tmp_output_dir):
        df = pd.DataFrame({
            "stock_code": ["600519.SH"] * 3,
            "date": pd.date_range("2024-01-01", periods=3),
            "close": [100.0, 101.0, 102.0],
        })
        storage = CSVStorage(output_path=tmp_output_dir)
        path = storage.save(df, source="kline", stock_list=["600519.SH"])
        assert os.path.exists(path)

        loaded = storage.load(file_path=path)
        assert len(loaded) == 3
        assert "stock_code" in loaded.columns

    def test_save_creates_directory(self, tmp_output_dir):
        nested = os.path.join(tmp_output_dir, "nested", "dir")
        df = pd.DataFrame({"a": [1]})
        storage = CSVStorage(output_path=nested)
        path = storage.save(df, source="test", stock_list=["TEST.SH"])
        assert os.path.exists(path)

    def test_load_nonexistent_raises(self, tmp_output_dir):
        storage = CSVStorage(output_path=tmp_output_dir)
        with pytest.raises(FileNotFoundError):
            storage.load(file_path="/nonexistent/file.csv")


class TestSQLiteStorage:
    def setup_method(self):
        PluginRegistry.clear()

    def test_save_and_load(self, tmp_output_dir):
        df = pd.DataFrame({
            "stock_code": ["600519.SH"] * 3,
            "date": pd.date_range("2024-01-01", periods=3),
            "close": [100.0, 101.0, 102.0],
        })
        storage = SQLiteStorage(output_path=tmp_output_dir)
        storage.save(df, source="kline", stock_list=["600519.SH"])

        loaded = storage.load(source="kline", stock_list=["600519.SH"])
        assert len(loaded) == 3

    def test_save_overwrites_existing(self, tmp_output_dir):
        df1 = pd.DataFrame({"close": [100.0]})
        df2 = pd.DataFrame({"close": [200.0, 300.0]})

        storage = SQLiteStorage(output_path=tmp_output_dir)
        storage.save(df1, source="kline", stock_list=["600519.SH"])
        storage.save(df2, source="kline", stock_list=["600519.SH"])

        loaded = storage.load(source="kline", stock_list=["600519.SH"])
        assert len(loaded) == 2

    def test_load_nonexistent_raises(self, tmp_output_dir):
        storage = SQLiteStorage(output_path=tmp_output_dir)
        with pytest.raises(FileNotFoundError):
            storage.load(source="kline", stock_list=["NONEXISTENT.SH"])


class TestParquetStorage:
    def setup_method(self):
        PluginRegistry.clear()

    def test_save_and_load(self, tmp_output_dir):
        df = pd.DataFrame({
            "stock_code": ["600519.SH"] * 3,
            "date": pd.date_range("2024-01-01", periods=3),
            "close": [100.0, 101.0, 102.0],
        })
        storage = ParquetStorage(output_path=tmp_output_dir)
        path = storage.save(df, source="kline", stock_list=["600519.SH"])
        assert os.path.exists(path)

        loaded = storage.load(file_path=path)
        assert len(loaded) == 3

    def test_load_nonexistent_raises(self, tmp_output_dir):
        storage = ParquetStorage(output_path=tmp_output_dir)
        with pytest.raises(FileNotFoundError):
            storage.load(file_path="/nonexistent/file.parquet")

    def test_round_trip_preserves_types(self, tmp_output_dir):
        df = pd.DataFrame({
            "stock_code": ["600519.SH"],
            "date": pd.to_datetime(["2024-01-01"]),
            "close": [100.0],
            "volume": [10000.0],
        })
        storage = ParquetStorage(output_path=tmp_output_dir)
        path = storage.save(df, source="kline", stock_list=["600519.SH"])
        loaded = storage.load(file_path=path)

        assert loaded["close"].dtype == np.float64
        assert pd.api.types.is_datetime64_any_dtype(loaded["date"])
