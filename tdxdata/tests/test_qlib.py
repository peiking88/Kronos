import os

import numpy as np
import pandas as pd
import pytest

from tdxdata.qlib.qlib_bin import (
    build_calendar,
    build_instruments,
    df_to_qlib_bins,
    normalize_symbol,
    write_bin_file,
    write_calendar,
    write_instruments,
)


def _make_df(n=10, code="600519"):
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    return pd.DataFrame({
        "stock_code": [code] * n,
        "date": dates,
        "open": np.random.uniform(100, 110, n).astype(np.float32),
        "high": np.random.uniform(110, 120, n).astype(np.float32),
        "low": np.random.uniform(90, 100, n).astype(np.float32),
        "close": np.random.uniform(100, 115, n).astype(np.float32),
        "volume": np.random.uniform(10000, 50000, n).astype(np.float32),
        "factor": np.ones(n, dtype=np.float32),
    })


class TestNormalizeSymbol:
    def test_sh_stock(self):
        assert normalize_symbol("600519") == "SH600519"

    def test_sz_stock(self):
        assert normalize_symbol("000001") == "SZ000001"

    def test_cy_stock(self):
        assert normalize_symbol("300001") == "SZ300001"

    def test_kc_stock(self):
        assert normalize_symbol("688001") == "SH688001"

    def test_bj_stock(self):
        assert normalize_symbol("430001") == "BJ430001"

    def test_already_prefixed(self):
        assert normalize_symbol("SH600519") == "SH600519"
        assert normalize_symbol("sz000001") == "SZ000001"

    def test_etf(self):
        assert normalize_symbol("510050") == "SH510050"


class TestWriteBinFile:
    def test_write_and_read(self, tmp_path):
        data = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        filepath = str(tmp_path / "test" / "data.bin")
        write_bin_file(data, filepath)

        assert os.path.exists(filepath)
        loaded = np.fromfile(filepath, dtype=np.float32)
        np.testing.assert_array_equal(loaded, data)

    def test_creates_directory(self, tmp_path):
        filepath = str(tmp_path / "deep" / "nested" / "dir" / "data.bin")
        write_bin_file(np.array([1.0], dtype=np.float32), filepath)
        assert os.path.exists(filepath)


class TestWriteCalendar:
    def test_write_calendar(self, tmp_path):
        dates = ["2024-01-02", "2024-01-03", "2024-01-04"]
        filepath = str(tmp_path / "cal" / "day.txt")
        write_calendar(dates, filepath)

        assert os.path.exists(filepath)
        with open(filepath) as f:
            lines = [l.strip() for l in f.readlines()]
        assert lines == dates


class TestWriteInstruments:
    def test_write_instruments(self, tmp_path):
        instruments = [
            ("SH600519", "2024-01-02", "2024-12-31"),
            ("SZ000001", "2024-01-02", "2024-12-31"),
        ]
        filepath = str(tmp_path / "inst" / "all.txt")
        write_instruments(instruments, filepath)

        assert os.path.exists(filepath)
        with open(filepath) as f:
            lines = f.readlines()
        assert len(lines) == 2
        assert "SH600519\t2024-01-02\t2024-12-31" in lines[0]


class TestDfToQlibBins:
    def test_single_stock(self, tmp_path):
        df = _make_df(10)
        qlib_dir = str(tmp_path / "qlib")
        written = df_to_qlib_bins(df, qlib_dir)

        assert len(written) > 0
        feat_dir = os.path.join(qlib_dir, "features", "SH600519")
        assert os.path.isdir(feat_dir)

        close_file = os.path.join(feat_dir, "close.day.bin")
        assert os.path.exists(close_file)
        data = np.fromfile(close_file, dtype=np.float32)
        assert len(data) == 10

    def test_multiple_stocks(self, tmp_path):
        df1 = _make_df(5, "600519")
        df2 = _make_df(8, "000001")
        df = pd.concat([df1, df2], ignore_index=True)

        qlib_dir = str(tmp_path / "qlib")
        df_to_qlib_bins(df, qlib_dir)

        assert os.path.isdir(os.path.join(qlib_dir, "features", "SH600519"))
        assert os.path.isdir(os.path.join(qlib_dir, "features", "SZ000001"))

    def test_custom_fields(self, tmp_path):
        df = _make_df(5)
        qlib_dir = str(tmp_path / "qlib")
        df_to_qlib_bins(df, qlib_dir, fields=["open", "close"])

        feat_dir = os.path.join(qlib_dir, "features", "SH600519")
        assert os.path.exists(os.path.join(feat_dir, "open.day.bin"))
        assert os.path.exists(os.path.join(feat_dir, "close.day.bin"))
        assert not os.path.exists(os.path.join(feat_dir, "high.day.bin"))

    def test_missing_columns_raises(self, tmp_path):
        df = pd.DataFrame({"a": [1]})
        with pytest.raises(ValueError, match="stock_code"):
            df_to_qlib_bins(df, str(tmp_path))

    def test_minute_freq(self, tmp_path):
        df = _make_df(5)
        qlib_dir = str(tmp_path / "qlib")
        df_to_qlib_bins(df, qlib_dir, freq="1min")

        close_file = os.path.join(
            qlib_dir, "features", "SH600519", "close.1min.bin"
        )
        assert os.path.exists(close_file)


class TestBuildCalendar:
    def test_build_calendar(self, tmp_path):
        df = _make_df(10)
        qlib_dir = str(tmp_path / "qlib")
        cal_path = build_calendar(df, qlib_dir)

        assert os.path.exists(cal_path)
        with open(cal_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 10
        assert lines == sorted(lines)

    def test_dedup_dates(self, tmp_path):
        df1 = _make_df(5, "600519")
        df2 = _make_df(5, "000001")
        df = pd.concat([df1, df2], ignore_index=True)

        qlib_dir = str(tmp_path / "qlib")
        cal_path = build_calendar(df, qlib_dir)

        with open(cal_path) as f:
            lines = [l.strip() for l in f if l.strip()]
        assert len(lines) == 5


class TestBuildInstruments:
    def test_build_instruments(self, tmp_path):
        df1 = _make_df(5, "600519")
        df2 = _make_df(8, "000001")
        df = pd.concat([df1, df2], ignore_index=True)

        qlib_dir = str(tmp_path / "qlib")
        inst_path = build_instruments(df, qlib_dir)

        assert os.path.exists(inst_path)
        with open(inst_path) as f:
            lines = f.readlines()
        assert len(lines) == 2

        symbols = [l.split("\t")[0] for l in lines]
        assert "SH600519" in symbols
        assert "SZ000001" in symbols

    def test_custom_instrument_name(self, tmp_path):
        df = _make_df(5)
        qlib_dir = str(tmp_path / "qlib")
        inst_path = build_instruments(df, qlib_dir, instrument_name="csi300")

        assert inst_path.endswith("csi300.txt")
        assert os.path.exists(inst_path)


class TestQlibStorage:
    def test_save_and_load(self, tmp_path):
        from tdxdata.storage.qlib import QlibStorage

        df = _make_df(10)
        storage = QlibStorage(output_path=str(tmp_path / "qlib"))
        result_path = storage.save(df)

        assert os.path.isdir(result_path)

        loaded = storage.load(symbol="SH600519")
        assert not loaded.empty
        assert "close" in loaded.columns
        assert "open" in loaded.columns
        assert len(loaded) == 10

    def test_save_creates_structure(self, tmp_path):
        from tdxdata.storage.qlib import QlibStorage

        df = _make_df(5)
        qlib_dir = str(tmp_path / "qlib")
        storage = QlibStorage(output_path=qlib_dir)
        storage.save(df)

        assert os.path.exists(os.path.join(qlib_dir, "calendars", "day.txt"))
        assert os.path.exists(os.path.join(qlib_dir, "instruments", "all.txt"))
        assert os.path.isdir(os.path.join(qlib_dir, "features", "SH600519"))

    def test_load_missing_symbol_raises(self, tmp_path):
        from tdxdata.storage.qlib import QlibStorage

        storage = QlibStorage(output_path=str(tmp_path))
        with pytest.raises(FileNotFoundError):
            storage.load(symbol="SH999999")

    def test_load_no_symbol_raises(self, tmp_path):
        from tdxdata.storage.qlib import QlibStorage

        storage = QlibStorage(output_path=str(tmp_path))
        with pytest.raises(ValueError, match="symbol"):
            storage.load()
