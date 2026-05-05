import os
import json
import pytest
import pandas as pd

from tdxdata.sync.state import SyncState
from tdxdata.sync.gap_detector import GapDetector
from tdxdata.sync.manager import SyncManager


class TestSyncState:
    def test_save_and_load(self, tmp_path):
        state_path = str(tmp_path / "sync_state.json")
        state = SyncState(state_path=state_path)
        state.update_sync("600519.SH", "kline_1d", "2024-01-05")

        state2 = SyncState(state_path=state_path)
        result = state2.get_last_sync("600519.SH", "kline_1d")
        assert result == "2024-01-05"

    def test_get_last_sync_no_state(self, tmp_path):
        state_path = str(tmp_path / "nonexistent.json")
        state = SyncState(state_path=state_path)
        assert state.get_last_sync("600519.SH", "kline_1d") is None

    def test_clear(self, tmp_path):
        state_path = str(tmp_path / "sync_state.json")
        state = SyncState(state_path=state_path)
        state.update_sync("600519.SH", "kline_1d", "2024-01-05")
        state.clear()
        assert not os.path.exists(state_path)

    def test_date_range_stored(self, tmp_path):
        state_path = str(tmp_path / "sync_state.json")
        state = SyncState(state_path=state_path)
        state.update_sync("600519.SH", "kline_1d", "2024-01-05",
                          date_range={"start": "2024-01-01", "end": "2024-01-05"})

        with open(state_path) as f:
            data = json.load(f)
        assert data["600519.SH"]["kline_1d"]["date_range"]["start"] == "2024-01-01"


class TestGapDetector:
    def test_no_gaps(self):
        df = pd.DataFrame({
            "date": pd.date_range("2024-01-01", periods=5, freq="B"),
            "close": [100.0] * 5,
        })
        detector = GapDetector()
        gaps = detector.detect(df)
        assert gaps == []

    def test_detect_gaps(self):
        dates = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05"])
        df = pd.DataFrame({
            "date": dates,
            "close": [100.0, 101.0, 102.0],
        })
        detector = GapDetector()
        gaps = detector.detect(df)
        assert len(gaps) > 0

    def test_empty_dataframe(self):
        df = pd.DataFrame()
        detector = GapDetector()
        gaps = detector.detect(df)
        assert gaps == []

    def test_single_date(self):
        df = pd.DataFrame({
            "date": pd.to_datetime(["2024-01-02"]),
            "close": [100.0],
        })
        detector = GapDetector()
        gaps = detector.detect(df)
        assert gaps == []


class TestSyncManager:
    def test_get_incremental_range_no_previous(self, tmp_path):
        state_path = str(tmp_path / "sync_state.json")
        state = SyncState(state_path=state_path)
        manager = SyncManager(state=state)
        start, end = manager.get_incremental_range("600519.SH", "kline_1d")
        assert start is None
        assert end is not None

    def test_get_incremental_range_with_previous(self, tmp_path):
        state_path = str(tmp_path / "sync_state.json")
        state = SyncState(state_path=state_path)
        state.update_sync("600519.SH", "kline_1d", "2024-01-05")

        manager = SyncManager(state=state)
        start, end = manager.get_incremental_range("600519.SH", "kline_1d")
        assert start == "2024-01-05"

    def test_update_sync_state(self, tmp_path):
        state_path = str(tmp_path / "sync_state.json")
        state = SyncState(state_path=state_path)
        manager = SyncManager(state=state)
        manager.update_sync_state("600519.SH", "kline_1d")

        result = state.get_last_sync("600519.SH", "kline_1d")
        assert result is not None
