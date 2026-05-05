import logging
from datetime import datetime
from typing import Optional

from tdxdata.sync.state import SyncState
from tdxdata.sync.gap_detector import GapDetector

logger = logging.getLogger(__name__)


class SyncManager:
    def __init__(self, state: Optional[SyncState] = None, gap_detector: Optional[GapDetector] = None):
        self._state = state or SyncState()
        self._gap_detector = gap_detector or GapDetector()

    def get_incremental_range(self, stock_code: str, data_type: str) -> tuple[Optional[str], str]:
        last_sync = self._state.get_last_sync(stock_code, data_type)
        end_date = datetime.now().strftime("%Y-%m-%d")
        if last_sync is None:
            logger.info(f"No previous sync for {stock_code}/{data_type}, full fetch required")
            return None, end_date
        return last_sync, end_date

    def update_sync_state(self, stock_code: str, data_type: str, date_range: Optional[dict] = None) -> None:
        now = datetime.now().strftime("%Y-%m-%d")
        self._state.update_sync(stock_code, data_type, now, date_range)

    def detect_gaps(self, df, freq: str = "B") -> list[str]:
        return self._gap_detector.detect(df, freq)
