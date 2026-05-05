import json
import os
import logging
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_STATE_PATH = os.path.expanduser("~/.tdxdata/sync_state.json")


class SyncState:
    def __init__(self, state_path: str = DEFAULT_STATE_PATH):
        self._state_path = state_path
        self._state: dict = {}

    @property
    def state_path(self) -> str:
        return self._state_path

    def load(self) -> dict:
        if os.path.exists(self._state_path):
            with open(self._state_path, "r", encoding="utf-8") as f:
                self._state = json.load(f)
        else:
            self._state = {}
        return self._state

    def save(self) -> None:
        dir_path = os.path.dirname(self._state_path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        with open(self._state_path, "w", encoding="utf-8") as f:
            json.dump(self._state, f, indent=2, ensure_ascii=False, default=str)
        logger.info(f"Sync state saved to: {self._state_path}")

    def get_last_sync(self, stock_code: str, data_type: str) -> Optional[str]:
        if not self._state:
            self.load()
        entry = self._state.get(stock_code, {}).get(data_type, {})
        return entry.get("last_sync")

    def update_sync(self, stock_code: str, data_type: str, last_sync: str, date_range: Optional[dict] = None) -> None:
        if not self._state:
            self.load()
        if stock_code not in self._state:
            self._state[stock_code] = {}
        self._state[stock_code][data_type] = {
            "last_sync": last_sync,
            "updated_at": datetime.now().isoformat(),
        }
        if date_range:
            self._state[stock_code][data_type]["date_range"] = date_range
        self.save()

    def clear(self) -> None:
        self._state = {}
        if os.path.exists(self._state_path):
            os.remove(self._state_path)
        logger.info("Sync state cleared")
