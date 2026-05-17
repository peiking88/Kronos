import logging
from datetime import datetime

import pandas as pd

from tdxdata.core.connection import TdxConnection
from tdxdata.core.registry import register_source
from tdxdata.sources.base import DataSourceBase

logger = logging.getLogger(__name__)


@register_source("realtime_snapshot")
class RealtimeSnapshotSource(DataSourceBase):
    SNAPSHOT_FIELDS = [
        "stock_code", "name", "datetime",
        "open", "high", "low", "close", "pre_close",
        "volume", "amount",
        "bid_price1", "bid_volume1",
        "bid_price2", "bid_volume2",
        "bid_price3", "bid_volume3",
        "bid_price4", "bid_volume4",
        "bid_price5", "bid_volume5",
        "ask_price1", "ask_volume1",
        "ask_price2", "ask_volume2",
        "ask_price3", "ask_volume3",
        "ask_price4", "ask_volume4",
        "ask_price5", "ask_volume5",
        "turnover_rate",
    ]

    def fetch(self, stock_list: list[str] | None = None, stock_code: str | None = None, **kwargs) -> pd.DataFrame:
        codes = stock_list or ([stock_code] if stock_code else [])
        if not codes:
            raise ValueError("Either stock_list or stock_code must be provided")

        try:
            client = self._connection.client
            result = client.quotes(symbol=codes)
        except Exception as e:
            logger.error(f"Error fetching realtime quotes: {e}")
            return pd.DataFrame()

        if result is None or result.empty:
            return pd.DataFrame()

        result = result.copy()
        result["datetime"] = datetime.now()

        col_map = {
            "code": "stock_code",
            "name": "name",
            "open": "open",
            "high": "high",
            "low": "low",
            "price": "close",
            "pre_close": "pre_close",
            "volume": "volume",
            "amount": "amount",
            "bid1": "bid_price1",
            "ask1": "ask_price1",
            "bid_vol1": "bid_volume1",
            "ask_vol1": "ask_volume1",
            "bid2": "bid_price2",
            "ask2": "ask_price2",
            "bid_vol2": "bid_volume2",
            "ask_vol2": "ask_volume2",
            "bid3": "bid_price3",
            "ask3": "ask_price3",
            "bid_vol3": "bid_volume3",
            "ask_vol3": "ask_volume3",
            "bid4": "bid_price4",
            "ask4": "ask_price4",
            "bid_vol4": "bid_volume4",
            "ask_vol4": "ask_volume4",
            "bid5": "bid_price5",
            "ask5": "ask_price5",
            "bid_vol5": "bid_volume5",
            "ask_vol5": "ask_volume5",
        }
        result = self._normalize_columns(result, col_map)

        for col in self.SNAPSHOT_FIELDS:
            if col not in result.columns:
                result[col] = pd.NA

        return result[self.SNAPSHOT_FIELDS]
