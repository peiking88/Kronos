import logging

import pandas as pd

from tdxdata.core.connection import TdxConnection
from tdxdata.core.registry import register_source
from tdxdata.sources.base import DataSourceBase

logger = logging.getLogger(__name__)


@register_source("daily_basic")
class DailyBasicSource(DataSourceBase):
    def fetch(self, stock_code: str, date: str | None = None, **kwargs) -> pd.DataFrame:
        try:
            client = self._connection.client
            result = client.xdxr(symbol=stock_code)
        except Exception as e:
            logger.error(f"Error fetching daily basic for {stock_code}: {e}")
            return pd.DataFrame()

        if result is None or result.empty:
            return pd.DataFrame()

        result = result.copy()
        result["stock_code"] = stock_code

        col_map = {
            "code": "stock_code",
        }
        result = self._normalize_columns(result, col_map)

        if "date" in result.columns:
            result["date"] = pd.to_datetime(result["date"])

        return result
