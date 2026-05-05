import logging

import pandas as pd

from tdxdata.core.connection import TdxConnection
from tdxdata.core.registry import register_source
from tdxdata.sources.base import DataSourceBase

logger = logging.getLogger(__name__)


@register_source("tick")
class TickDataSource(DataSourceBase):
    def fetch(self, stock_code: str, date: str | None = None, **kwargs) -> pd.DataFrame:
        try:
            client = self._connection.client
            if date:
                result = client.transactions(
                    symbol=stock_code, start=0, offset=2000, date=date
                )
            else:
                result = client.transaction(symbol=stock_code, start=0, offset=2000)
        except Exception as e:
            logger.error(f"Error fetching tick data for {stock_code}: {e}")
            return pd.DataFrame()

        if result is None or result.empty:
            return pd.DataFrame()

        result = result.copy()
        result["stock_code"] = stock_code

        col_map = {
            "time": "datetime",
            "price": "price",
            "vol": "volume",
            "num": "order_id",
            "buyorsell": "buy_sell_flag",
        }
        result = self._normalize_columns(result, col_map)

        if "datetime" in result.columns:
            result["datetime"] = pd.to_datetime(result["datetime"], format="mixed")

        keep = ["stock_code", "datetime", "price", "volume"]
        if "buy_sell_flag" in result.columns:
            keep.append("buy_sell_flag")
        if "amount" in result.columns:
            keep.append("amount")
        keep = [c for c in keep if c in result.columns]
        return result[keep]
