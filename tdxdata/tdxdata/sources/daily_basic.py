import logging
import os

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
        result["stock_code"] = str(stock_code)

        if "date" not in result.columns and all(c in result.columns for c in ("year", "month", "day")):
            result["date"] = pd.to_datetime(
                result["year"].astype(str) + "-" +
                result["month"].astype(str).str.zfill(2) + "-" +
                result["day"].astype(str).str.zfill(2)
            )
            result.drop(columns=["year", "month", "day"], inplace=True)

        if "date" in result.columns:
            result["date"] = pd.to_datetime(result["date"])

        output_path = kwargs.get("output_path")
        if output_path:
            os.makedirs(output_path, exist_ok=True)
            result.to_csv(os.path.join(output_path, f"{stock_code}.csv"), index=False)

        return result
