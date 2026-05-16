import logging
from typing import Optional

import pandas as pd

from tdxdata.core.connection import TdxConnection
from tdxdata.core.registry import register_source
from tdxdata.sources.adjust import ADJUST_MAP, apply_adjust
from tdxdata.sources.base import DataSourceBase

logger = logging.getLogger(__name__)

FREQUENCY_MAP = {
    "1m": 8,
    "5m": 0,
    "15m": 1,
    "30m": 2,
    "1h": 3,
    "1d": 9,
    "1w": 5,
    "1mon": 6,
}


@register_source("history_kline")
class HistoryKlineSource(DataSourceBase):
    def fetch(
        self,
        stock_list: list[str],
        start_date: str,
        end_date: str,
        period: str = "1d",
        dividend_type: str = "front",
        **kwargs,
    ) -> pd.DataFrame:
        if period not in FREQUENCY_MAP:
            raise ValueError(
                f"Unsupported period '{period}'. "
                f"Supported: {sorted(FREQUENCY_MAP.keys())}"
            )

        adjust = ADJUST_MAP.get(dividend_type)
        result_parts = []

        for code in stock_list:
            try:
                df = self._fetch_single(code, start_date, end_date, period, adjust)
                if df is not None and not df.empty:
                    result_parts.append(df)
            except Exception as e:
                logger.error(f"Error fetching kline for {code}: {e}")

        if not result_parts:
            return pd.DataFrame()

        final_df = pd.concat(result_parts, ignore_index=True)
        return final_df

    def _fetch_single(
        self,
        code: str,
        start_date: str,
        end_date: str,
        period: str,
        adjust: Optional[str],
    ) -> pd.DataFrame:
        freq = FREQUENCY_MAP[period]
        client = self._connection.client

        if period in ("1d", "1w", "1mon"):
            df = client.get_k_data(code, start_date, end_date)
        else:
            df = client.bars(symbol=code, frequency=freq, offset=800)

        if df is None or df.empty:
            return pd.DataFrame()

        df = df.copy()
        df["stock_code"] = code

        if adjust and period in ("1d", "1w", "1mon"):
            df = apply_adjust(df, code, adjust)

        col_map = {}
        if "open" in df.columns:
            col_map["open"] = "open"
        if "high" in df.columns:
            col_map["high"] = "high"
        if "low" in df.columns:
            col_map["low"] = "low"
        if "close" in df.columns:
            col_map["close"] = "close"
        if "vol" in df.columns:
            col_map["vol"] = "volume"
        if "volume" not in df.columns and "vol" in df.columns:
            df["volume"] = df["vol"]
        if "amount" in df.columns:
            col_map["amount"] = "amount"

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        elif "datetime" in df.columns:
            df["date"] = pd.to_datetime(df["datetime"])

        keep_cols = ["stock_code", "date", "open", "high", "low", "close", "volume", "amount"]
        keep_cols = [c for c in keep_cols if c in df.columns]
        extra_cols = [c for c in df.columns if c not in keep_cols]
        return df[keep_cols + extra_cols]
