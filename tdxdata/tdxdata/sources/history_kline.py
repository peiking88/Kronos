import logging
from typing import Optional

import pandas as pd

from tdxdata.core.registry import register_source
from tdxdata.sources.adjust import ADJUST_MAP, apply_adjust
from tdxdata.sources.base import FREQUENCY_MAP, RESAMPLE_MAP, DataSourceBase, resample_kline

logger = logging.getLogger(__name__)


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
        return self._batch_fetch(
            stock_list,
            lambda code: self._fetch_single(code, start_date, end_date,
                                            period, adjust),
            label="kline",
        )

    def _fetch_single(
        self,
        code: str,
        start_date: str,
        end_date: str,
        period: str,
        adjust: Optional[str],
    ) -> pd.DataFrame:
        base_info = RESAMPLE_MAP.get(period)
        if base_info:
            base_df = self._fetch_single(
                code, start_date, end_date, base_info["base"], adjust,
            )
            if base_df is not None and not base_df.empty:
                base_df = resample_kline(base_df, base_info["freq"])
            return base_df

        freq = FREQUENCY_MAP[period]
        client = self._connection.client

        if period in ("1d", "1w", "1mon"):
            df = client.get_k_data(code, start_date, end_date)
        else:
            df = client.bars(symbol=code, frequency=freq, offset=800)

        if df is None or df.empty:
            return pd.DataFrame()

        df = self._normalize_kline_df(df, code)

        if adjust:
            df = apply_adjust(df, code, adjust,
                              quotes_client=self._connection.client)

        return df
