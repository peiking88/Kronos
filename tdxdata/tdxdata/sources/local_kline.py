import logging
from typing import Optional

import pandas as pd

from tdxdata.core.connection import TdxConnection
from tdxdata.core.registry import register_source
from tdxdata.sources.base import (
    DEFAULT_TDXDIR, PERIOD_MAP, DataSourceBase, get_tdx_reader,
)

logger = logging.getLogger(__name__)

_LOCAL_PERIODS = {"1d", "1m", "5m"}


@register_source("local_kline")
class LocalKlineSource(DataSourceBase):
    def __init__(self, connection: TdxConnection, tdxdir: Optional[str] = None):
        super().__init__(connection)
        self._tdxdir = tdxdir or DEFAULT_TDXDIR
        self._reader = None

    def _get_reader(self):
        if self._reader is None:
            self._reader = get_tdx_reader(self._tdxdir)
        return self._reader

    def fetch(
        self,
        stock_list: Optional[list[str]] = None,
        stock_code: Optional[str] = None,
        period: str = "1d",
        tdxdir: Optional[str] = None,
        dividend_type: str = "none",
        **kwargs,
    ) -> pd.DataFrame:
        codes = stock_list or ([stock_code] if stock_code else [])
        if not codes:
            raise ValueError("Either stock_list or stock_code must be provided")

        if period not in _LOCAL_PERIODS:
            raise ValueError(
                f"Unsupported period '{period}'. "
                f"Supported: {sorted(_LOCAL_PERIODS)}"
            )

        if tdxdir:
            self._tdxdir = tdxdir
            self._reader = None

        reader = self._get_reader()
        return self._batch_fetch(
            codes,
            lambda code: self._fetch_single(reader, code, period),
            label="local_kline",
        )

    def _fetch_single(
        self, reader, code: str, period: str
    ) -> Optional[pd.DataFrame]:
        method = PERIOD_MAP[period]

        if method == "daily":
            df = reader.daily(symbol=code)
        elif method == "minute_1":
            df = reader.minute(symbol=code, suffix=1)
        elif method == "minute_5":
            df = reader.fzline(symbol=code)
        else:
            return None

        if df is None or df.empty:
            return None

        return self._normalize_kline_df(df, code)
