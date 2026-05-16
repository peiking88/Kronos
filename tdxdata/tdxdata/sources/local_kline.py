import logging
import os
from typing import Optional

import pandas as pd

from mootdx.reader import Reader

from tdxdata.core.connection import TdxConnection
from tdxdata.core.registry import register_source
from tdxdata.sources.adjust import ADJUST_MAP, apply_adjust
from tdxdata.sources.base import DataSourceBase

logger = logging.getLogger(__name__)

DEFAULT_TDXDIR = os.path.expanduser("~/.local/share/tdxcfv/drive_c/tc/")

PERIOD_METHOD = {
    "1d": "daily",
    "1m": "minute_1",
    "5m": "minute_5",
}


@register_source("local_kline")
class LocalKlineSource(DataSourceBase):
    def __init__(self, connection: TdxConnection, tdxdir: Optional[str] = None):
        super().__init__(connection)
        self._tdxdir = tdxdir or DEFAULT_TDXDIR
        self._reader = None

    def _get_reader(self):
        if self._reader is None:
            tdxdir = self._tdxdir
            if not os.path.isdir(tdxdir):
                raise FileNotFoundError(
                    f"TDX directory not found: {tdxdir}. "
                    f"Set tdxdir parameter or ensure default path exists."
                )
            self._reader = Reader.factory(market="std", tdxdir=tdxdir)
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

        if period not in PERIOD_METHOD:
            raise ValueError(
                f"Unsupported period '{period}'. "
                f"Supported: {sorted(PERIOD_METHOD.keys())}"
            )

        if tdxdir:
            self._tdxdir = tdxdir
            self._reader = None

        adjust = ADJUST_MAP.get(dividend_type)
        reader = self._get_reader()
        result_parts = []

        for code in codes:
            try:
                df = self._fetch_single(reader, code, period, adjust)
                if df is not None and not df.empty:
                    result_parts.append(df)
            except Exception as e:
                logger.error(f"Error reading local data for {code}: {e}")

        if not result_parts:
            return pd.DataFrame()

        return pd.concat(result_parts, ignore_index=True)

    def _fetch_single(
        self, reader, code: str, period: str, adjust: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        method = PERIOD_METHOD[period]

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

        df = df.copy()

        if df.index.name in ("date", "datetime"):
            df.reset_index(inplace=True)

        df["stock_code"] = code

        if adjust and period == "1d":
            df = apply_adjust(df, code, adjust)

        col_map = {}
        if "vol" in df.columns:
            col_map["vol"] = "volume"
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        elif "datetime" in df.columns:
            df["date"] = pd.to_datetime(df["datetime"], errors="coerce")

        df = self._normalize_columns(df, col_map)

        keep_cols = ["stock_code", "date", "open", "high", "low", "close", "volume", "amount"]
        keep_cols = [c for c in keep_cols if c in df.columns]
        extra_cols = [c for c in df.columns if c not in keep_cols]
        return df[keep_cols + extra_cols]
