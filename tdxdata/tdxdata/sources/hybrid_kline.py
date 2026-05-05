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

STANDARD_COLUMNS = ["stock_code", "date", "open", "high", "low", "close", "volume", "amount"]


@register_source("hybrid_kline")
class HybridKlineSource(DataSourceBase):
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
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
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
        result_parts = []

        for code in codes:
            try:
                df = self._fetch_hybrid(code, start_date, end_date, period, adjust)
                if df is not None and not df.empty:
                    result_parts.append(df)
            except Exception as e:
                logger.error(f"Error in hybrid fetch for {code}: {e}")

        if not result_parts:
            return pd.DataFrame()

        return pd.concat(result_parts, ignore_index=True)

    def _fetch_hybrid(
        self,
        code: str,
        start_date: Optional[str],
        end_date: Optional[str],
        period: str,
        adjust: Optional[str],
    ) -> Optional[pd.DataFrame]:
        local_df = self._read_local(code, period)

        if local_df is None or local_df.empty:
            logger.info(f"No local data for {code}, fetching all from network")
            return self._fetch_remote(code, start_date, end_date, period, adjust)

        local_df = self._normalize(local_df, code)

        local_end = local_df["date"].max()
        local_start = local_df["date"].min()

        if start_date:
            start_dt = pd.Timestamp(start_date)
        else:
            start_dt = local_start

        if end_date:
            end_dt = pd.Timestamp(end_date)
        else:
            end_dt = pd.Timestamp.now().normalize()

        need_remote = False
        remote_start = None
        remote_end = None

        if local_start > start_dt:
            need_remote = True
            remote_start = start_dt.strftime("%Y-%m-%d")
            remote_end = (local_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            logger.info(
                f"Local data starts at {local_start.date()}, "
                f"fetching earlier data from {remote_start} to {remote_end}"
            )

        if local_end < end_dt:
            need_remote = True
            remote_start = (local_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            remote_end = end_dt.strftime("%Y-%m-%d")
            logger.info(
                f"Local data ends at {local_end.date()}, "
                f"fetching newer data from {remote_start} to {remote_end}"
            )

        if not need_remote:
            df = local_df
            if start_date:
                df = df[df["date"] >= start_dt]
            if end_date:
                df = df[df["date"] <= end_dt]
            df = df.reset_index(drop=True)
            if adjust and period == "1d":
                df = apply_adjust(df, code, adjust)
            return df

        remote_df = self._fetch_remote(code, remote_start, remote_end, period, adjust=None)

        if remote_df is not None and not remote_df.empty:
            remote_df = self._normalize(remote_df, code)
            combined = pd.concat([local_df, remote_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=["date"], keep="last")
            combined = combined.sort_values("date").reset_index(drop=True)
        else:
            combined = local_df

        if start_date:
            combined = combined[combined["date"] >= start_dt]
        if end_date:
            combined = combined[combined["date"] <= end_dt]
        combined = combined.reset_index(drop=True)

        if adjust and period == "1d":
            combined = apply_adjust(combined, code, adjust)

        return combined

    def _read_local(self, code: str, period: str) -> Optional[pd.DataFrame]:
        try:
            reader = self._get_reader()
        except FileNotFoundError:
            logger.info("TDX directory not found, skipping local read")
            return None

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
        return df

    def _fetch_remote(
        self,
        code: str,
        start_date: Optional[str],
        end_date: Optional[str],
        period: str,
        adjust: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        if start_date is None and end_date is None:
            return None

        if period not in FREQUENCY_MAP:
            return None

        try:
            freq = FREQUENCY_MAP[period]
            client = self._connection.client
        except Exception as e:
            logger.warning(f"Remote connection unavailable: {e}")
            return None

        try:
            if period in ("1d", "1w", "1mon"):
                df = client.get_k_data(code, start_date, end_date)
            else:
                df = client.bars(symbol=code, frequency=freq, offset=800)
        except Exception as e:
            logger.warning(f"Remote fetch failed for {code}: {e}")
            return None

        if df is None or df.empty:
            return None

        return df.copy()

    def _normalize(self, df: pd.DataFrame, code: str) -> pd.DataFrame:
        df["stock_code"] = code

        if "vol" in df.columns and "volume" not in df.columns:
            df["volume"] = df["vol"]

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
        elif "datetime" in df.columns:
            df["date"] = pd.to_datetime(df["datetime"], errors="coerce")

        keep = [c for c in STANDARD_COLUMNS if c in df.columns]
        extra = [c for c in df.columns if c not in keep]
        return df[keep + extra].copy()
