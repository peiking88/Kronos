import logging
from typing import Optional

import pandas as pd

from tdxdata.core.connection import TdxConnection
from tdxdata.core.registry import register_source
from tdxdata.sources.adjust import ADJUST_MAP, apply_adjust
from tdxdata.sources.base import (
    DEFAULT_TDXDIR, FREQUENCY_MAP, PERIOD_MAP, RESAMPLE_MAP,
    DataSourceBase, get_tdx_reader, resample_kline,
)

logger = logging.getLogger(__name__)


@register_source("hybrid_kline")
class HybridKlineSource(DataSourceBase):
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

        if period not in FREQUENCY_MAP:
            raise ValueError(
                f"Unsupported period '{period}'. "
                f"Supported: {sorted(FREQUENCY_MAP.keys())}"
            )

        if tdxdir:
            self._tdxdir = tdxdir
            self._reader = None

        adjust = ADJUST_MAP.get(dividend_type)
        return self._batch_fetch(
            codes,
            lambda code: self._fetch_hybrid(code, start_date, end_date,
                                            period, adjust),
            label="hybrid_kline",
        )

    def _compute_remote_range(
        self,
        local_df: pd.DataFrame,
        start_date: Optional[str],
        end_date: Optional[str],
    ) -> tuple[Optional[str], Optional[str]]:
        """Determine the date range that needs to be fetched remotely.

        Returns (remote_start, remote_end) or (None, None) if local data covers
        the requested range entirely.
        """
        local_start = local_df["date"].min()
        local_end = local_df["date"].max()
        start_dt = pd.Timestamp(start_date) if start_date else local_start
        end_dt = pd.Timestamp(end_date) if end_date else pd.Timestamp.now().normalize()

        remote_start = None
        remote_end = None

        if local_start > start_dt:
            remote_start = start_dt.strftime("%Y-%m-%d")
            remote_end = (local_start - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            logger.info(
                f"Local data starts at {local_start.date()}, "
                f"fetching earlier data from {remote_start} to {remote_end}"
            )

        if local_end < end_dt:
            remote_start = (local_end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            remote_end = end_dt.strftime("%Y-%m-%d")
            logger.info(
                f"Local data ends at {local_end.date()}, "
                f"fetching newer data from {remote_start} to {remote_end}"
            )

        return remote_start, remote_end

    def _fetch_hybrid(
        self,
        code: str,
        start_date: Optional[str],
        end_date: Optional[str],
        period: str,
        adjust: Optional[str],
    ) -> Optional[pd.DataFrame]:
        base_info = RESAMPLE_MAP.get(period)
        if base_info:
            base_df = self._fetch_hybrid(
                code, start_date, end_date, base_info["base"], adjust,
            )
            if base_df is not None and not base_df.empty:
                base_df = resample_kline(base_df, base_info["freq"])
            return base_df

        local_df = self._read_local(code, period)

        if local_df is None or local_df.empty:
            logger.info(f"No local data for {code}, fetching all from network")
            return self._fetch_remote(code, start_date, end_date, period, adjust)

        local_df = self._normalize_kline_df(local_df, code)
        start_dt = pd.Timestamp(start_date) if start_date else local_df["date"].min()
        end_dt = pd.Timestamp(end_date) if end_date else pd.Timestamp.now().normalize()

        remote_start, remote_end = self._compute_remote_range(
            local_df, start_date, end_date
        )

        if remote_start is None and remote_end is None:
            df = local_df
            if start_date:
                df = df[df["date"] >= start_dt]
            if end_date:
                df = df[df["date"] <= end_dt]
            df = df.reset_index(drop=True)
            if adjust and not df.empty:
                df = apply_adjust(df, code, adjust,
                                  quotes_client=self._connection.client)
            return df

        remote_df = self._fetch_remote(code, remote_start, remote_end, period, adjust=None)

        if remote_df is not None and not remote_df.empty:
            remote_df = self._normalize_kline_df(remote_df, code)
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

        if adjust and not combined.empty:
            combined = apply_adjust(combined, code, adjust,
                                    quotes_client=self._connection.client)

        return combined

    def _read_local(self, code: str, period: str) -> Optional[pd.DataFrame]:
        try:
            reader = self._get_reader()
        except FileNotFoundError:
            logger.info("TDX directory not found, skipping local read")
            return None

        method = PERIOD_MAP.get(period)
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

        return df.copy()

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

        df = self._normalize_kline_df(df, code)

        if adjust:
            df = apply_adjust(df, code, adjust,
                              quotes_client=self._connection.client)
        return df
