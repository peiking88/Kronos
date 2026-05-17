import logging
import os
from abc import ABC, abstractmethod
from typing import Any, Callable, Optional

import pandas as pd
from mootdx.reader import Reader

from tdxdata.core.connection import TdxConnection

logger = logging.getLogger(__name__)

DEFAULT_TDXDIR = os.path.expanduser("~/.local/share/tdxcfv/drive_c/tc/")

FREQUENCY_MAP = {
    "1m": 8, "5m": 0, "15m": 1, "30m": 2,
    "1h": 3, "1d": 9, "1w": 5, "1mon": 6,
}

PERIOD_MAP = {
    "1m": "minute_1",
    "5m": "minute_5",
    "15m": None,
    "30m": None,
    "1h": None,
    "1d": "daily",
    "1w": None,
    "1mon": None,
}

STANDARD_COLUMNS = ["stock_code", "date", "open", "high", "low", "close", "volume", "amount"]

DEFAULT_VOL_MAP = {"vol": "volume"}

RESAMPLE_MAP = {
    "15m": {"base": "5m", "freq": "15min"},
    "30m": {"base": "5m", "freq": "30min"},
    "1h": {"base": "5m", "freq": "1h"},
    "1w": {"base": "1d", "freq": "W"},
    "1mon": {"base": "1d", "freq": "ME"},
}

_AGG_RULES = {
    "open": "first",
    "high": "max",
    "low": "min",
    "close": "last",
    "volume": "sum",
    "amount": "sum",
}


def resample_kline(df: pd.DataFrame, target_freq: str) -> pd.DataFrame:
    if df.empty:
        return df

    code = df["stock_code"].iloc[0]
    df = df.set_index("date")
    agg_cols = [c for c in _AGG_RULES if c in df.columns]
    resampled = df[agg_cols].resample(target_freq).agg(_AGG_RULES)
    resampled = resampled.dropna(subset=["open"]).reset_index()
    resampled.loc[:, "stock_code"] = code
    keep = [c for c in STANDARD_COLUMNS if c in resampled.columns]
    extra = [c for c in resampled.columns if c not in keep]
    return resampled[keep + extra].copy()


def _reset_date_index(df: pd.DataFrame) -> pd.DataFrame:
    """If the index is a date/datetime axis, reset it to a regular column.

    Shared between _normalize_kline_df and apply_adjust to keep index
    handling consistent across all data paths.
    """
    if df.index.name in ("date", "datetime"):
        return df.reset_index(drop=df.index.name in df.columns)
    return df


def get_tdx_reader(tdxdir: str) -> Reader:
    if not os.path.isdir(tdxdir):
        raise FileNotFoundError(
            f"TDX directory not found: {tdxdir}. "
            f"Set tdxdir parameter or ensure default path exists."
        )
    return Reader.factory(market="std", tdxdir=tdxdir)


class DataSourceBase(ABC):
    def __init__(self, connection: TdxConnection):
        self._connection = connection

    @abstractmethod
    def fetch(self, **kwargs) -> Any:
        pass

    def _normalize_columns(self, df: pd.DataFrame, column_map: dict[str, str]) -> pd.DataFrame:
        if df.empty:
            return df
        rename = {k: v for k, v in column_map.items() if k in df.columns}
        return df.rename(columns=rename)

    def _normalize_kline_df(self, df: pd.DataFrame, code: str) -> pd.DataFrame:
        df = _reset_date_index(df.copy())
        df["stock_code"] = code

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"].astype(str), errors="coerce")
        elif "datetime" in df.columns:
            df["date"] = pd.to_datetime(df["datetime"].astype(str), errors="coerce")

        df = self._normalize_columns(df, DEFAULT_VOL_MAP)

        keep = [c for c in STANDARD_COLUMNS if c in df.columns]
        extra = [c for c in df.columns if c not in keep]
        return df[keep + extra].copy()

    def _batch_fetch(self, codes: list[str], fetch_fn: Callable,
                     label: str) -> pd.DataFrame:
        parts = []
        for code in codes:
            try:
                df = fetch_fn(code)
                if df is not None and not df.empty:
                    parts.append(df)
            except Exception as e:
                logger.warning(f"Error in {label} for {code}: {e}")
        if not parts:
            return pd.DataFrame()
        return pd.concat(parts, ignore_index=True)
