import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from tdxdata.core.registry import register_storage
from tdxdata.qlib.qlib_bin import (
    build_calendar,
    build_instruments,
    df_to_qlib_bins,
    normalize_symbol,
)
from tdxdata.storage.base import StorageBase

logger = logging.getLogger(__name__)

DEFAULT_QLIB_FIELDS = ["open", "close", "high", "low", "volume", "factor"]


@register_storage("qlib")
class QlibStorage(StorageBase):
    def save(self, df: pd.DataFrame, **kwargs) -> str:
        output_path = self._output_path or "./data/qlib"
        freq = kwargs.get("freq", "day")
        fields = kwargs.get("fields", DEFAULT_QLIB_FIELDS)
        instrument_name = kwargs.get("instrument_name", "all")

        os.makedirs(output_path, exist_ok=True)

        written = df_to_qlib_bins(df, output_path, freq=freq, fields=fields)
        build_calendar(df, output_path, freq=freq)
        build_instruments(df, output_path, instrument_name=instrument_name)

        stocks = df["stock_code"].nunique() if "stock_code" in df.columns else 0
        logger.info(f"Saved {stocks} stocks to Qlib format at {output_path}")
        return output_path

    def load(self, **kwargs) -> pd.DataFrame:
        qlib_dir = kwargs.get("qlib_dir", self._output_path or "./data/qlib")
        freq = kwargs.get("freq", "day")
        symbol = kwargs.get("symbol")
        fields = kwargs.get("fields", DEFAULT_QLIB_FIELDS)

        if not symbol:
            raise ValueError("symbol is required for QlibStorage.load()")

        parts = {}
        for field in fields:
            filepath = os.path.join(
                qlib_dir, "features", symbol, f"{field}.{freq}.bin"
            )
            if not os.path.exists(filepath):
                continue
            arr = np.fromfile(filepath, dtype=np.float32)
            parts[field] = arr

        if not parts:
            raise FileNotFoundError(
                f"No Qlib data found for {symbol} in {qlib_dir}"
            )

        length = len(next(iter(parts.values())))
        df = pd.DataFrame(parts)

        cal_path = os.path.join(qlib_dir, "calendars", f"{freq}.txt")
        if os.path.exists(cal_path):
            with open(cal_path) as f:
                dates = [line.strip() for line in f if line.strip()]
            if len(dates) >= length:
                df["date"] = dates[:length]

        df["stock_code"] = symbol
        return df
