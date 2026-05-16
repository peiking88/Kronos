import logging
import os
from typing import Optional

import numpy as np
import pandas as pd

from tdxdata.qlib.qlib_bin import (
    build_calendar,
    build_instruments,
    df_to_qlib_bins,
    normalize_symbol,
)

logger = logging.getLogger(__name__)


class QlibConverter:
    def __init__(
        self,
        tdxdir: Optional[str] = None,
        qlib_dir: str = "./data/qlib",
    ):
        from tdxdata.sources.local_kline import DEFAULT_TDXDIR

        self.tdxdir = tdxdir or DEFAULT_TDXDIR
        self.qlib_dir = qlib_dir

    def convert(
        self,
        stock_list: list[str],
        period: str = "1d",
        dividend_type: str = "none",
        instrument_name: str = "all",
        fields: Optional[list[str]] = None,
    ) -> dict:
        df = self._fetch_data(stock_list, period, dividend_type)
        if df.empty:
            logger.warning("No data fetched, conversion aborted")
            return {"status": "empty", "files": [], "stocks": 0}

        freq = "day" if period == "1d" else "1min"

        df = self._prepare_for_qlib(df, dividend_type)

        os.makedirs(self.qlib_dir, exist_ok=True)
        written = df_to_qlib_bins(df, self.qlib_dir, freq=freq, fields=fields)
        cal_path = build_calendar(df, self.qlib_dir, freq=freq)
        inst_path = build_instruments(df, self.qlib_dir, instrument_name=instrument_name)

        stocks = df["stock_code"].nunique()
        logger.info(
            f"Converted {stocks} stocks, {len(df)} rows to Qlib format at {self.qlib_dir}"
        )

        return {
            "status": "ok",
            "files": written,
            "calendar": cal_path,
            "instruments": inst_path,
            "stocks": stocks,
            "rows": len(df),
        }

    def _fetch_data(
        self, stock_list: list[str], period: str, dividend_type: str
    ) -> pd.DataFrame:
        from tdxdata.core.connection import TdxConnection
        from tdxdata.sources.local_kline import LocalKlineSource

        conn = TdxConnection()
        conn._initialized = True
        source = LocalKlineSource(conn, tdxdir=self.tdxdir)
        return source.fetch(
            stock_list=stock_list, period=period, dividend_type=dividend_type
        )

    def _prepare_for_qlib(
        self, df: pd.DataFrame, dividend_type: str
    ) -> pd.DataFrame:
        df = df.copy()

        if "factor" not in df.columns:
            df["factor"] = 1.0

        if "money" not in df.columns and "amount" in df.columns:
            df["money"] = df["amount"]

        if dividend_type == "none":
            self._compute_factor(df)

        for col in ["open", "high", "low", "close", "volume", "money"]:
            if col in df.columns:
                df[col] = df[col].astype(np.float32)

        return df

    def _compute_factor(self, df: pd.DataFrame) -> None:
        if "factor" not in df.columns:
            df["factor"] = 1.0
            return

        if (df["factor"] == 1.0).all() and "close" in df.columns:
            logger.info("Factor column is all 1.0, keeping original prices")
