import logging
import os
import struct
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_QLIB_FIELDS = ["open", "close", "high", "low", "volume", "factor"]


def write_bin_file(data: np.ndarray, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    arr = np.array(data, dtype=np.float32)
    arr.tofile(filepath)


def write_calendar(dates: list[str], filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        for d in dates:
            f.write(f"{d}\n")


def write_instruments(
    instruments: list[tuple[str, str, str]], filepath: str
) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        for symbol, start, end in instruments:
            f.write(f"{symbol}\t{start}\t{end}\n")


def normalize_symbol(code: str) -> str:
    code = str(code).strip()
    upper = code.upper()
    if upper.startswith(("SH", "SZ", "BJ")):
        return upper
    if code.startswith("6") or code.startswith("5") or code.startswith("9"):
        return f"SH{code}"
    if code.startswith("0") or code.startswith("3") or code.startswith("2"):
        return f"SZ{code}"
    if code.startswith("4") or code.startswith("8"):
        return f"BJ{code}"
    return f"SH{code}"


def df_to_qlib_bins(
    df: pd.DataFrame,
    qlib_dir: str,
    freq: str = "day",
    fields: Optional[list[str]] = None,
) -> list[str]:
    if fields is None:
        fields = DEFAULT_QLIB_FIELDS

    if "stock_code" not in df.columns or "date" not in df.columns:
        raise ValueError("DataFrame must contain 'stock_code' and 'date' columns")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    written = []

    for code, group in df.groupby("stock_code"):
        symbol = normalize_symbol(code)
        group = group.sort_values("date").reset_index(drop=True)

        feat_dir = os.path.join(qlib_dir, "features", symbol)
        os.makedirs(feat_dir, exist_ok=True)

        for field in fields:
            if field not in group.columns:
                continue
            arr = group[field].values.astype(np.float32)
            filepath = os.path.join(feat_dir, f"{field}.{freq}.bin")
            write_bin_file(arr, filepath)
            written.append(filepath)

        logger.debug(f"Wrote {len(group)} rows for {symbol}")

    return written


def build_calendar(
    df: pd.DataFrame, qlib_dir: str, freq: str = "day"
) -> str:
    dates = (
        pd.to_datetime(df["date"])
        .dt.strftime("%Y-%m-%d")
        .sort_values()
        .unique()
        .tolist()
    )
    filepath = os.path.join(qlib_dir, "calendars", f"{freq}.txt")
    write_calendar(dates, filepath)
    logger.info(f"Wrote calendar with {len(dates)} trading days to {filepath}")
    return filepath


def build_instruments(
    df: pd.DataFrame, qlib_dir: str, instrument_name: str = "all"
) -> str:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    instruments = []
    for code, group in df.groupby("stock_code"):
        symbol = normalize_symbol(code)
        dates = group["date"].sort_values()
        instruments.append((symbol, dates.iloc[0], dates.iloc[-1]))

    filepath = os.path.join(qlib_dir, "instruments", f"{instrument_name}.txt")
    write_instruments(instruments, filepath)
    logger.info(f"Wrote {len(instruments)} instruments to {filepath}")
    return filepath
