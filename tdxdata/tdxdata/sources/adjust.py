import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

ADJUST_MAP = {
    "front": "qfq",
    "back": "hfq",
    "none": None,
}


def apply_adjust(df: pd.DataFrame, code: str, adjust: str) -> pd.DataFrame:
    if not adjust or adjust not in ("qfq", "hfq"):
        return df

    try:
        factor_df = fetch_factor(code, adjust)
    except Exception as e:
        logger.warning(f"Failed to fetch adjust factor for {code}: {e}")
        return df

    if factor_df is None or factor_df.empty:
        return df

    if df.index.name in ("date", "datetime"):
        df = df.reset_index(drop=True)

    date_col = "date" if "date" in df.columns else "datetime"
    if date_col not in df.columns:
        return df

    df[date_col] = pd.to_datetime(df[date_col])
    factor_df.index = pd.to_datetime(factor_df.index)

    df = df.sort_values(date_col).reset_index(drop=True)
    factor_df = factor_df.sort_index()

    merged = pd.merge_asof(
        df,
        factor_df[["factor"]],
        left_on=date_col,
        right_index=True,
        direction="backward" if adjust == "qfq" else "forward",
    )

    if "factor" not in merged.columns:
        return df

    merged["factor"] = merged["factor"].ffill().bfill().fillna(1.0)

    for col in ["open", "high", "low", "close"]:
        if col in merged.columns:
            merged[col] = merged[col] * merged["factor"]

    merged = merged.drop(columns=["factor"], errors="ignore")
    return merged


def fetch_factor(code: str, adjust: str) -> Optional[pd.DataFrame]:
    import httpx
    from mootdx.utils import get_stock_market

    symbol = code.replace("sh", "").replace("sz", "").replace("bj", "")
    market = get_stock_market(symbol, string=True)
    full_symbol = f"{market}{symbol}"

    url = f"https://finance.sina.com.cn/realstock/company/{full_symbol}/{adjust}.js"
    rsp = httpx.get(url, timeout=10)

    raw = rsp.text.split("=")[1].split("\n")[0]
    data = eval(raw)
    factor_df = pd.DataFrame(data["data"])
    factor_df.columns = ["date", "factor"]
    factor_df["date"] = pd.to_datetime(factor_df["date"])
    factor_df = factor_df.set_index("date")
    factor_df["factor"] = factor_df["factor"].astype(float)
    return factor_df
