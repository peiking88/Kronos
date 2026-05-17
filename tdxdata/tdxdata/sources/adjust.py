import logging
import time

import pandas as pd

from tdxdata.sources.base import _reset_date_index

logger = logging.getLogger(__name__)

ADJUST_MAP = {
    "front": "qfq",
    "back": "hfq",
    "none": None,
    "qfq": "qfq",
    "hfq": "hfq",
}


def fetch_factor(code: str, adjust: str, quotes_client) -> pd.DataFrame:
    if quotes_client is None:
        raise ValueError("quotes_client is required for factor fetching")
    return _retry_fetch(code, adjust, quotes_client)


def _retry_fetch(code: str, adjust: str, quotes_client,
                 max_retries: int = 3) -> pd.DataFrame:
    delay = 1.0
    for attempt in range(max_retries):
        try:
            xdxr = quotes_client.xdxr(symbol=code)
            end = pd.Timestamp.now().strftime("%Y-%m-%d")
            kline = quotes_client.get_k_data(code, "1990-01-01", end)
            return compute_factor_from_xdxr(xdxr, kline, adjust)
        except Exception as e:
            if attempt < max_retries - 1:
                logger.warning(
                    f"获取因子失败 (第{attempt + 1}次): {code} {adjust}: {e}，"
                    f"{delay}s 后重试"
                )
                time.sleep(delay)
                delay = min(delay * 2, 30.0)
            else:
                logger.error(
                    f"获取因子失败 (已重试{max_retries}次): {code} {adjust}: {e}"
                )
                raise


def compute_factor_from_xdxr(
    xdxr: pd.DataFrame | None,
    kline: pd.DataFrame | None,
    adjust: str,
) -> pd.DataFrame:
    if xdxr is None or xdxr.empty or kline is None or kline.empty:
        return pd.DataFrame(columns=["factor"])

    events = _normalize_xdxr_events(xdxr)
    prices = _pre_close_prices(events, kline)
    if not prices:
        return pd.DataFrame(columns=["factor"])

    reverse = adjust == "qfq"
    events = events.sort_values("date", ascending=not reverse).reset_index(drop=True)
    cumulative = 1.0
    records = []

    for _, event in events.iterrows():
        date = pd.Timestamp(event["date"])
        pre_close = prices.get(date.strftime("%Y-%m-%d"))
        if pre_close is None or pre_close == 0 or (isinstance(pre_close, float) and pd.isna(pre_close)):
            factor = cumulative
        else:
            fenhong = _per_share(event.get("fenhong", 0.0))
            peigujia = float(event.get("peigujia", 0.0) or 0.0)
            songzhuangu = _per_share(event.get("songzhuangu", 0.0))
            peigu = _per_share(event.get("peigu", 0.0))
            category = event.get("category", 1)
            name = event.get("name", "")

            if category in (1, 2) or name == "除权除息":
                if pd.isna(fenhong):
                    event_factor = 1.0
                else:
                    numerator = pre_close - fenhong + peigujia * peigu
                    denominator = pre_close * (1 + songzhuangu + peigu)
                    if denominator == 0 or numerator == 0:
                        event_factor = 1.0
                    elif adjust == "qfq":
                        event_factor = numerator / denominator
                    else:
                        event_factor = denominator / numerator
            else:
                event_factor = 1.0

            cumulative *= event_factor
            factor = cumulative

        records.append({"date": date, "factor": factor})

    result = pd.DataFrame(records)
    if result.empty:
        return pd.DataFrame(columns=["factor"])
    result = result.set_index("date").sort_index()
    return result[["factor"]]


def _normalize_xdxr_events(xdxr: pd.DataFrame) -> pd.DataFrame:
    events = xdxr.copy()
    if "date" not in events.columns and all(c in events.columns for c in ("year", "month", "day")):
        events["date"] = pd.to_datetime(
            events["year"].astype(str) + "-" +
            events["month"].astype(str).str.zfill(2) + "-" +
            events["day"].astype(str).str.zfill(2),
            errors="coerce",
        )
    elif "date" in events.columns:
        events["date"] = pd.to_datetime(events["date"], errors="coerce")

    events = events.dropna(subset=["date"])
    for col in ["fenhong", "peigujia", "songzhuangu", "peigu"]:
        if col not in events.columns:
            events[col] = 0.0
    if "category" not in events.columns:
        events["category"] = 1
    return events


def _pre_close_prices(events: pd.DataFrame, kline: pd.DataFrame) -> dict[str, float]:
    prices = {}
    data = _reset_date_index(kline.copy())
    if "date" in data.columns:
        data["date"] = pd.to_datetime(data["date"], errors="coerce")
    elif "datetime" in data.columns:
        data["date"] = pd.to_datetime(data["datetime"], errors="coerce")
    else:
        data["date"] = pd.to_datetime(data.index, errors="coerce")

    data = data.dropna(subset=["date"]).sort_values("date")
    if "close" not in data.columns:
        return prices

    for date in events["date"]:
        prev = data[data["date"] < date]
        if not prev.empty:
            prices[pd.Timestamp(date).strftime("%Y-%m-%d")] = float(prev.iloc[-1]["close"])
    return prices


def _per_share(value) -> float:
    value = float(value or 0.0)
    return value / 10 if value >= 1 else value


def apply_adjust(df: pd.DataFrame, code: str, adjust: str,
                 quotes_client=None) -> pd.DataFrame:
    if not adjust or adjust not in ("qfq", "hfq"):
        return df

    try:
        factor_df = fetch_factor(code, adjust, quotes_client=quotes_client)
    except Exception as e:
        logger.warning(f"获取复权因子失败 {code}: {e}")
        return df

    if factor_df is None or factor_df.empty:
        return df

    df = _reset_date_index(df)

    date_col = "date" if "date" in df.columns else "datetime"
    if date_col not in df.columns:
        return df

    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col].astype(str))
    factor_df = factor_df.copy()
    factor_df.index = pd.to_datetime(factor_df.index)

    # pandas 3.x 下不同来源的 datetime64 精度可能不一致（[s] vs [us]），
    # 统一转为 datetime64[us] 避免 merge_asof 报 incompatible merge keys
    common_dtype = "datetime64[us]"
    df[date_col] = df[date_col].astype(common_dtype)
    factor_df.index = factor_df.index.astype(common_dtype)

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

    if merged.empty:
        return df

    merged.loc[:, "factor"] = merged["factor"].ffill().bfill().fillna(1.0)

    if adjust == "qfq":
        latest_factor = factor_df["factor"].iloc[-1]
        if latest_factor > 0:
            merged["factor"] = merged["factor"] / latest_factor

    for col in ["open", "high", "low", "close"]:
        if col in merged.columns:
            merged[col] = merged[col] * merged["factor"]

    merged = merged.drop(columns=["factor"], errors="ignore")
    return merged
