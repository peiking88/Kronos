import logging
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class GapDetector:
    def detect(self, df: pd.DataFrame, freq: str = "B") -> list[str]:
        if df.empty or "date" not in df.columns:
            return []

        dates = pd.to_datetime(df["date"]).sort_values().unique()
        if len(dates) < 2:
            return []

        full_range = pd.date_range(start=dates.min(), end=dates.max(), freq=freq)
        existing = pd.DatetimeIndex(dates)
        missing = full_range.difference(existing)

        gaps = [d.strftime("%Y-%m-%d") for d in missing]
        if gaps:
            logger.info(f"Detected {len(gaps)} gap(s) in data")
        return gaps
