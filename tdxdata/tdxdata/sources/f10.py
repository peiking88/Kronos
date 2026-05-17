import logging

import pandas as pd

from tdxdata.core.connection import TdxConnection
from tdxdata.core.registry import register_source
from tdxdata.sources.base import DataSourceBase

logger = logging.getLogger(__name__)

F10_SECTION_NAMES = [
    "最新提示",
    "公司概况",
    "财务分析",
    "股东研究",
    "股本结构",
    "分红扩股",
    "行业题材",
]


@register_source("f10")
class F10DataSource(DataSourceBase):
    def fetch(
        self,
        stock_code: str,
        sections: list[str] | None = None,
        **kwargs,
    ) -> dict[str, pd.DataFrame]:
        target_sections = sections if sections else F10_SECTION_NAMES
        result = {}

        for section in target_sections:
            try:
                df = self._fetch_section(stock_code, section)
                if df is not None and not df.empty:
                    result[section] = df
            except Exception as e:
                logger.warning(f"F10 section '{section}' not available for {stock_code}: {e}")

        return result

    def _fetch_section(self, stock_code: str, section: str) -> pd.DataFrame:
        try:
            client = self._connection.client
            raw = client.F10(symbol=stock_code, name=section)
        except Exception as e:
            logger.error(f"Error fetching F10 {section} for {stock_code}: {e}")
            return pd.DataFrame()

        if raw is None:
            return pd.DataFrame()

        if isinstance(raw, dict):
            return pd.DataFrame([raw])
        elif isinstance(raw, pd.DataFrame):
            return raw
        elif isinstance(raw, list):
            return pd.DataFrame(raw)
        return pd.DataFrame()
