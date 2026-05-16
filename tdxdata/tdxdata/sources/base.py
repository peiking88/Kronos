from abc import ABC, abstractmethod
from typing import Any

import pandas as pd

from tdxdata.core.connection import TdxConnection


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
