import os
from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd

from tdxdata.errors import StorageError


class StorageBase(ABC):
    def __init__(self, output_path: Optional[str] = None):
        self._output_path = output_path

    @abstractmethod
    def save(self, df: pd.DataFrame, **kwargs) -> Any:
        pass

    def load(self, **kwargs) -> pd.DataFrame:
        raise StorageError(f"{type(self).__name__} does not support loading")

    def _resolve_file_path(self, ext: str, **kwargs) -> str:
        """Derive output file path from kwargs (stock_list, stock_code, source).

        Used by single-file-per-stock backends (CSV, Parquet).
        """
        output_path = self._output_path or "."
        source = kwargs.get("source", "data")
        stock_list = kwargs.get("stock_list")
        if stock_list:
            safe_code = stock_list[0].replace(".", "_")
        else:
            safe_code = kwargs.get("stock_code", "unknown") or "unknown"
            safe_code = safe_code.replace(".", "_")

        dir_path = os.path.join(output_path, source)
        os.makedirs(dir_path, exist_ok=True)
        return os.path.join(dir_path, f"{safe_code}.{ext}")
