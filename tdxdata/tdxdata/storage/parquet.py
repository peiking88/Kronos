import os
import logging

import pandas as pd

from tdxdata.core.registry import register_storage
from tdxdata.storage.base import StorageBase

logger = logging.getLogger(__name__)


@register_storage("parquet")
class ParquetStorage(StorageBase):
    def save(self, df: pd.DataFrame, **kwargs) -> str:
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

        file_path = os.path.join(dir_path, f"{safe_code}.parquet")
        df.to_parquet(file_path, engine="pyarrow", compression="snappy", index=False)

        logger.info(f"Data saved to Parquet: {file_path}")
        return file_path

    def load(self, **kwargs) -> pd.DataFrame:
        file_path = kwargs.get("file_path")
        if not file_path:
            raise ValueError("file_path is required for ParquetStorage.load()")

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Parquet file not found: {file_path}")

        df = pd.read_parquet(file_path, engine="pyarrow")
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
