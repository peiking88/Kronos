import os
import sqlite3
import logging

import pandas as pd

from tdxdata.core.registry import register_storage
from tdxdata.storage.base import StorageBase

logger = logging.getLogger(__name__)


@register_storage("sqlite")
class SQLiteStorage(StorageBase):
    def _get_db_path(self, **kwargs) -> str:
        output_path = self._output_path or "."
        os.makedirs(output_path, exist_ok=True)
        return os.path.join(output_path, "tdxdata.db")

    def _get_table_name(self, **kwargs) -> str:
        source = kwargs.get("source", "data")
        stock_list = kwargs.get("stock_list", ["unknown"])
        safe_code = stock_list[0].replace(".", "_")
        return f"{source}_{safe_code}"

    def save(self, df: pd.DataFrame, **kwargs) -> str:
        db_path = self._get_db_path(**kwargs)
        table_name = self._get_table_name(**kwargs)

        with sqlite3.connect(db_path) as conn:
            df.to_sql(table_name, conn, if_exists="replace", index=False)

        logger.info(f"Data saved to SQLite: {db_path} -> {table_name}")
        return db_path

    def load(self, **kwargs) -> pd.DataFrame:
        db_path = self._get_db_path(**kwargs)
        table_name = self._get_table_name(**kwargs)

        if not os.path.exists(db_path):
            raise FileNotFoundError(f"SQLite database not found: {db_path}")

        with sqlite3.connect(db_path) as conn:
            df = pd.read_sql(f"SELECT * FROM [{table_name}]", conn)

        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"])
        return df
