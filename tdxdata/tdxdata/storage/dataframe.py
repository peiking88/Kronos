import pandas as pd

from tdxdata.core.registry import register_storage
from tdxdata.storage.base import StorageBase


@register_storage("dataframe")
class DataFrameStorage(StorageBase):
    def save(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        return df
