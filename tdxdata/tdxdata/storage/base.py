from abc import ABC, abstractmethod
from typing import Any, Optional

import pandas as pd


class StorageBase(ABC):
    def __init__(self, output_path: Optional[str] = None):
        self._output_path = output_path

    @abstractmethod
    def save(self, df: pd.DataFrame, **kwargs) -> Any:
        pass

    @abstractmethod
    def load(self, **kwargs) -> pd.DataFrame:
        pass
