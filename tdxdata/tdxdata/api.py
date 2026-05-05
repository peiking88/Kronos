import logging
from typing import Any, Optional

import pandas as pd

from tdxdata.core.connection import TdxConnection
from tdxdata.core.data_manager import DataManager
from tdxdata.core.registry import PluginRegistry
from tdxdata.errors.resource import ResourceManager
from tdxdata.logging.logger import get_logger

logger = get_logger()


class TdxData:
    def __init__(self, server: Optional[tuple] = None, timeout: int = 15):
        self._server = server
        self._timeout = timeout
        self._connection = TdxConnection()
        self._data_manager: Optional[DataManager] = None
        self._resource_manager = ResourceManager(self._connection)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def connect(self) -> None:
        self._connection.initialize(server=self._server, timeout=self._timeout)
        self._data_manager = DataManager(self._connection)
        logger.info("TdxData connected via mootdx")

    def close(self) -> None:
        self._connection.close()
        logger.info("TdxData closed")

    def fetch(self, source: str, output: str = "dataframe",
              output_path: Optional[str] = None, **kwargs) -> Any:
        self._ensure_connected()
        assert self._data_manager is not None
        return self._data_manager.fetch(
            source=source, output=output, output_path=output_path, **kwargs
        )

    def fetch_history(
        self,
        stock_list: list[str],
        start_date: str,
        end_date: str,
        period: str = "1d",
        dividend_type: str = "front",
        output: str = "dataframe",
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        return self.fetch(
            source="history_kline",
            stock_list=stock_list,
            start_date=start_date,
            end_date=end_date,
            period=period,
            dividend_type=dividend_type,
            output=output,
            output_path=output_path,
        )

    def fetch_realtime(
        self,
        stock_code: Optional[str] = None,
        stock_list: Optional[list[str]] = None,
        output: str = "dataframe",
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        return self.fetch(
            source="realtime_snapshot",
            stock_code=stock_code,
            stock_list=stock_list,
            output=output,
            output_path=output_path,
        )

    def fetch_tick(
        self,
        stock_code: str,
        date: Optional[str] = None,
        output: str = "dataframe",
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        return self.fetch(
            source="tick",
            stock_code=stock_code,
            date=date,
            output=output,
            output_path=output_path,
        )

    def fetch_f10(
        self,
        stock_code: str,
        sections: Optional[list[str]] = None,
    ) -> dict[str, pd.DataFrame]:
        self._ensure_connected()
        assert self._data_manager is not None
        return self._data_manager.fetch(
            source="f10", stock_code=stock_code, sections=sections
        )

    def fetch_basic(
        self,
        stock_code: str,
        date: Optional[str] = None,
        output: str = "dataframe",
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        return self.fetch(
            source="daily_basic",
            stock_code=stock_code,
            date=date,
            output=output,
            output_path=output_path,
        )

    def fetch_financial(
        self,
        stock_code: str,
        output: str = "dataframe",
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        return self.fetch(
            source="financial",
            stock_code=stock_code,
            output=output,
            output_path=output_path,
        )

    def fetch_local(
        self,
        stock_list: Optional[list[str]] = None,
        stock_code: Optional[str] = None,
        period: str = "1d",
        tdxdir: Optional[str] = None,
        dividend_type: str = "none",
        output: str = "dataframe",
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        return self.fetch(
            source="local_kline",
            stock_list=stock_list,
            stock_code=stock_code,
            period=period,
            tdxdir=tdxdir,
            dividend_type=dividend_type,
            output=output,
            output_path=output_path,
        )

    def fetch_hybrid(
        self,
        stock_list: Optional[list[str]] = None,
        stock_code: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        period: str = "1d",
        tdxdir: Optional[str] = None,
        dividend_type: str = "none",
        output: str = "dataframe",
        output_path: Optional[str] = None,
    ) -> pd.DataFrame:
        return self.fetch(
            source="hybrid_kline",
            stock_list=stock_list,
            stock_code=stock_code,
            start_date=start_date,
            end_date=end_date,
            period=period,
            tdxdir=tdxdir,
            dividend_type=dividend_type,
            output=output,
            output_path=output_path,
        )

    def to_qlib(
        self,
        stock_list: list[str],
        period: str = "1d",
        dividend_type: str = "none",
        tdxdir: Optional[str] = None,
        qlib_dir: str = "./data/qlib",
        instrument_name: str = "all",
    ) -> dict:
        from tdxdata.qlib.converter import QlibConverter

        converter = QlibConverter(tdxdir=tdxdir, qlib_dir=qlib_dir)
        return converter.convert(
            stock_list=stock_list,
            period=period,
            dividend_type=dividend_type,
            instrument_name=instrument_name,
        )

    def _ensure_connected(self) -> None:
        if self._data_manager is None:
            self.connect()


