import logging
from typing import Any, Optional

import pandas as pd

from mootdx.consts import MARKET_SH, MARKET_SZ

from tdxdata.core.connection import ResourceManager, TdxConnection
from tdxdata.core.data_manager import DataManager

logger = logging.getLogger(__name__)


class TdxData:
    def __init__(self, server: Optional[tuple] = None, timeout: int = 15):
        self._server = server
        self._timeout = timeout
        self._connection = TdxConnection()
        self._data_manager: Optional[DataManager] = None
        self._resource_manager = ResourceManager(self._connection)
        self._stock_cache: dict[int, pd.DataFrame] = {}

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
        return self.fetch(
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

    def sync_status(self, stock_code: str, data_type: str = "history_kline") -> Optional[str]:
        """查询上次同步时间。

        Args:
            stock_code: 股票代码
            data_type: 数据类型，如 "history_kline"、"local_kline"

        Returns:
            上次同步日期（YYYY-MM-DD），未同步返回 None
        """
        self._ensure_connected()
        assert self._data_manager is not None
        return self._data_manager.sync_manager.get_last_sync(stock_code, data_type)

    def _get_stocks(self, market: int) -> pd.DataFrame:
        """获取指定市场的股票列表（带缓存）。"""
        if market not in self._stock_cache:
            self._ensure_connected()
            self._stock_cache[market] = self._connection.client.stocks(market=market)
        return self._stock_cache[market]

    @staticmethod
    def _market_from_code(code: str) -> int:
        """根据股票代码推断市场。6/9 开头为沪市，其余为深市。"""
        if code[0] in ("6", "9"):
            return MARKET_SH
        return MARKET_SZ

    def get_stock_name(self, stock_code: str) -> Optional[str]:
        """由股票代码获取股票名称。

        Args:
            stock_code: 股票代码，如 "600519"

        Returns:
            股票名称，如 "贵州茅台"；未找到返回 None
        """
        market = self._market_from_code(stock_code)
        df = self._get_stocks(market)
        matched = df[df["code"] == stock_code]
        if matched.empty:
            return None
        return str(matched["name"].values[0])

    def _ensure_connected(self) -> None:
        if self._data_manager is None:
            self.connect()


