import importlib
import logging
from typing import Any, Optional

import pandas as pd

from tdxdata.core.connection import TdxConnection
from tdxdata.core.registry import (
    PluginRegistry, _SOURCE_MODULES, _STORAGE_MODULES,
)
from tdxdata.errors import (
    CircuitBreaker, CircuitBreakerOpenError, ConnectionError,
    DataFetchError, RetryExhaustedError, RetryPolicy,
)
from tdxdata.sync import SyncManager

logger = logging.getLogger(__name__)

RETRIABLE_EXCEPTIONS = (DataFetchError, ConnectionError, OSError, TimeoutError)


_SOURCES_LOADED = False


def _ensure_sources_loaded():
    global _SOURCES_LOADED
    if _SOURCES_LOADED:
        return
    for mod in _SOURCE_MODULES:
        importlib.import_module(mod)
    _SOURCES_LOADED = True


def _ensure_storages_loaded():
    if "dataframe" in PluginRegistry.list_storages():
        return
    for mod in _STORAGE_MODULES:
        importlib.import_module(mod)


class DataManager:
    def __init__(self, connection: TdxConnection):
        self._connection = connection
        self._retry_policy = RetryPolicy(max_retries=3, base_delay=1.0)
        self._circuit_breaker = CircuitBreaker(failure_threshold=5, recovery_timeout=60)
        self._sync_manager = SyncManager()

    @property
    def sync_manager(self) -> SyncManager:
        return self._sync_manager

    def fetch(
        self,
        source: str,
        output: str = "dataframe",
        output_path: Optional[str] = None,
        incremental: bool = False,
        **kwargs,
    ) -> Any:
        _ensure_sources_loaded()
        _ensure_storages_loaded()

        if not self._circuit_breaker.allow_request():
            raise CircuitBreakerOpenError(
                "请求被熔断器拒绝，服务暂时不可用"
            )

        if incremental:
            kwargs = self._apply_incremental(source, kwargs)

        source_kwargs = kwargs
        if source == "daily_basic" and output_path is not None:
            source_kwargs = dict(kwargs)
            source_kwargs["output_path"] = output_path

        source_cls = PluginRegistry.get_source(source)
        source_instance = source_cls(self._connection)

        try:
            df = self._retry_policy.execute(source_instance.fetch,
                                          retriable=RETRIABLE_EXCEPTIONS, **source_kwargs)
            self._circuit_breaker.record_success()
        except RetryExhaustedError:
            self._circuit_breaker.record_failure()
            raise

        if incremental:
            self._update_sync_state(source, kwargs)

        is_empty = (
            df is None
            or (isinstance(df, pd.DataFrame) and df.empty)
            or (isinstance(df, dict) and len(df) == 0)
        )
        if is_empty:
            logger.warning(f"No data returned from source '{source}'")
            return df

        if output == "dataframe":
            return df

        if not isinstance(df, pd.DataFrame):
            logger.warning(f"Cannot save non-DataFrame data via '{output}' storage")
            return df

        storage_cls = PluginRegistry.get_storage(output)
        storage_instance = storage_cls(output_path=output_path)
        storage_instance.save(df, source=source, **kwargs)
        logger.info(f"Data saved via {output} storage")
        return df

    def _apply_incremental(self, source: str, kwargs: dict) -> dict:
        """增量同步：根据上次同步时间自动调整 start_date。"""
        codes = kwargs.get("stock_list") or []
        if kwargs.get("stock_code"):
            codes = [kwargs["stock_code"]]
        if not codes:
            return kwargs

        start_dates = []
        for code in codes:
            last_sync, _ = self._sync_manager.get_incremental_range(code, source)
            start_dates.append(last_sync)

        # 取所有股票中最早的 last_sync 作为起点，全 None 则不改
        valid = [s for s in start_dates if s is not None]
        if valid:
            kwargs = dict(kwargs)
            kwargs["start_date"] = min(valid)
            logger.info(f"增量同步: start_date={kwargs['start_date']}")

        return kwargs

    def _update_sync_state(self, source: str, kwargs: dict) -> None:
        """更新所有相关股票的同步时间戳。"""
        codes = kwargs.get("stock_list") or []
        if kwargs.get("stock_code"):
            codes = [kwargs["stock_code"]]
        for code in codes:
            self._sync_manager.update_sync_state(code, source)
