import logging
from typing import Any, Optional

import pandas as pd

from tdxdata.core.connection import TdxConnection
from tdxdata.core.registry import PluginRegistry

logger = logging.getLogger(__name__)


_SOURCES_LOADED = False


def _ensure_sources_loaded():
    global _SOURCES_LOADED
    if _SOURCES_LOADED:
        return
    import tdxdata.sources.history_kline  # noqa: F401
    import tdxdata.sources.realtime_snapshot  # noqa: F401
    import tdxdata.sources.tick  # noqa: F401
    import tdxdata.sources.financial  # noqa: F401
    import tdxdata.sources.f10  # noqa: F401
    import tdxdata.sources.daily_basic  # noqa: F401
    import tdxdata.sources.local_kline  # noqa: F401
    import tdxdata.sources.hybrid_kline  # noqa: F401
    _SOURCES_LOADED = True


def _ensure_storages_loaded():
    if "dataframe" in PluginRegistry.list_storages():
        return
    import tdxdata.storage.dataframe  # noqa: F401
    import tdxdata.storage.csv  # noqa: F401
    import tdxdata.storage.sqlite  # noqa: F401
    import tdxdata.storage.parquet  # noqa: F401


class DataManager:
    def __init__(self, connection: TdxConnection):
        self._connection = connection

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

        source_cls = PluginRegistry.get_source(source)
        source_instance = source_cls(self._connection)

        df = source_instance.fetch(**kwargs)

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
