import sys
from typing import Type

_source_registry: dict[str, Type] = {}
_storage_registry: dict[str, Type] = {}

_SOURCE_MODULES = [
    "tdxdata.sources.history_kline",
    "tdxdata.sources.realtime_snapshot",
    "tdxdata.sources.tick",
    "tdxdata.sources.financial",
    "tdxdata.sources.f10",
    "tdxdata.sources.daily_basic",
    "tdxdata.sources.local_kline",
    "tdxdata.sources.hybrid_kline",
]

_STORAGE_MODULES = [
    "tdxdata.storage.dataframe",
]


def register_source(name: str):
    def decorator(cls: Type) -> Type:
        if name in _source_registry:
            raise ValueError(f"Data source '{name}' already registered")
        _source_registry[name] = cls
        return cls
    return decorator


def register_storage(name: str):
    def decorator(cls: Type) -> Type:
        if name in _storage_registry:
            raise ValueError(f"Storage '{name}' already registered")
        _storage_registry[name] = cls
        return cls
    return decorator


class PluginRegistry:
    @staticmethod
    def get_source(name: str) -> Type:
        if name not in _source_registry:
            raise KeyError(f"Data source '{name}' not found. Available: {list(_source_registry.keys())}")
        return _source_registry[name]

    @staticmethod
    def get_storage(name: str) -> Type:
        if name not in _storage_registry:
            raise KeyError(f"Storage '{name}' not found. Available: {list(_storage_registry.keys())}")
        return _storage_registry[name]

    @staticmethod
    def list_sources() -> list[str]:
        return list(_source_registry.keys())

    @staticmethod
    def list_storages() -> list[str]:
        return list(_storage_registry.keys())

    @staticmethod
    def clear():
        _source_registry.clear()
        _storage_registry.clear()
        for mod in _SOURCE_MODULES + _STORAGE_MODULES:
            sys.modules.pop(mod, None)
