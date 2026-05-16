import pytest
from tdxdata.core.registry import (
    PluginRegistry,
    register_source,
    register_storage,
    _source_registry,
    _storage_registry,
)


class TestPluginRegistry:
    def setup_method(self):
        PluginRegistry.clear()

    def test_register_source(self):
        @register_source("test_source")
        class FakeSource:
            pass

        assert "test_source" in PluginRegistry.list_sources()
        assert PluginRegistry.get_source("test_source") is FakeSource

    def test_register_duplicate_source_raises(self):
        @register_source("dup_source")
        class Fake1:
            pass

        with pytest.raises(ValueError, match="already registered"):
            @register_source("dup_source")
            class Fake2:
                pass

    def test_get_source_not_found(self):
        with pytest.raises(KeyError, match="not found"):
            PluginRegistry.get_source("nonexistent")

    def test_register_storage(self):
        @register_storage("test_storage")
        class FakeStorage:
            pass

        assert "test_storage" in PluginRegistry.list_storages()
        assert PluginRegistry.get_storage("test_storage") is FakeStorage

    def test_register_duplicate_storage_raises(self):
        @register_storage("dup_storage")
        class Fake1:
            pass

        with pytest.raises(ValueError, match="already registered"):
            @register_storage("dup_storage")
            class Fake2:
                pass

    def test_get_storage_not_found(self):
        with pytest.raises(KeyError, match="not found"):
            PluginRegistry.get_storage("nonexistent")

    def test_list_sources_empty(self):
        assert PluginRegistry.list_sources() == []

    def test_list_storages_empty(self):
        assert PluginRegistry.list_storages() == []

    def test_clear(self):
        @register_source("s1")
        class S1:
            pass

        @register_storage("st1")
        class ST1:
            pass

        PluginRegistry.clear()
        assert PluginRegistry.list_sources() == []
        assert PluginRegistry.list_storages() == []
