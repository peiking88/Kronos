import logging
import os
import sys
from pathlib import Path
from typing import Optional

from tdxdata.errors.exceptions import PluginNotFoundError

logger = logging.getLogger(__name__)

TDX_SEARCH_PATHS = [
    r"D:\new_tdx",
    r"C:\new_tdx",
    r"D:\lwj\new_tdx",
    r"C:\Program Files\new_tdx",
    r"D:\tdx",
    r"C:\tdx",
]

TDX_PLUGIN_REL_PATH = os.path.join("PYPlugins", "user")
TDX_PLUGIN_MODULE = "tqcenter"


class PluginManager:
    def __init__(self, tdx_path: Optional[str] = None):
        self._tdx_path = tdx_path
        self._plugin_path: Optional[str] = None
        self._tq_module = None
        self._initialized = False

    @property
    def tdx_path(self) -> Optional[str]:
        return self._tdx_path

    @property
    def plugin_path(self) -> Optional[str]:
        return self._plugin_path

    @property
    def is_available(self) -> bool:
        return self._tq_module is not None

    def discover(self) -> Optional[str]:
        if self._tdx_path:
            return self._discover_at(self._tdx_path)

        for path in TDX_SEARCH_PATHS:
            result = self._discover_at(path)
            if result:
                return result

        logger.warning("TDX installation not found in any known path")
        return None

    def _discover_at(self, base_path: str) -> Optional[str]:
        plugin_path = os.path.join(base_path, TDX_PLUGIN_REL_PATH)
        if not os.path.isdir(plugin_path):
            return None

        tq_init = os.path.join(plugin_path, TDX_PLUGIN_MODULE, "__init__.py")
        if not os.path.isfile(tq_init):
            return None

        self._tdx_path = base_path
        self._plugin_path = plugin_path
        logger.info(f"Found TDX plugin at: {plugin_path}")
        return plugin_path

    def setup_path(self):
        if not self._plugin_path:
            discovered = self.discover()
            if not discovered:
                raise PluginNotFoundError("Cannot setup path: TDX plugin not discovered")

        if self._plugin_path not in sys.path:
            sys.path.insert(0, self._plugin_path)
            logger.info(f"Added to sys.path: {self._plugin_path}")

    def load_module(self):
        self.setup_path()
        try:
            import importlib
            self._tq_module = importlib.import_module(TDX_PLUGIN_MODULE)
            logger.info(f"Loaded tqcenter module from: {self._plugin_path}")
            return self._tq_module
        except ImportError as e:
            raise PluginNotFoundError(f"Failed to import tqcenter: {e}")

    def get_module(self):
        if self._tq_module is None:
            return self.load_module()
        return self._tq_module
