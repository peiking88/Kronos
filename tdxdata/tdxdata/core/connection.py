import logging
from typing import Optional

from mootdx import config as mootdx_config
from mootdx.quotes import Quotes

from tdxdata.errors import ConnectionError

logger = logging.getLogger(__name__)

DEFAULT_SERVERS = {
    "HQ": [("119.147.212.81", 7709), ("112.74.214.43", 7727), ("221.231.141.60", 7709)],
    "EX": [("112.74.214.43", 7727)],
}


def _resolve_server(server: Optional[tuple]) -> Optional[tuple]:
    if server is not None:
        return server

    try:
        bestip = mootdx_config.get("BESTIP", {}).get("HQ", "")
        if isinstance(bestip, (tuple, list)) and len(bestip) == 2:
            return bestip
    except Exception:
        pass

    try:
        hq_list = mootdx_config.get("SERVER", {}).get("HQ", [])
        if hq_list:
            first = hq_list[0]
            return (first[1], first[2])
    except Exception:
        pass

    fallback = DEFAULT_SERVERS["HQ"][0]
    logger.debug(f"Using fallback server: {fallback}")
    return fallback


class TdxConnection:
    def __init__(self):
        self._client = None
        self._initialized = False

    @classmethod
    def local_only(cls) -> "TdxConnection":
        """Create a connection for local-only data access (no remote server).

        The returned connection has no remote client — it is only valid for
        source/storage paths that do not call self._connection.client.
        """
        conn = cls()
        conn._initialized = True
        return conn

    @property
    def client(self):
        if self._client is None:
            raise ConnectionError("mootdx not connected. Call initialize() first.")
        return self._client

    @property
    def is_connected(self) -> bool:
        return self._initialized

    def initialize(self, server: Optional[tuple] = None, timeout: int = 15) -> None:
        if self._initialized:
            logger.warning("Connection already initialized, skipping")
            return

        resolved = _resolve_server(server)
        try:
            self._client = Quotes.factory(
                market="std",
                server=resolved,
                timeout=timeout,
                quiet=True,
            )
            self._initialized = True
            logger.info("TDX connection initialized via mootdx")
        except Exception as e:
            raise ConnectionError(f"Failed to initialize mootdx connection: {e}") from e

    def close(self) -> None:
        if self._initialized and self._client is not None:
            try:
                self._client.close()
                logger.info("TDX connection closed")
            except Exception as e:
                logger.error(f"Error closing TDX connection: {e}")
            finally:
                self._client = None
                self._initialized = False

    def reconnect(self) -> None:
        if not self._initialized:
            raise ConnectionError("Not connected. Call initialize() first.")
        self._client.reconnect()
        logger.info("TDX connection reconnected")


class ResourceManager:
    def __init__(self, connection: TdxConnection):
        self._connection = connection

    def __enter__(self):
        return self._connection

    def __exit__(self, exc_type, exc_val, exc_tb):
        self._connection.close()
        if exc_type is not None:
            logger.error(f"Connection closed due to error: {exc_val}")
        return False
