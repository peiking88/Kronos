import pytest
from unittest.mock import MagicMock, patch

from tdxdata.core.connection import TdxConnection
from tdxdata.errors.exceptions import ConnectionError


class TestTdxConnection:
    def test_initial_state(self):
        conn = TdxConnection()
        assert conn.is_connected is False

    def test_client_raises_when_not_connected(self):
        conn = TdxConnection()
        with pytest.raises(ConnectionError, match="not connected"):
            conn.client

    @patch("tdxdata.core.connection.Quotes")
    def test_initialize_success(self, mock_quotes_cls):
        mock_instance = MagicMock()
        mock_quotes_cls.factory.return_value = mock_instance

        conn = TdxConnection()
        conn.initialize()
        assert conn.is_connected is True
        mock_quotes_cls.factory.assert_called_once_with(
            market="std", server=None, timeout=15, quiet=True
        )

    @patch("tdxdata.core.connection.Quotes")
    def test_initialize_with_server(self, mock_quotes_cls):
        mock_instance = MagicMock()
        mock_quotes_cls.factory.return_value = mock_instance

        conn = TdxConnection()
        server = ("127.0.0.1", 7709)
        conn.initialize(server=server, timeout=30)
        assert conn.is_connected is True
        mock_quotes_cls.factory.assert_called_once_with(
            market="std", server=server, timeout=30, quiet=True
        )

    @patch("tdxdata.core.connection.Quotes")
    def test_initialize_skip_if_already_connected(self, mock_quotes_cls):
        mock_instance = MagicMock()
        mock_quotes_cls.factory.return_value = mock_instance

        conn = TdxConnection()
        conn.initialize()
        call_count = mock_quotes_cls.factory.call_count

        conn.initialize()
        assert mock_quotes_cls.factory.call_count == call_count

    @patch("tdxdata.core.connection.Quotes")
    def test_close(self, mock_quotes_cls):
        mock_instance = MagicMock()
        mock_quotes_cls.factory.return_value = mock_instance

        conn = TdxConnection()
        conn.initialize()
        conn.close()
        assert conn.is_connected is False
        mock_instance.close.assert_called_once()

    def test_close_when_not_connected(self):
        conn = TdxConnection()
        conn.close()
        assert conn.is_connected is False

    @patch("tdxdata.core.connection.Quotes")
    def test_reconnect(self, mock_quotes_cls):
        mock_instance = MagicMock()
        mock_quotes_cls.factory.return_value = mock_instance

        conn = TdxConnection()
        conn.initialize()
        conn.reconnect()
        mock_instance.reconnect.assert_called_once()

    def test_reconnect_not_connected_raises(self):
        conn = TdxConnection()
        with pytest.raises(ConnectionError, match="Not connected"):
            conn.reconnect()

    @patch("tdxdata.core.connection.Quotes")
    def test_initialize_failure_raises(self, mock_quotes_cls):
        mock_quotes_cls.factory.side_effect = Exception("Connection refused")

        conn = TdxConnection()
        with pytest.raises(ConnectionError, match="Failed to initialize mootdx"):
            conn.initialize()
