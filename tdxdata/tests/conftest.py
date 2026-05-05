import pytest
from unittest.mock import MagicMock
import pandas as pd
import numpy as np
from datetime import datetime


@pytest.fixture
def mock_client():
    client = MagicMock()
    client.close = MagicMock()
    client.reconnect = MagicMock()
    return client


@pytest.fixture
def sample_kline_df():
    dates = pd.date_range("2024-01-01", periods=5, freq="B")
    return pd.DataFrame({
        "open": [100.0, 101.0, 102.0, 103.0, 104.0],
        "high": [105.0, 106.0, 107.0, 108.0, 109.0],
        "low": [99.0, 100.0, 101.0, 102.0, 103.0],
        "close": [104.0, 105.0, 106.0, 107.0, 108.0],
        "vol": [10000, 11000, 12000, 13000, 14000],
        "amount": [1000000, 1100000, 1200000, 1300000, 1400000],
        "date": dates,
    })


@pytest.fixture
def mock_connection(mock_client):
    from tdxdata.core.connection import TdxConnection
    conn = TdxConnection()
    conn._client = mock_client
    conn._initialized = True
    return conn


@pytest.fixture
def tmp_output_dir(tmp_path):
    return str(tmp_path)
