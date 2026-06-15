"""
prefetch_factors 复权因子批量获取模块单元测试。

测试覆盖（scripts/predict_stocks.py）:
    - _cache_month_key(): 月份键提取
    - prefetch_factors(): 一个月缓存命中 / 过期重取 / 获取失败回退旧缓存
    - 4 线程并发获取（ThreadPoolExecutor 实际调用）
    - 指数代码跳过
"""

import json
import os
import sys
import threading
from unittest.mock import patch

import pytest

# 确保项目根目录在 sys.path 中
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts import predict_stocks
from scripts.predict_stocks import (
    prefetch_factors, _cache_month_key,
    FACTOR_CACHE_1D, FACTOR_WORKERS,
)


@pytest.fixture
def tmp_cache_file(tmp_path, monkeypatch):
    """将因子缓存文件重定向到临时目录，避免污染真实缓存。"""
    cache_path = tmp_path / ".factor_1d.json"
    monkeypatch.setattr(predict_stocks, "FACTOR_CACHE_1D", str(cache_path))
    return str(cache_path)


# ======================================================================
# _cache_month_key
# ======================================================================

class TestCacheMonthKey:
    def test_normal_date(self):
        assert _cache_month_key("2026-06-15") == "2026-06"

    def test_first_day_of_month(self):
        assert _cache_month_key("2026-06-01") == "2026-06"

    def test_empty_string(self):
        assert _cache_month_key("") is None

    def test_non_string(self):
        assert _cache_month_key(None) is None


# ======================================================================
# prefetch_factors — 缓存命中（同月内有效）
# ======================================================================

class TestCacheHit:
    def test_same_month_cache_hit(self, tmp_cache_file):
        """同月内的缓存命中，不触发网络获取。"""
        today_str = str(__import__("datetime").datetime.now().date())
        # 写入当月缓存
        with open(tmp_cache_file, "w") as f:
            json.dump({
                "sh600353": {"factor": 10.71, "date": today_str},
            }, f)

        with patch("scripts.predict_stocks.derive_factor") as mock_derive:
            result = prefetch_factors(["sh600353"], {})
            mock_derive.assert_not_called()  # 缓存命中，不应调用网络

        factor, ok = result["sh600353"]
        assert factor == pytest.approx(10.71)
        assert ok is True

    def test_index_code_skipped(self, tmp_cache_file):
        """指数代码直接返回 1.0 且 ok=True，不查缓存也不获取。"""
        with patch("scripts.predict_stocks.derive_factor") as mock_derive:
            result = prefetch_factors(["sh999999", "sz399001"], {})
            mock_derive.assert_not_called()

        for code in ("sh999999", "sz399001"):
            factor, ok = result[code]
            assert factor == 1.0
            assert ok is True


# ======================================================================
# prefetch_factors — 缓存过期触发并发获取
# ======================================================================

class TestCacheExpiry:
    def test_last_month_cache_expired(self, tmp_cache_file):
        """上月缓存过期，触发 derive_factor 重新获取。"""
        import pandas as pd
        # 写入上月缓存（构造一个确定的非当月日期）
        with open(tmp_cache_file, "w") as f:
            json.dump({
                "sh600353": {"factor": 9.50, "date": "2000-01-01"},
            }, f)

        with patch("scripts.predict_stocks.get_data", return_value=pd.DataFrame({"close": [1.0]})), \
             patch("scripts.predict_stocks.derive_factor", return_value=10.71) as mock_derive:
            result = prefetch_factors(["sh600353"], {})

        mock_derive.assert_called_once()
        factor, ok = result["sh600353"]
        assert factor == pytest.approx(10.71)
        assert ok is True

        # 新值已写回缓存
        with open(tmp_cache_file) as f:
            cache = json.load(f)
        assert cache["sh600353"]["factor"] == pytest.approx(10.71)


# ======================================================================
# prefetch_factors — 获取失败回退旧缓存
# ======================================================================

class TestFallbackToOldCache:
    def test_failed_fetch_falls_back_to_old_cache(self, tmp_cache_file):
        """最新因子获取失败（返回 1.0）但存在旧缓存，应回退使用旧值。"""
        import pandas as pd
        with open(tmp_cache_file, "w") as f:
            json.dump({
                "sh600353": {"factor": 10.71, "date": "2000-01-01"},  # 过期但有值
            }, f)

        with patch("scripts.predict_stocks.get_data", return_value=pd.DataFrame({"close": [1.0]})), \
             patch("scripts.predict_stocks.derive_factor", return_value=1.0):  # 获取失败
            result = prefetch_factors(["sh600353"], {})

        factor, ok = result["sh600353"]
        assert factor == pytest.approx(10.71)  # 回退到旧缓存
        assert ok is True

    def test_failed_fetch_no_old_cache(self, tmp_cache_file):
        """获取失败且无旧缓存，返回 1.0，ok=False（输出后复权价格）。"""
        import pandas as pd
        with patch("scripts.predict_stocks.get_data", return_value=pd.DataFrame({"close": [1.0]})), \
             patch("scripts.predict_stocks.derive_factor", return_value=1.0):
            result = prefetch_factors(["sh600353"], {})

        factor, ok = result["sh600353"]
        assert factor == 1.0
        assert ok is False


# ======================================================================
# prefetch_factors — 4 线程并发
# ======================================================================

class TestConcurrency:
    def test_thread_pool_invoked_with_4_workers(self, tmp_cache_file):
        """确认 ThreadPoolExecutor 以 max_workers=4 被调用。"""
        from concurrent.futures import ThreadPoolExecutor
        import pandas as pd

        captured = {}

        class _Spy(ThreadPoolExecutor):
            """记录 max_workers 后委托给真实 ThreadPoolExecutor。"""
            def __init__(self, *args, **kwargs):
                captured["max_workers"] = kwargs.get("max_workers")
                super().__init__(*args, **kwargs)

        codes = [f"sh60000{i}" for i in range(5)]
        # mock get_data 返回非空 df，确保走到 derive_factor 分支
        with patch("scripts.predict_stocks.get_data", return_value=pd.DataFrame({"close": [1.0]})), \
             patch("scripts.predict_stocks.derive_factor", return_value=2.0):
            with patch("scripts.predict_stocks.ThreadPoolExecutor", _Spy):
                result = prefetch_factors(codes, {})

        assert captured["max_workers"] == FACTOR_WORKERS == 4
        assert len(result) == len(codes)
        for code in codes:
            assert result[code] == (2.0, True)

    def test_actual_concurrency(self, tmp_cache_file):
        """多只股票在并发线程中处理（验证非串行）。"""
        import pandas as pd
        active = 0
        max_active = 0
        lock = threading.Lock()

        def slow_derive(code, df, verbose=True):
            nonlocal active, max_active
            with lock:
                active += 1
                max_active = max(max_active, active)
            import time
            time.sleep(0.05)
            with lock:
                active -= 1
            return 2.0

        codes = ["sh600000", "sz000001", "sh600353", "sz002741"]
        with patch("scripts.predict_stocks.get_data", return_value=pd.DataFrame({"close": [1.0]})), \
             patch("scripts.predict_stocks.derive_factor", side_effect=slow_derive):
            result = prefetch_factors(codes, {})

        # 并发时峰值活跃数应 > 1（串行则恒为 1）
        assert max_active > 1, f"未观察到并发（max_active={max_active}）"
        assert len(result) == len(codes)

    def test_multiple_codes_all_fetched(self, tmp_cache_file):
        """多只股票并发获取，结果完整返回。"""
        import pandas as pd
        codes = ["sh600000", "sz000001", "sh600353", "sz002741", "sh601318"]
        with patch("scripts.predict_stocks.get_data", return_value=pd.DataFrame({"close": [1.0]})), \
             patch("scripts.predict_stocks.derive_factor", return_value=3.14):
            result = prefetch_factors(codes, {})

        assert set(result.keys()) == set(codes)
        for code in codes:
            factor, ok = result[code]
            assert factor == pytest.approx(3.14)
            assert ok is True

    def test_worker_exception_does_not_crash(self, tmp_cache_file):
        """单个 worker 抛异常不应导致整体崩溃，应回退为 1.0。"""
        import pandas as pd

        def boom(code, df, verbose=True):
            raise RuntimeError("网络错误")

        with patch("scripts.predict_stocks.get_data", return_value=pd.DataFrame({"close": [1.0]})), \
             patch("scripts.predict_stocks.derive_factor", side_effect=boom):
            result = prefetch_factors(["sh600353"], {})

        factor, ok = result["sh600353"]
        assert factor == 1.0
        assert ok is False
