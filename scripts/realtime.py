#!/usr/bin/env python3
"""
盘中实时行情补充模块。

在交易时段内，从 TDengine 查询当日最新 K 线（若已写入），
拼接到历史 DataFrame 末尾。

用法:
    from scripts.realtime import append_realtime_bars
    append_realtime_bars(codes, data_map, factor_map)
"""

import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from taosws import connect

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


_TRADING_START = (9, 25)
_TRADING_END = (15, 5)


def is_trading_hours() -> bool:
    """判断当前时间是否可能在 A 股交易时段。"""
    now = datetime.now()
    if now.weekday() >= 5:
        return False
    t = (now.hour, now.minute)
    return _TRADING_START <= t <= _TRADING_END


def _strip_prefix(code):
    """将 sh600000 / sz002741 转为纯数字代码。"""
    if code.startswith("sh") or code.startswith("sz") or code.startswith("bj"):
        return code[2:]
    return code


def _is_index(code):
    """判断是否为指数代码。"""
    return code.startswith("sh999") or code.startswith("sz399")


def append_realtime_bars(codes, data_map, factor_map):
    """为每只股票追加今日最新 K 线（从 TDengine 查询）。

    若 TDengine 中已有今日的 bar（收盘后或实时写入），则追加。
    指数代码跳过（TDengine 中无指数数据）。

    Args:
        codes: 股票代码列表 ['sh600000', 'sz002741', ...]
        data_map: {code: DataFrame} — 现有历史数据（会被原地修改）
        factor_map: {code: float} — 复权因子

    Returns:
        int — 追加了实时行情的股票数量
    """
    if not is_trading_hours():
        return 0

    valid_codes = [c for c in codes if c in data_map and data_map[c] is not None
                   and not _is_index(c)]
    if not valid_codes:
        return 0

    today = pd.Timestamp(datetime.now().date())
    today_str = today.strftime("%Y-%m-%d")
    appended = 0

    conn = connect()
    try:
        for orig_code in valid_codes:
            try:
                r = conn.query(
                    f"select ts, open, high, low, close, volume, amount "
                    f"from tdx.k_{orig_code}_1d "
                    f"where ts >= '{today_str}' order by ts desc limit 1"
                )
                rows = list(r)
                if not rows:
                    continue

                row = rows[0]
                ts = pd.Timestamp(row[0]).tz_localize(None)
                if ts.date() != today.date():
                    continue

                close_price = float(row[4])
                open_price = float(row[1])
                if close_price <= 0 or open_price <= 0:
                    continue

                volume = float(row[5])
                if pd.isna(volume) or volume <= 0:
                    continue

                high_price = float(row[2])
                low_price = float(row[3])
                amt = float(row[6])

                factor = factor_map.get(orig_code, 1.0)
                if factor > 0 and factor != 1.0:
                    open_price *= factor
                    high_price *= factor
                    low_price *= factor
                    close_price *= factor

                df = data_map[orig_code]
                bar = pd.DataFrame({
                    'open': [open_price],
                    'high': [high_price],
                    'low': [low_price],
                    'close': [close_price],
                    'vol': [volume],
                    'amt': [amt],
                }, index=pd.DatetimeIndex([today]))

                if len(df) > 0 and df.index[-1].date() == today.date():
                    df.iloc[-1] = bar.iloc[0]
                else:
                    data_map[orig_code] = pd.concat([df, bar])
                appended += 1
            except Exception:
                pass
    finally:
        conn.close()

    if appended > 0:
        print(f"  已追加 {appended}/{len(valid_codes)} 只股票的当日 K 线")
    return appended
